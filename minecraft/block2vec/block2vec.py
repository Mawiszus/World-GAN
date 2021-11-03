import math
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from fuzzywuzzy import process
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from minecraft.block2vec.block2vec_dataset import Block2VecDataset
from minecraft.block2vec.image_annotations_3d import ImageAnnotations3D
from minecraft.block2vec.skip_gram_model import SkipGramModel
from sklearn.metrics import ConfusionMatrixDisplay
from tap import Tap
from torch.utils.data import DataLoader
from utils import load_pkl
import umap.umap_ as umap


sub_coord_dict = dict(
    ruins=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    simple_beach=[0.0, 0.5, 0.0, 1.0, 0.0, 1.0],
    desert=[0.25, 0.75, 0.0, 1.0, 0.25, 0.75],
    plains=[0.25, 0.75, 0.0, 1.0, 0.25, 0.75],
    swamp=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    vanilla_village=[0.33333, 0.66667, 0.0, 1.0, 0.33333, 0.66667],
    vanilla_mineshaft=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
)


class Block2VecArgs(Tap):
    input_world_path: str = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "minecraft_worlds",
            "Drehmal v2.1 PRIMORDIAL",
        )
    )
    output_path: str = os.path.join("output", "block2vec")
    emb_dimension: int = 32
    epochs: int = 30
    batch_size: int = 256
    initial_lr: float = 1e-3
    input_area_name: str = "ruins"
    sub_coords: List[float] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]  # defines which coords of the full coord are are
    cutout_coords: bool = True
    neighbor_radius: int = 1

    def process_args(self) -> None:
        coords = load_pkl(
            "/primordial_coords_dict",
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "..",
                "..",
                "input",
                "minecraft",
            ),
        )
        self.sub_coords = sub_coord_dict[self.input_area_name]
        self.coords = []
        tmp_coords = coords[self.input_area_name]
        sub_coords = [(self.sub_coords[0], self.sub_coords[1]),
                      (self.sub_coords[2], self.sub_coords[3]),
                      (self.sub_coords[4], self.sub_coords[5])]
        for i, (start, end) in enumerate(sub_coords):
            curr_len = tmp_coords[i][1] - tmp_coords[i][0]
            if isinstance(start, float):
                tmp_start = curr_len * start + tmp_coords[i][0]
                tmp_end = curr_len * end + tmp_coords[i][0]
            elif isinstance(start, int):
                tmp_start = tmp_coords[i][0] + start
                tmp_end = tmp_coords[i][0] + end
            else:
                AttributeError("Unexpected type for sub_coords")
                tmp_start = tmp_coords[i][0]
                tmp_end = tmp_coords[i][1]

            self.coords.append((int(tmp_start), int(tmp_end)))
        # self.input_world_coords = coords[self.input_area_name]
        self.output_path = os.path.join(
            self.output_path, self.input_area_name)
        logger.info(self.output_path)


class Block2Vec(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.args: Block2VecArgs = Block2VecArgs().from_dict(kwargs)
        self.save_hyperparameters()
        self.dataset = Block2VecDataset(
            self.args.input_world_path,
            coords=self.args.coords,
            cutout_coords=self.args.cutout_coords,
            neighbor_radius=self.args.neighbor_radius,
        )
        self.emb_size = len(self.dataset.block2idx)
        self.model = SkipGramModel(self.emb_size, self.args.emb_dimension)
        self.textures = dict()
        self.learning_rate = self.args.initial_lr

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def training_step(self, batch, *args, **kwargs):
        loss = self.forward(*batch)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            math.ceil(len(self.dataset) / self.args.batch_size) *
            self.args.epochs,
        )
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count() or 1,
        )

    def on_epoch_end(self):
        embedding_dict = self.save_embedding(
            self.dataset.idx2block, self.args.output_path
        )
        self.create_confusion_matrix(
            self.dataset.idx2block, self.args.output_path)
        self.plot_embeddings(embedding_dict, self.args.output_path)

    def read_texture(self, block: str):
        if block not in self.textures and block != "air":
            texture_candidates = Path(
                "/home/schubert/projects/TOAD-GAN/minecraft/block2vec/textures"
            ).glob("*.png")
            # use of absolute path is intentional
            match = process.extractOne(block, texture_candidates)
            if match is not None:
                logger.info("Matches {} with {} texture file", block, match[0])
                self.textures[block] = plt.imread(match[0])
        if block not in self.textures:
            self.textures[block] = np.ones(shape=[16, 16, 3])
        return self.textures[block]

    def save_embedding(self, id2block: Dict[int, str], output_path: str):
        embeddings = self.model.target_embeddings.weight
        # embeddings = embeddings / torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        embeddings = embeddings.cpu().data.numpy()
        embedding_dict = {}
        with open(os.path.join(output_path, "embeddings.txt"), "w") as f:
            f.write("%d %d\n" % (len(id2block), self.args.emb_dimension))
            for wid, w in id2block.items():
                e = " ".join(map(lambda x: str(x), embeddings[wid]))
                embedding_dict[w] = torch.from_numpy(embeddings[wid])
                f.write("%s %s\n" % (w, e))
        np.save(os.path.join(output_path, "embeddings.npy"), embeddings)
        with open(os.path.join(output_path, f"representations.pkl"), "wb") as f:
            pickle.dump(embedding_dict, f)
        return embedding_dict

    def plot_embeddings(self, embedding_dict: Dict[str, np.ndarray], output_path: str):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        legend = [label.replace("minecraft:", "")
                  for label in embedding_dict.keys()]
        texture_imgs = [self.read_texture(block) for block in legend]
        embeddings = torch.stack(list(embedding_dict.values())).numpy()
        if embeddings.shape[-1] != 3:
            embeddings_3d = umap.UMAP(
                n_neighbors=5, min_dist=0.3, n_components=3
            ).fit_transform(embeddings)
        else:
            embeddings_3d = embeddings
        for embedding in embeddings_3d:
            ax.scatter(*embedding, alpha=0)
        ia = ImageAnnotations3D(embeddings_3d, texture_imgs, legend, ax, fig)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "scatter_3d.png"), dpi=300)
        plt.close("all")

    def create_confusion_matrix(self, id2block: Dict[int, str], output_path: str):
        rcParams.update({"font.size": 6})
        names = []
        dists = np.zeros((len(id2block), len(id2block)))
        for i, b1 in id2block.items():
            names.append(b1.split(":")[1])
            for j, b2 in id2block.items():
                dists[i, j] = F.mse_loss(
                    self.model.target_embeddings.weight.data[i],
                    self.model.target_embeddings.weight.data[j],
                )
        confusion_display = ConfusionMatrixDisplay(dists, display_labels=names)
        confusion_display.plot(include_values=False,
                               xticks_rotation="vertical")
        confusion_display.ax_.set_xlabel("")
        confusion_display.ax_.set_ylabel("")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "dist_matrix.png"))
        plt.close()
