# %%
import os
from pathlib import Path
from typing import List
from config import Config

from minecraft.level_utils import read_level, read_level_from_file
from utils import load_pkl, save_pkl
import torch
import transformers
import inflect
from minecraft.block_grammar import get_sentence, get_neighbor_sentence

sub_coord_dict = dict(
    ruins=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    simple_beach=[0.0, 0.5, 0.0, 1.0, 0.0, 1.0],
    desert=[0.25, 0.75, 0.0, 1.0, 0.25, 0.75],
    plains=[0.25, 0.75, 0.0, 1.0, 0.25, 0.75],
    swamp=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    vanilla_village=[0.33333, 0.66667, 0.0, 1.0, 0.33333, 0.66667],
    vanilla_mineshaft=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
)

if __name__ == '__main__':
    use_neighbors = False
    use_naive = False
    embedding_dim = 32
    inflect = inflect.engine()
    model_str = 'bert-base-uncased'
    model = transformers.BertModel.from_pretrained(model_str)
    model.eval()
    tokenizer = transformers.BertTokenizer.from_pretrained(model_str)
    unmasker = transformers.pipeline('fill-mask', model=model_str)

    names = ["ruins", "simple_beach", "desert", "plains", "swamp",
             "vanilla_village", "vanilla_mineshaft"]
    world_names_and_pl = [("ruins", True), ("beach", False), ("desert", False), ("plains", True), ("swamp", False),
                          ("village", False), ("mineshaft", False)]
    # names = ["ruins"]
    # world_names_and_pl = [("ruins", True)]
    # names = ["vanilla_village"]
    # world_names_and_pl = [("village", False)]

    for n, name in enumerate(names):
        prepath = f"/home/awiszus/Project/World-GAN/input/minecraft/{name}/"
        # prepath = f"/home/schubert/projects/World-GAN/input/minecraft/{name}/"
        print(name)
        if not use_neighbors:
            opt = Config().parse_args()
            opt.input_area_name = name
            opt.sub_coords = sub_coord_dict[name]
            opt.process_args()
            print(opt)
            level = read_level(opt)
            token_list = opt.token_list
            # token_dict = load_pkl("representations", prepath)
            # token_list = list(token_dict.keys())
            # dummy to make zip work with both use_neighbors and not
            token_list_vectors = token_list
        else:
            token_dict = load_pkl("representations", prepath)
            old_list = list(token_dict.keys())
            token_list_vectors = torch.load(
                prepath + "neighbor_token_list.pth")
            token_list = torch.load(
                prepath + "neighbor_token_list_translation.pth")
            token_list_str = []
            for vec in token_list_vectors:
                token_list_str.append(str(vec))

        token_names: List[str] = []
        clean_names: List[str] = []
        for token, token_vector in zip(token_list, token_list_vectors):
            clean_token = token.replace("minecraft:", "").replace(
                "_", " ")
            # apparently grass is counted as plural by inflect but it isn't so extra check
            if isinstance(inflect.singular_noun(clean_token), bool) or (clean_token.find("grass") >= 0):
                is_plural = False
                # token_names.append(f"This {clean_token} is part this village.")
            else:
                is_plural = True
                # token_names.append(f"These {clean_token} are part of this village.")
            if not use_neighbors:
                token_names.append(get_sentence(clean_token, is_plural, world_names_and_pl[n][0],
                                                world_names_and_pl[n][1], unmasker))
            else:
                neighbors = []
                plurals = []
                for i in range(token_vector.shape[0]):
                    if token_vector[i] > 0:
                        tok = old_list[i]
                        n_token = tok.replace("minecraft:", "").replace(
                            "_", " ").replace("chest", "treasure chest")
                        if n_token.replace("block", "") != n_token:
                            n_token = "patch of " + \
                                n_token.replace(" block", "")
                        # apparently grass is counted as plural by inflect but it isn't so extra check
                        if isinstance(inflect.singular_noun(clean_token), bool) or (clean_token.find("grass") >= 0):
                            plurals.append(False)
                        else:
                            plurals.append(True)
                        neighbors.append(n_token)
                if not neighbors:
                    neighbors.append(clean_token)
                token_names.append(get_neighbor_sentence(clean_token, is_plural, world_names_and_pl[n][0],
                                                         world_names_and_pl[n][1], neighbors, plurals, unmasker))
            clean_names.append(clean_token)

        print(clean_names)
        print(token_names)
        # token_names: List[str] = [token.replace("minecraft:", "").replace("_", " ").replace("chest", "treasure chest")
        #                           for token in token_list]

        natural_token_dict = {}
        if not use_neighbors:
            t_list = token_list
        else:
            t_list = token_list_str

        with torch.no_grad():
            for token_name, token in zip(token_names if not use_naive else clean_names, t_list):
                ids = tokenizer.encode(token_name)
                tokens = tokenizer.convert_ids_to_tokens(ids)
                bert_output = model.forward(torch.tensor(
                    ids).unsqueeze(0), encoder_hidden_states=True)
                final_layer_embeddings = bert_output[0][-1]
                natural_token_dict[token] = final_layer_embeddings[0]

        if not use_neighbors:
            save_pkl(natural_token_dict, f"natural_representations{'_naive' if use_naive else ''}", prepath)
        else:
            save_pkl(natural_token_dict,
                     "natural_representations_neighbors", prepath)

        # %%
        natural_tokens = torch.stack(list(natural_token_dict.values()))
        # %%
        import pymde
        import matplotlib.pyplot as plt
        from adjustText import adjust_text

        fig = plt.figure()
        embedding = pymde.preserve_distances(
            natural_tokens, embedding_dim=embedding_dim, verbose=True).embed()
        ax = pymde.plot(embedding, marker_size=5)
        annotations = []
        for i, token_name in enumerate(token_names):
            annotations.append(ax.annotate(
                token_name, (embedding[i, 0], embedding[i, 1])))
        adjust_text(annotations)
        plt.savefig(os.path.join(prepath, "mde-plot.png"))

        # %%
        natural_token_dict_small_no_norm = {}
        natural_token_dict_small = {}
        for token_name, e in zip(t_list, embedding):
            natural_token_dict_small_no_norm[token_name] = e
            natural_token_dict_small[token_name] = e / torch.norm(e, p=2)
        if not use_neighbors:
            save_pkl(natural_token_dict_small_no_norm,
                     f"natural_representations_small_no_norm_{embedding_dim}{'_naive' if use_naive else ''}", prepath)
            save_pkl(natural_token_dict_small,
                     f"natural_representations_small_{embedding_dim}{'_naive' if use_naive else ''}", prepath)
        else:
            save_pkl(natural_token_dict_small,
                     "natural_representations_small_neighbors", prepath)
        # %%
