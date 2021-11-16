from utils import load_pkl
import torch
import transformers
import pymde
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Read new sentence
new_sentence = "valuable"
use_natural_rep = False
use_naive = True

# Load embeddings
repr_dim = 32
input_area_name = "ruins"
block2repr = load_pkl(f"natural_representations_small_no_norm_{repr_dim}{'_naive' if use_naive else ''}",
                      f"/home/awiszus/Project/World-GAN/input/minecraft/{input_area_name}/")
embeddings = torch.stack(list(block2repr.values()))
embedding_dim = embeddings.shape[1]

# Load original representations
block2natural = load_pkl(f"natural_representations{'_naive' if use_naive else ''}",
                         f"/home/awiszus/Project/World-GAN/input/minecraft/{input_area_name}/")
nat_embeddings = torch.stack(list(block2natural.values()))

# Encode new sentence
model_str = 'bert-base-uncased'
model = transformers.BertModel.from_pretrained(model_str)
model.eval()
tokenizer = transformers.BertTokenizer.from_pretrained(model_str)

ids = tokenizer.encode(new_sentence)
tokens = tokenizer.convert_ids_to_tokens(ids)
bert_output = model.forward(torch.tensor(
    ids).unsqueeze(0), encoder_hidden_states=True)
final_layer_embeddings = bert_output[0][-1].detach()

# Make incremental mde
anchor_constraint = pymde.Anchored(
    anchors=torch.arange(embeddings.shape[0]),
    values=embeddings,
)
incremental_mde = pymde.preserve_distances(
    torch.cat([nat_embeddings, final_layer_embeddings[0].unsqueeze(0)], dim=0),
    constraint=anchor_constraint,
    embedding_dim=embedding_dim,
    verbose=True)

inc_embed = incremental_mde.embed()
if not use_natural_rep:
    new_embedded_vec = inc_embed[-1]  # / torch.norm(inc_embed[-1], p=2)
else:
    new_embedded_vec = final_layer_embeddings[0].unsqueeze(0)

# Calculate distances to old embeddings
dists = torch.zeros(embeddings.shape[0])
for i in range(embeddings.shape[0]):
    if not use_natural_rep:
        dists[i] = torch.norm(embeddings[i] / torch.norm(embeddings[i], p=2) -
                              new_embedded_vec / torch.norm(new_embedded_vec, p=2))
    else:
        dists[i] = torch.norm(nat_embeddings[i] - new_embedded_vec)

keys = list(block2repr.keys())
df = pd.DataFrame()
dist_dict = {}
for i, d in enumerate(dists):
    dist_dict[keys[i].replace('minecraft:', '')] = d.item()
    print(f"{keys[i].replace('minecraft:', '')} : {d}")

df = df.append(dist_dict, ignore_index=True)

print(f"Closest token to sentence is: Nr. {dists.argmin()} - {list(block2repr.keys())[dists.argmin()]}")

# Show results (in plot?)
palette = "turbo"
# Seaborn:
plt.figure(figsize=(15, 8))

sorted_names = sorted(dist_dict, key=dist_dict.get, reverse=False)
p = sns.barplot(data=df, palette=palette, order=sorted_names)
plt.title(f'Distances to "{new_sentence}"')
plt.xticks(rotation=30, ha="right")
plt.show()
