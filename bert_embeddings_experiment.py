# %%
from typing import List
from utils import load_pkl, save_pkl
import torch
import transformers
model = transformers.BertModel.from_pretrained('bert-base-uncased')
model.eval()
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

names = ["ruins", "simple_beach", "desert", "plains", "swamp", "vanilla_village", "vanilla_mineshaft"]

for name in names:
    prepath = f"/home/awiszus/Project/World-GAN/input/minecraft/{name}/"
    token_dict = load_pkl(
        "representations", prepath)
    token_list = list(token_dict.keys())
    token_names: List[str] = [token.replace("minecraft:", "").replace("_", " ")
                              for token in token_list]

    natural_token_dict = {}
    with torch.no_grad():
        for token_name, token in zip(token_names, token_list):
            ids = tokenizer.encode(token_name)
            tokens = tokenizer.convert_ids_to_tokens(ids)
            bert_output = model.forward(torch.tensor(
                ids).unsqueeze(0), encoder_hidden_states=True)
            final_layer_embeddings = bert_output[0][-1]
            natural_token_dict[token] = final_layer_embeddings[0]

    save_pkl(natural_token_dict, "natural_representations", prepath)

    #%%
    natural_tokens = torch.stack(list(natural_token_dict.values()))
    # %%
    import pymde
    import matplotlib.pyplot as plt
    from adjustText import adjust_text


    fig = plt.figure()
    embedding = pymde.preserve_distances(natural_tokens, embedding_dim=3, verbose=True).embed()
    ax = pymde.plot(embedding, marker_size=5)
    annotations = []
    for i, token_name in enumerate(token_names):
        annotations.append(ax.annotate(token_name, (embedding[i, 0], embedding[i, 1])))
    adjust_text(annotations)

    # %%
    natural_token_dict_small = {}
    for token_name, e in zip(token_list, embedding):
        natural_token_dict_small[token_name] = e
    save_pkl(natural_token_dict_small, "natural_representations_small", prepath)
    # %%
