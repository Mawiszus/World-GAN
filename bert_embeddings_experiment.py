# %%
import os
from typing import List
from utils import load_pkl, save_pkl
import torch
import transformers
import inflect
from minecraft.block_grammar import get_sentence
inflect = inflect.engine()
model_str = 'bert-base-uncased'
model = transformers.BertModel.from_pretrained(model_str)
model.eval()
tokenizer = transformers.BertTokenizer.from_pretrained(model_str)
unmasker = transformers.pipeline('fill-mask', model=model_str)

# names = ["ruins", "simple_beach", "desert", "plains", "swamp", "vanilla_village", "vanilla_mineshaft"]
names = ["ruins"]
# names = ["vanilla_village"]

for name in names:
    prepath = f"/home/awiszus/Project/World-GAN/input/minecraft/{name}/"
    token_dict = load_pkl(
        "representations", prepath)
    token_list = list(token_dict.keys())
    token_names: List[str] = []
    clean_names: List[str] = []
    for token in token_list:
        clean_token = token.replace("minecraft:", "").replace("_", " ").replace("chest", "treasure chest")
        if clean_token.replace("block", "") != clean_token:
            clean_token = "patch of " + clean_token.replace(" block", "")
        # apparently grass is counted as plural by inflect but it isn't so extra check
        if isinstance(inflect.singular_noun(clean_token), bool) or (clean_token.find("grass") >= 0):
            is_plural = False
            # token_names.append(f"This {clean_token} is part this village.")
        else:
            is_plural = True
            # token_names.append(f"These {clean_token} are part of this village.")
        token_names.append(get_sentence(clean_token, is_plural, "ruins", True, unmasker))
        clean_names.append(clean_token)

    # token_names: List[str] = [token.replace("minecraft:", "").replace("_", " ").replace("chest", "treasure chest")
    #                           for token in token_list]

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
    embedding = pymde.preserve_distances(natural_tokens, embedding_dim=8, verbose=True).embed()
    ax = pymde.plot(embedding, marker_size=5)
    annotations = []
    for i, token_name in enumerate(token_names):
        annotations.append(ax.annotate(token_name, (embedding[i, 0], embedding[i, 1])))
    adjust_text(annotations)
    plt.savefig(os.path.join(prepath, "mde-plot.png"))

    # %%
    natural_token_dict_small = {}
    for token_name, e in zip(token_list, embedding):
        natural_token_dict_small[token_name] = e
    save_pkl(natural_token_dict_small, "natural_representations_small", prepath)
    # %%
