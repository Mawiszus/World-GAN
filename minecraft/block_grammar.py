import numpy as np
import torch

N_SENTENCES = 6  # len(sentence_mid)


def get_sentence(block_name, is_block_plural, world_name, is_world_plural, unmasker):
    if is_block_plural:
        sentence_start = f"These {block_name} are "
    else:
        sentence_start = f"This {block_name} is "

    if is_world_plural:
        sentence_end = f" these {world_name}."
    else:
        sentence_end = f" this {world_name}."

    sentences = [
        sentence_start + "[MASK]" + sentence_end,
        sentence_start + "[MASK] of" + sentence_end,
        sentence_start + "[MASK] in" + sentence_end,
        sentence_start + "[MASK] to" + sentence_end,
    ]

    results = []
    scores = np.zeros((len(sentences),))
    for i, sentence in enumerate(sentences):
        result = unmasker(sentence, top_k=1)[0]
        results.append(result)
        scores[i] = result["score"]

    sentence = results[scores.argmax()]["sequence"]

    # sentence_mid = [
    #     "part of",
    #     "hidden in",
    #     "on top of",
    #     "below",
    #     "next to"
    # ]

    # sentence = sentence_start + sentence_mid[n_sentence] + sentence_end
    return sentence


def get_neighbor_sentence(block_name, is_block_plural, world_name, is_world_plural, neighbors, n_plurals, unmasker):
    if is_block_plural:
        sentence_start = f"These {block_name} are "
    else:
        sentence_start = f"This {block_name} is "

    sentence_mid = "surrounded by"
    if len(neighbors) > 1:
        for i, n in enumerate(neighbors[:-1]):
            if n_plurals[i]:
                sentence_mid = sentence_mid + f" {n},"
            else:
                sentence_mid = sentence_mid + f" a {n},"
        if n_plurals[-1]:
            sentence_mid = sentence_mid + f" and {neighbors[-1]}. It is "
        else:
            sentence_mid = sentence_mid + f" and a {neighbors[-1]}. It is "
    else:
        if neighbors[0] == block_name:
            # surrounded by only itself
            sentence_mid = ""
        else:
            # only one other block in neighborhood
            if n_plurals[0]:
                sentence_mid = f"next to {neighbors[0]}. It is "
            else:
                sentence_mid = f"next to a {neighbors[0]}. It is "

    if is_world_plural:
        sentence_end = f" these {world_name}."
    else:
        sentence_end = f" this {world_name}."

    sentences = [
        sentence_start + sentence_mid + "[MASK]" + sentence_end,
        sentence_start + sentence_mid + "[MASK] of" + sentence_end,
        sentence_start + sentence_mid + "[MASK] in" + sentence_end,
        sentence_start + sentence_mid + "[MASK] to" + sentence_end,
    ]

    results = []
    scores = np.zeros((len(sentences),))
    for i, sentence in enumerate(sentences):
        result = unmasker(sentence, top_k=1)[0]
        results.append(result)
        scores[i] = result["score"]

    sentence = results[scores.argmax()]["sequence"]
    return sentence



