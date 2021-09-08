import numpy as np

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
