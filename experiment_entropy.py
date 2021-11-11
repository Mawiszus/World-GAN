import numpy as np
import torch
from scipy.stats import entropy
from itertools import islice
import os
import pandas as pd
from statistics import mean, variance
from tqdm import tqdm
from config import Config
from minecraft.level_utils import read_level as mc_read_level
from minecraft.level_utils import one_hot_to_blockdata_level
from generate_minecraft_samples import sub_coord_dict


class GenerateEntropyConfig(Config):
    folder: str = None  # folder containing torch blockdata
    not_cuda: bool = False  # disables cuda

    def process_args(self):
        super().process_args()


def entropy1(x):
    # Can do 2D but only up to dim=64
    counts = np.histogramdd(x)[0]
    dist = counts / np.sum(counts)
    # dist = x
    logs = np.log2(np.where(dist > 0, dist, 1))
    return -np.sum(dist * logs)


def window(seq, n=2):
    """Sliding window width n from seq.  From old itertools recipes."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def get_transition_matrix(sample, dim=0):
    if dim == 0:
        i_max = sample.shape[1]
        j_max = sample.shape[2]
    elif dim == 1:
        i_max = sample.shape[0]
        j_max = sample.shape[2]
    elif dim == 2:
        i_max = sample.shape[0]
        j_max = sample.shape[1]
    else:
        raise NotImplementedError("This is only implemented for 3 dims because I'm lazy to look up how to do this.")

    pairs_list = []
    for i in range(i_max):
        for j in range(j_max):
            if dim == 0:
                pairs_list += list(window(sample[:, i, j]))
            elif dim == 1:
                pairs_list += list(window(sample[i, :, j]))
            elif dim == 2:
                pairs_list += list(window(sample[i, j, :]))

    pairs = pd.DataFrame(pairs_list, columns=['state1', 'state2'])
    counts = pairs.groupby('state1')['state2'].value_counts()
    df = (counts / counts.sum()).unstack()
    df = df.fillna(0)
    return df


if __name__ == '__main__':
    # config
    opt = GenerateEntropyConfig().parse_args()
    opt.sub_coords = sub_coord_dict[opt.input_area_name]
    opt.process_args()

    opt.game = 'minecraft'
    opt.ImgGen = None
    replace_tokens = None

    # Load Real
    real = mc_read_level(opt)
    opt.level_shape = real.shape[2:]

    np_real = one_hot_to_blockdata_level(real, None, opt.block2repr, opt.repr_type).numpy()

    t_mat_x = get_transition_matrix(np_real, 0)
    t_mat_y = get_transition_matrix(np_real, 1)
    t_mat_z = get_transition_matrix(np_real, 2)

    ent_real_0 = entropy(t_mat_x, base=None).mean()
    ent_real_1 = entropy(t_mat_y, base=None).mean()
    ent_real_2 = entropy(t_mat_z, base=None).mean()

    print(
        f"Entropy of the original sample: ({ent_real_0:.3f}, {ent_real_1:.3f}, {ent_real_2:.3f})")

    torch.save([ent_real_0, ent_real_1, ent_real_2], os.path.join(opt.folder, "../real_entropy.pt"))

    # Fake
    ent_0 = []
    ent_1 = []
    ent_2 = []
    for filename in tqdm(os.listdir(opt.folder)):
        if filename.endswith(".pt"):
            fake = torch.load(os.path.join(opt.folder, filename))

            t_fake_x = get_transition_matrix(fake, 0)
            t_fake_y = get_transition_matrix(fake, 1)
            t_fake_z = get_transition_matrix(fake, 2)

            ent_fake_0 = entropy(t_fake_x, base=None).mean()
            ent_fake_1 = entropy(t_fake_y, base=None).mean()
            ent_fake_2 = entropy(t_fake_z, base=None).mean()

            ent_0.append(ent_fake_0)
            ent_1.append(ent_fake_1)
            ent_2.append(ent_fake_2)

            #print(
                # f"Entropy of the fake sample: ({ent_fake_0.mean():.3f}, {ent_fake_1.mean():.3f}, {ent_fake_2.mean():.3f})")

        else:
            continue

    print(f"Overall fake Entropy mean: ({mean(ent_0):.3f}, {mean(ent_1):.3f}, {mean(ent_2):.3f})")
    print(f"Overall fake Entropy vari: ({variance(ent_0):.3f}, {variance(ent_1):.3f}, {variance(ent_2):.3f})")

    torch.save([mean(ent_0), mean(ent_1), mean(ent_2)], os.path.join(opt.folder, "../mean_entropy.pt"))
    torch.save([variance(ent_0), variance(ent_1), variance(ent_2)], os.path.join(opt.folder, "../var_entropy.pt"))



