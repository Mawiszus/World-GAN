import os
from tap import Tap
import json
from torch import load as torch_load
import pandas as pd
import numpy as np


class ResultConfig(Tap):
    # name of the folder containing multiple evaluated runs
    top_folder: str = "/home/schubert/projects/World-GAN/output/ablation"

    def __init__(self, *args, underscores_to_dashes: bool = False, explicit_bool: bool = False, **kwargs):
        super().__init__(*args, underscores_to_dashes, explicit_bool, **kwargs)


if __name__ == '__main__':
    cfg = ResultConfig().parse_args()

    variant_names = {
        "bert_naive_8": "$\\text{World-GAN}$",
        "bert_8": "$\\text{World-GAN}^{p}$",
        "bert_naive_32": "$\\text{World-GAN}$",
        "bert_32": "$\\text{World-GAN}^{p}$",
    }

    entropy_orig = {
        "desert": 0.38,
        "plains": 0.70,
        "ruins": 1.12,
        "beach": 0.61,
        "swamp": 1.05,
        "mine shaft": 0.85,
        "village": 0.62
    }

    df = pd.DataFrame()
    for variant in os.listdir(cfg.top_folder):
        print(variant)
        if not os.path.isdir(os.path.join(cfg.top_folder, variant)):
            continue
        for d in os.listdir(os.path.join(cfg.top_folder, variant)):
            if os.path.isdir(os.path.join(cfg.top_folder, variant, d)):
                curr_files = os.path.join(cfg.top_folder, variant, d, "files")
                try:
                    with open(os.path.join(curr_files, "wandb-metadata.json")) as jsonfile:
                        metadata = json.load(jsonfile)
                except Exception:
                    print(f"No metadata for {d}")
                    continue
                name = metadata["args"][-8].replace("_",
                                                    " ").replace("vanilla ", "").replace("simple ", "")
                if name == "mineshaft":
                    name = "mine shaft"
                try:
                    with open(os.path.join(curr_files, "random_samples", "results.json")) as jsonfile:
                        results = json.load(jsonfile)
                        reshape_results = {"TPKL-Div": np.mean(list(results["tpkldiv"]["mean"].values())),
                                           "Levenshtein": results["levenshtein"]["mean"]}
                except Exception:
                    print(f"No results for {d}")
                    reshape_results = {}
                try:
                    result_entropy = torch_load(os.path.join(
                        curr_files, "random_samples", "mean_entropy.pt"))
                    reshape_results["Entropy"] = sum(
                        result_entropy)/len(result_entropy)
                except Exception:
                    print(f"No entropy for {d}")
                reshape_results["Variant"] = variant_names[variant]
                reshape_results["Structure"] = name
                reshape_results["Entropy (orig.)"] = entropy_orig[name]
                if "32" not in variant:
                    continue

                df = pd.concat(
                    [df, pd.DataFrame(reshape_results, index=[name])])

    df = df.sort_index()
    df1 = df.pivot_table(["TPKL-Div", "Levenshtein"], ["Structure", "Variant"])
    print(df.pivot_table(["TPKL-Div", "Levenshtein"], ["Variant"], aggfunc=np.mean).to_latex(float_format=lambda x: '%10.2f' % x, escape=False))
    df2 = df.pivot_table(
        ["Entropy"], ["Structure", "Entropy (orig.)"], "Variant")
    print(df1.to_latex(float_format=lambda x: '%10.2f' % x, escape=False))
    print("-----------------")
    print(df.pivot_table(["Entropy"], [], "Variant", aggfunc=np.mean).to_latex(float_format=lambda x: '%10.2f' % x, escape=False))
    print(df2.to_latex(
        float_format=lambda x: '%10.2f' % x, escape=False))
    with open(os.path.join(cfg.top_folder, "tabular_results.tex"), "w") as txt_file:
        txt_file.write(df.to_latex(float_format=lambda x: '%10.2f' % x))
