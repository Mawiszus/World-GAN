import os
from tap import Tap
import json
from torch import load as torch_load
import pandas as pd

class ResultConfig(Tap):
    top_folder: str  # name of the folder containing multiple evaluated runs

    def __init__(self, *args, underscores_to_dashes: bool = False, explicit_bool: bool = False, **kwargs):
        super().__init__(*args, underscores_to_dashes, explicit_bool, **kwargs)


if __name__ == '__main__':
    cfg = ResultConfig().parse_args()

    df = pd.DataFrame()
    for d in os.listdir(cfg.top_folder):
        curr_files = os.path.join(cfg.top_folder, d, "files")
        with open(os.path.join(curr_files, "wandb-metadata.json")) as jsonfile:
            metadata = json.load(jsonfile)
        name = metadata["args"][-8].replace("_", " ")
        try:
            with open(os.path.join(curr_files, "random_samples", "results.json")) as jsonfile:
                results = json.load(jsonfile)
                reshape_results = {"TPKL-Div 5 mean":results["tpkldiv"]["mean"]["5"],
                                   "TPKL-Div 5 var":results["tpkldiv"]["var"]["5"],
                                   "TPKL-Div 10 mean":results["tpkldiv"]["mean"]["10"],
                                   "TPKL-Div 10 var":results["tpkldiv"]["var"]["10"],
                                   "Levenshtein mean":results["levenshtein"]["mean"],
                                   "Levenshtein var":results["levenshtein"]["var"]}
        except Exception:
            print(f"No results for {d}")
            reshape_results = {}
        try:
            result_entropy = torch_load(os.path.join(curr_files, "random_samples", "mean_entropy.pt"))
            reshape_results["entropy mean"] = sum(result_entropy)/len(result_entropy)
        except Exception:
            print(f"No entropy for {d}")

        df = pd.concat([df, pd.DataFrame(reshape_results, index=[name])])

    df = df.sort_index()
    print(df.to_latex(float_format=lambda x: '%10.2f' % x))
    with open(os.path.join(cfg.top_folder, "tabular_results.tex"), "w") as txt_file:
        txt_file.write(df.to_latex(float_format=lambda x: '%10.2f' % x))

