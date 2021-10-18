#%%
import os
import wandb

api = wandb.Api()

runs = api.runs("tnt/world-gan")

#%%
for run in runs:
    if "ablation_journal" in run.tags:
        repr_type = "bert" if "bert" in run.tags else "bert_naive"
        output_dir = f"./output/ablation/{repr_type}/{run.id}"
        os.makedirs(output_dir, exist_ok=True)
        for file in run.files():
            file.download(root=output_dir)
 
# %%
