# %%
import argparse

import wandb

parser = argparse.ArgumentParser(description="Migrate a W&B replay configuration.")
parser.add_argument("run_path", help="W&B run path in entity/project/run-id form")
args = parser.parse_args()

api = wandb.Api()
run = api.run(args.run_path)

print(run.config["upsample_pos"])
print(run.config["replay"]["upsample_pos"])

run.config["replay"]["upsample_pos"] = run.config["upsample_pos"]
run.update()
