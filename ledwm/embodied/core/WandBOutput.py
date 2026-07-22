import os
from typing import Dict

import numpy as np
from termcolor import cprint
import wandb
import collections
import re
import pathlib
from ledwm.WM import MASK_VALUE_DIST
from ledwm.common import (
    atten2str,
    # avatar2str,
    # avatar_entity2str,
    image2str,
    image_error2str,
)
from ledwm.embodied.core import path
from collections import defaultdict, deque
from functools import partial as bind


def get_logdir_from_config(config):
    main_task, sub_task = config.task.split("_")
    if main_task == "lwm":
        exp_name = f"{config.data}"  # token or sent
    else:
        exp_name = f"{sub_task}_{config.data}"

    if config.overfit or config.run.overfit_eps:
        exp_name = f"{exp_name}_overfit"

    # lwm / sent
    res = pathlib.Path(config.logdir_base) / main_task / exp_name
    cprint(f"logdir = {res}", "green")
    return res


def create_wandb_init(config, logdir):
    wandb_id_file = f"{str(logdir)}/wandb_id.txt"
    wandb_pa = path.Path(wandb_id_file)
    if wandb_pa.exists():
        print("!! Resuming wandb run !!")
        wandb_id = wandb_pa.read().strip()
    else:
        wandb_id = wandb.util.generate_id()  # type: ignore
        wandb_pa.write(str(wandb_id))

    group_name = get_logdir_from_config(config).name
    cprint(f"logger.wandb | group={group_name} | name={logdir.name}")
    run = wandb.init(
        id=wandb_id,
        resume="allow",
        project="messenger",
        name=logdir.name,
        group=group_name,
        # sync_tensorboard=config.sync_tfboard,
        sync_tensorboard=False,
        config=dict(config),
        settings=wandb.Settings(code_dir=".."),
        entity=os.environ.get("WANDB_ENTITY") or None,
    )
    # log code, exclude files with "jax" in path
    # path of the parent of the current file
    parent_path = pathlib.Path(__file__).parent.parent.parent
    wandb.run.log_code(parent_path, exclude_fn=lambda x: "jax/" in x or ".history" in x)  # type: ignore
    return run


def make_table_data_text(metrics: Dict[str, np.ndarray], id2sent=None):
    is_first = metrics["table_is_first"]
    action = metrics["table_action"]
    openl_reward = metrics["table_openl_reward"]
    openl_cont = metrics["table_openl_cont"]
    cont_data = metrics["table_cont_data"]
    cont_pred = metrics["table_cont_pred"]
    reward_data = metrics["table_reward_data"]
    reward_pred = metrics["table_reward_pred"]
    step = metrics["table_step"]
    entity_pos = metrics["table_entity_pos"]
    avatar_pos = metrics["table_avatar_pos"]

    res = {
        "table_atten": atten2str(
            metrics["table_atten"],
            metrics["table_entity_ids"],
            metrics["table_sent_ids"],
            id2sent,
            is_first,
            reward_data,
            entity_pos,
            avatar_pos,
        ),
        # "table_pos": avatar_entity2str(
        #     metrics["table_entity_id"],
        #     metrics["table_entity_pos"],
        #     avatar_pos=metrics["table_avatar_pos"],
        #     rewards=reward_data,
        #     is_first=is_first,
        #     cont=cont_data,
        #     action=action,
        # ),
    }
    if "table_atten1" in metrics:
        res["table_atten1"] = atten2str(
            metrics["table_atten1"],
            metrics["table_entity_ids"],
            metrics["table_sent_ids"],
            id2sent,
            is_first,
            reward_data,
            entity_pos,
            avatar_pos,
        )
    if "table_atten2" in metrics:
        res["table_atten2"] = atten2str(
            metrics["table_atten2"],
            metrics["table_entity_ids"],
            metrics["table_sent_ids"],
            id2sent,
            is_first,
            reward_data,
            entity_pos,
            avatar_pos,
        )

    # TODO fix this
    if "table_image_gt" in metrics:
        image_gt = metrics["table_image_gt"]
        gt_multihot = metrics["table_image_gt_multihot"]
        res.update(
            {
                "table_image_gt": image2str(
                    image_gt, is_first, action, reward_data, cont_data, step
                ),
                "table_image_gt_multihot": image2str(
                    gt_multihot, is_first, action, reward_data, step
                ),
            }
        )
        if "table_image_pred" in metrics:
            pred = metrics["table_image_pred"]
            openl = metrics["table_image_openl"]

            res.update(
                {
                    "table_image_diff": image_error2str(
                        gt_multihot, pred, action, is_first
                    ),
                    "table_image_pred": image2str(
                        pred, is_first, action, reward_pred, cont_pred, step
                    ),
                    "table_image_openl": image2str(
                        openl, None, action, openl_reward, openl_cont, step
                    ),
                }
            )
    return res


TABLE_LEN = 5


class WandBOutput:
    def __init__(
        self,
        run,
        config=None,
        pattern=r".*",
        table_keys=None,
        real_step_log=False,
        table_names=None,
    ):
        self._pattern = re.compile(pattern)
        self._wandb = wandb
        self.real_step_log = real_step_log
        self.step2real_step = {}
        self.last_real_step = None
        if table_keys is not None:
            self.table = collections.defaultdict(bind(deque, maxlen=TABLE_LEN))
            self.columns = list(table_keys)
            print(f"logger.wandb_table | columns={self.columns}")

    def __call__(self, summaries):
        bystep = collections.defaultdict(dict)
        table_step = None
        if hasattr(self, "table"):
            table_step = collections.defaultdict(dict)

        for step, name, value in summaries:
            if name == "real_step":
                self.step2real_step[step] = int(value)

            if "table_" in name:
                if not hasattr(self, "table"):
                    continue
                if not isinstance(value, str):
                    if len(value.shape) == 0:
                        value = float(value)
                # table_data[step][name.split("table_")[-1]] = value
                assert table_step is not None
                table_step[step][name] = value

            elif isinstance(value, str):
                bystep[step][name] = value

            elif "hist" in name:
                value = value[..., 0].reshape(-1)
                try:
                    bystep[step][name] = wandb.Histogram(value)
                except Exception as e:
                    raise ValueError(f"Error in logging {name}: {e}", "red")

            elif "fig" in name:
                bystep[step][name] = wandb.Image(value)
                cprint(f"logger.wandb_figure | name={name}", "green")

            elif len(value.shape) == 0:
                bystep[step][name] = float(value)

            elif len(value.shape) == 1:
                # if value has nan then continue
                if np.isnan(value).any():
                    cprint(f"logger.invalid_value | backend=wandb | name={name} | type=nan")
                    continue
                # check if it has inf
                if np.isinf(value).any():
                    cprint(f"logger.invalid_value | backend=wandb | name={name} | type=inf")
                    continue
                # try:
                if "game_id" in name:
                    # value has -1, filter out
                    # debug_value = value.copy()
                    value = value[value != -1]

                # remove all values = MASK_VALUE_DIST
                value = value[value != MASK_VALUE_DIST]
                bystep[step][name] = wandb.Histogram(value)

            elif len(value.shape) in (2, 3):
                value = value[..., None] if len(value.shape) == 2 else value
                # if len(value.shape) == 3:
                #     assert value.shape[3] in [1, 3, 4], f"{value.shape=}, {name=}"
                if value.dtype != np.uint8:
                    value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
                value = np.transpose(value, [2, 0, 1])
                try:
                    bystep[step][name] = wandb.Image(value)
                except Exception as e:
                    cprint(
                        f"logger.wandb_error | name={name} | error={e}", "red"
                    )
                    continue

            elif len(value.shape) == 4:
                assert value.shape[3] in [1, 3, 4], value.shape
                value = np.transpose(value, [0, 3, 1, 2])
                if value.dtype != np.uint8:
                    value = (255 * np.clip(value, 0, 1)).astype(np.uint8)
                bystep[step][name] = wandb.Video(value, fps=2)

        last_step = None
        for step, metrics in bystep.items():
            # metrics: {name: value}
            if self.real_step_log:
                if step in self.step2real_step:
                    self._wandb.log(metrics, step=self.step2real_step[step])
                    last_step = step
                    self.last_real_step = self.step2real_step[step]
                    # print("Wandb logged")
                else:
                    if self.last_real_step is not None:
                        self._wandb.log(metrics, step=self.last_real_step)
                    else:
                        self._wandb.log(metrics)
                    # print("Wandb logged")

            else:
                self._wandb.log(metrics)  # metrics: {'train*': *, 'val*': *}
                # print("Wandb logged")

        if hasattr(self, "table") and table_step is not None and len(table_step) > 0:
            # create table with many rows - each row is a step in table_data
            for step, table_data_step in list(table_step.items()):
                name2table = defaultdict(dict)
                for k, v in table_data_step.items():
                    table_name = k.split("table_")[0]
                    # remove the last / from table_name
                    table_col = k.split("table_")[-1]
                    name2table[table_name][table_col] = v

                # columns for this table_name
                for table_name, columns_data in name2table.items():
                    self.table[table_name].append(
                        [columns_data[key] for key in self.columns]
                    )

            for table_name, table_rows in list(self.table.items()):
                table_tag = f"{table_name}table"
                if self.real_step_log:
                    if last_step is not None and last_step in self.step2real_step:
                        self._wandb.log(
                            {
                                table_tag: wandb.Table(
                                    columns=self.columns, data=list(table_rows)
                                )
                            },
                            step=self.step2real_step[last_step],
                        )
                    # else:
                    # cprint(f"table: {last_step} not in self.step2real_step", "red")
                else:
                    self._wandb.log(
                        {
                            table_tag: wandb.Table(
                                columns=self.columns, data=list(table_rows)
                            )
                        }
                    )
