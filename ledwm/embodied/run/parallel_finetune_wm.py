# from multiprocessing import Lock, Value
# import pathlib
# import re
# import sys
# import time
# from collections import OrderedDict, defaultdict
# import wandb

# # tying and TYPE_CHECKING are used to avoid circular imports.
# from typing import TYPE_CHECKING, Dict
# from termcolor import cprint
# from common import get_multihot_image_from_pos, log_image
# from ledwm.embodied.core.base import Agent
# from ledwm.embodied.replay.generic import GenericReplay, convert2uuid

# from jax import numpy as jnp

# from embodied.core.config import Config
# from embodied.run.parallel import get_env_address, id2sent_from_env_cache

# if TYPE_CHECKING:
#     from ledwm.embodied.core.logger import Logger
#     from ledwm.embodied.replay.replays import Uniform
#     from ledwm.jaxagent import JAXAgent
#     from ledwm.embodied.core.base import Agent
#     from ledwm.embodied.replay.limiters import SamplesPerInsert

# from ledwm.embodied.replay.Prioritized import PrioritizedReplay, PrioritizedSampler
# from ledwm.embodied.run.smoothing import ReplayEps
# from ledwm import embodied
# import numpy as np


# def parallel_finetune_wm(
#     agent: "JAXAgent",
#     train_replay: "Uniform | PrioritizedReplay | ReplayEps",
#     logger: "Logger",
#     make_train_env,
#     args,
#     env_cache,
#     make_eval_env=None,
#     make_test_env=None,
#     eval_replay: "Uniform" = None,
#     test_replay=None,
#     config: "Config" = None,
# ):
#     logdir = embodied.Path(args.logdir)
#     checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
#     env_step = logger.step
#     real_env_step = embodied.Counter()
#     opt_step = embodied.Counter()
#     checkpoint.step = env_step
#     checkpoint.real_step = real_env_step
#     checkpoint.agent = agent
#     checkpoint.opt_step = opt_step
#     # checkpoint.best_dev_win_rate = 0

#     if args.from_checkpoint != "":
#         checkpoint.load(args.from_checkpoint, skip_key=args.skip_key, strict=True)
#     if args.load_checkpoint:
#         checkpoint.load_or_save()

#     # cprint(f"Resume: {env_step=}, {real_env_step=}", "red")
#     # if wandb is running
#     if wandb.run is not None:
#         cprint(f"Wandb step: {wandb.run.step}", "red")
#         env_step.load(wandb.run.step)
#         checkpoint.step = env_step

#     timer = embodied.Timer()
#     timer.wrap("agent", agent, ["policy", "train", "report", "save"])
#     timer.wrap("replay", train_replay, ["add", "save"])
#     timer.wrap("logger", logger, ["write"])
#     # usage = embodied.Usage(args.trace_malloc)

#     workers = []
#     print("start all threads")
#     global start_parallel
#     start_parallel = time.time()
#     train_env_ids = list(range(config.envs.amount))
#     worker_addr2type = {get_env_address(i): "train" for i in train_env_ids}
#     assert env_cache is not None
#     id2sent = id2sent_from_env_cache(env_cache)

#     if make_eval_env is not None:
#         assert args.script in ["parallel_train_eval", "parallel_train_eval_test"]
#         eval_env_ids = list(
#             range(len(train_env_ids), len(train_env_ids) + config.num_eval_envs)
#         )
#         worker_addr2type.update({get_env_address(i): "eval" for i in eval_env_ids})

#         if config.num_eval_envs == 1:
#             workers.append(
#                 embodied.distr.Thread(
#                     parallel_env, 1, make_eval_env, args, worker_addr2type, timer
#                 )
#             )
#         else:
#             for i in eval_env_ids:
#                 worker = embodied.distr.Process(
#                     parallel_env,
#                     i,
#                     make_eval_env,
#                     args,
#                     worker_addr2type,
#                     name=f"parallel_env_eval_{i}",
#                 )
#                 worker.start()
#                 workers.append(worker)
#             # usage.processes("eval_envs", workers)

#         if "test" in args.script:
#             cprint("INIT TEST ENV", "green")
#             assert make_test_env is not None, "make_test_env is None"
#             test_env_ids = list(
#                 range(
#                     len(worker_addr2type), len(worker_addr2type) + config.num_eval_envs
#                 )
#             )
#             worker_addr2type.update({get_env_address(i): "test" for i in test_env_ids})

#             if config.num_eval_envs == 1:
#                 workers.append(
#                     embodied.distr.Thread(
#                         parallel_env, 1, make_test_env, args, worker_addr2type, timer
#                     )
#                 )
#             else:
#                 for i in test_env_ids:
#                     worker = embodied.distr.Process(
#                         parallel_env,
#                         i,
#                         make_test_env,
#                         args,
#                         worker_addr2type,
#                         name=f"parallel_env_test_{i}",
#                     )
#                     worker.start()
#                     workers.append(worker)
#                 # usage.processes("test_envs", workers)

#     # TRAIN ENVS
#     if len(train_env_ids) == 1:
#         workers.append(
#             embodied.distr.Thread(
#                 parallel_env, 0, make_train_env, args, worker_addr2type, timer
#             )
#         )
#     else:
#         for i in train_env_ids:
#             worker = embodied.distr.Process(
#                 parallel_env,
#                 i,
#                 make_train_env,
#                 args,
#                 worker_addr2type,
#                 name=f"parallel_env_{i}",
#             )
#             worker.start()
#             workers.append(worker)

#     # usage.processes("envs", workers)  # envs_count

#     workers.append(
#         embodied.distr.Thread(
#             parallel_actor,
#             env_step,
#             real_env_step,
#             opt_step,
#             agent,
#             train_replay,
#             logger,
#             # timer,
#             args,
#             worker_addr2type,
#             checkpoint,
#             eval_replay,
#             test_replay,
#         )
#     )

#     workers.append(
#         embodied.distr.Thread(
#             parallel_learner,
#             env_step,
#             real_env_step,
#             opt_step,
#             agent,
#             train_replay,
#             logger,
#             # timer,
#             # usage,
#             checkpoint,
#             args,
#             id2sent,
#             config,
#         )
#     )

#     if make_eval_env is not None:
#         workers.append(
#             embodied.distr.Thread(
#                 parallel_eval,
#                 env_step,
#                 real_env_step,
#                 opt_step,
#                 agent,
#                 eval_replay,
#                 logger,
#                 timer,
#                 args,
#                 "eval",
#                 id2sent,
#             )
#         )

#     if make_test_env is not None:
#         workers.append(
#             embodied.distr.Thread(
#                 parallel_eval,
#                 env_step,
#                 real_env_step,
#                 opt_step,
#                 agent,
#                 test_replay,
#                 logger,
#                 timer,
#                 args,
#                 "test",
#                 id2sent,
#             )
#         )

#     embodied.distr.run(workers)
