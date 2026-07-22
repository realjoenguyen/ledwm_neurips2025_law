import pathlib
from termcolor import cprint
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ledwm.agent import Agent
from ledwm.embodied.core.checkpoint import Checkpoint
from ledwm.embodied.core.metrics import Metrics
from ledwm.embodied.core.path import Path
from ledwm.embodied.core.timer import Timer
from ledwm.embodied.core.when import Clock
from ledwm.embodied.core.logger import Logger
from ledwm.embodied.replay.generic import GenericReplay
import numpy as np
import jax.numpy as jnp

# from ledwm.embodied.run.parallel import inspect_batch_unittest
from ledwm.jaxutils import create_horizon_mask_from, extract_horizon_data_from
import time

TOKENIZER = None


class TimingWrapper:
    def __init__(self):
        self.times = {}
        self.counts = {}

    def time(self, name):
        """Return a context manager for timing operations"""
        return self._TimeContext(self, name)

    def _record_time(self, name, elapsed):
        """Record timing for an operation"""
        if name not in self.times:
            self.times[name] = []
            self.counts[name] = 0

        self.times[name].append(elapsed)
        self.counts[name] += 1

    def get_avg_times(self, window=10):
        """Get average times for the last 'window' operations"""
        avg_times = {}
        for name, times in self.times.items():
            if len(times) > 0:
                recent_times = times[-window:]
                avg_times[name] = sum(recent_times) / len(recent_times)
        return avg_times

    def format_times(self, window=10):
        """Format timing information for display"""
        avg_times = self.get_avg_times(window)
        if not avg_times:
            return ""

        time_strs = []
        for name, avg_time in avg_times.items():
            time_strs.append(f"{name}: {avg_time:.3f}s")

        return " | ".join(time_strs)

    class _TimeContext:
        def __init__(self, wrapper, name):
            self.wrapper = wrapper
            self.name = name
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
                self.wrapper._record_time(self.name, elapsed)


# def decode_tokens(tokens):
#     global TOKENIZER
#     if not TOKENIZER:
#         from transformers import T5Tokenizer

#         TOKENIZER = T5Tokenizer.from_pretrained("t5-small")
#     if len(tokens.shape) > 2:
#         tokens = tokens.reshape((-1, tokens.shape[-1]))
#     return TOKENIZER.batch_decode(tokens)


# def offline(agent, offline_ds, eval_replay, logger, args):
#     logdir = Path(args.logdir)
#     logdir.mkdirs()
#     print("Logdir", logdir)
#     should_expl = embodied.when.Until(args.expl_until)
#     should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
#     should_log = Clock(args.log_every)
#     should_save = Clock(args.save_every)
#     step = logger.step
#     metrics = Metrics()

#     timer = Timer()
#     timer.wrap("agent", agent, ["policy", "train", "report", "save"])
#     timer.wrap("logger", logger, ["write"])

#     eval_dataset = agent.dataset(eval_replay.dataset)

#     dataset = iter(offline_ds)
#     state = [None]  # To be writable from train step function below.

#     # Pretraining mode: prepare to save checkpoint
#     checkpoint = Checkpoint(logdir / "checkpoint.pkl")
#     timer.wrap("checkpoint", checkpoint, ["save", "load"])
#     checkpoint.step = step
#     checkpoint.agent = agent
#     checkpoint.load_or_save()
#     should_save(step)
#     print(f"Ckpt has step {checkpoint.step.value}")

#     for pretrain_iter in range(args.pretrain):
#         with timer.scope("dataset"):
#             batch = next(dataset)
#             batch = agent.postprocess(batch)
#             if pretrain_iter == 0:
#                 print("Batch:")
#                 for k, v in batch.items():
#                     print(f"{k} {v.shape}")

#         _, state[0], mets = agent.train(batch, state[0])
#         # Count pretrain steps
#         step.increment()
#         metrics.add(mets, prefix="pretrain")
#         if pretrain_iter % 500 == 0:
#             agg = metrics.result()
#             report = agent.report(batch)

#             report = {k: v for k, v in report.items() if "pretrain/" + k not in agg}
#             eval_report = agent.report(next(eval_dataset))
#             logger.add(agg)
#             logger.add(report, prefix="report")
#             logger.add(eval_report, prefix="eval/report")
#             logger.add(timer.stats(), prefix="timer")
#             logger.add({"epoch": dataset.epoch})
#             logger.write(fps=True)

#         if should_save(pretrain_iter):
#             checkpoint.save()

#     print("Pretraining done.")
#     return


class Dataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, replay: GenericReplay):
        self.replay = replay

    def __len__(self):
        return len(self.replay)

    def __getitem__(self, idx):
        return self.replay[idx]


def numpy_collate(batch):
    # batch is a list of dicts: [{data1: data1, label1: label1}, {data2: data2, label2: label2}, ...]
    return {k: np.stack([d[k] for d in batch]) for k in batch[0]}


def train_offline_wm(
    agent: Agent,
    train_replay: GenericReplay,
    eval_replay: GenericReplay,
    logger: Logger,
    config,
):
    args = config.run
    logdir = Path(config.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_save = Clock(args.save_every)
    step = logger.step

    timer = Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("replay", train_replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    metrics = Metrics()

    checkpoint = Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.step = step
    checkpoint.agent = agent

    if args.from_checkpoint != "":
        checkpoint.load(args.from_checkpoint)
    timer = Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("logger", logger, ["write"])
    should_save(step)

    print(f"Ckpt has step {checkpoint.step.value}")

    if args.torch_dataloader:
        train_dataset = Dataset(train_replay)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,  # TODO change this later
            collate_fn=numpy_collate,
            num_workers=8,
            drop_last=True,
        )

        test_dataset = Dataset(eval_replay)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=numpy_collate,
            drop_last=True,
            num_workers=8,
        )
    else:
        train_dataset = agent.dataset(train_replay.dataset)
        test_dataset = agent.dataset(eval_replay.dataset)

    def train_step(agent: Agent, batch, t, state=None, opt_step=None, overfit=False):
        _, state, mets = agent.train(batch, state, step=opt_step)
        if t % 100 == 0:
            if "rollout_reward_loss_full" in mets:  # bs, bl, horizon
                # last name of logdir, using pathlib
                exp_name = pathlib.Path(config.logdir).name
                with open(
                    f"rollout_reward_kl_{config.run.debug=}.txt_{exp_name}", "a"
                ) as f:
                    f.write(f"{t=}\n")
                    f.write(f"{batch['time_step'][0]}\n")
                    arr = mets["rollout_reward_loss_full"][0]  # bl, horizon
                    horizon_mask_from_next = create_horizon_mask_from(
                        batch["is_first"], batch["is_last"], config.imag_horizon
                    )
                    reward_from_next = extract_horizon_data_from(
                        batch["reward"], config.imag_horizon, horizon_mask_from_next
                    )[0]
                    horizon_mask_from_next = horizon_mask_from_next.astype(jnp.float32)
                    # sum of reward loss at each h
                    for h in range(config.imag_horizon):
                        total_reward_loss = mets["rollout_reward_loss_full"][
                            :, :, h
                        ].sum()
                        total_mask = horizon_mask_from_next[:, :, h].sum()
                        f.write(
                            f"avg_reward_loss_at_{h=}: {total_reward_loss / total_mask}\n"
                        )
                        if "rollout_dyn_loss_full" in mets:
                            f.write(
                                f"avg_rollout_dyn_loss_at_{h=}: {mets['rollout_dyn_loss_full'][:, :, h].sum() / total_mask}\n"
                            )

                    with np.printoptions(formatter={"float_kind": "{:.1f}".format}):
                        for i in range(arr.shape[0]):
                            f.write(f"{i=}\n")
                            f.write(f"reward_loss=\n{arr[i]}\n")
                            f.write(f"reward_from_next=\n{reward_from_next[i]}\n")
                            f.write(
                                f"horizon_mask_from_next=\n{horizon_mask_from_next[0, i]}\n"
                            )
                            if "rollout_dyn_loss_full" in mets:
                                f.write(
                                    f"dyn_loss=\n{mets['rollout_dyn_loss_full'][0][i]}\n"
                                )
                            f.write(f"time_step=\n{batch['time_step'][0][i]}\n")

                            f.write("-" * 50 + "\n")
                        print("Write to file")

        mets = {k: v for k, v in mets.items() if "loss_full" not in k}
        metrics.add(mets, prefix="train")
        interval = 50 if not overfit else 1

        if t % interval == 0:
            agg = metrics.result()
            report = agent.report(batch, step=opt_step)
            report = {k: v for k, v in report.items() if "train/" + k not in agg}

            # remove rollout_reward_1_pos_loss_full and rollout_reward_1_neg_loss_full from report
            report = {k: v for k, v in report.items() if "loss_full" not in k}
            logger.add(agg, prefix=None)
            logger.add(report, prefix="report")
            logger.write(fps=True)

        return mets

    def eval_after_epoch(agent: Agent, test_loader, logger: Logger, opt_step=None):
        metrics = Metrics()
        timing_wrapper = TimingWrapper()
        pbar = tqdm(test_loader, desc="Offline eval batches")

        for i, raw_batch in enumerate(pbar):
            with timing_wrapper.time("postprocess"):
                batch = agent.postprocess(raw_batch)

            with timing_wrapper.time("report"):
                report = agent.report(batch, step=opt_step)

            metrics.add(report, prefix="report")

            # Update progress bar with timing info
            if i % 10 == 0:  # Update every 10 iterations
                timing_info = timing_wrapper.format_times()
                if timing_info:
                    pbar.set_description(f"Offline eval batches | {timing_info}")

        logger.add(metrics.result(), prefix="test")
        logger.write(fps=True)

    if args.torch_dataloader:
        batch = next(iter(train_loader))
        batch = agent.postprocess(batch)
        # inspect_batch_unittest(batch, config)
        state = None

        # overfit one batch
        if config.overfit_batch:
            batch = next(iter(train_loader))
            batch = agent.postprocess(batch)
            state = None
            exp_name = pathlib.Path(config.logdir).name
            with open(
                f"rollout_reward_kl_{config.run.debug=}.txt_{exp_name}",
                "w",
            ) as f:
                # write horizon_mask_from_next to file
                horizon_mask_from_next = create_horizon_mask_from(
                    batch["is_first"], batch["is_last"], config.imag_horizon
                )[0]
                f.write("horizon_mask_from_next.shape: ")
                f.write(f"{horizon_mask_from_next.shape}\n")
                f.write("horizon_mask_from_next:\n")
                f.write(f"{horizon_mask_from_next}\n")
                # reward values
                f.write("reward_values:\n")
                f.write(f"{batch['reward'][0]}\n")

            timing_wrapper = TimingWrapper()
            pbar = tqdm(range(config.overfit_batch_steps), desc="Overfit batch")

            for t in pbar:
                with timing_wrapper.time("train"):
                    mets = train_step(
                        agent,
                        batch,
                        t,
                        opt_step=t if args.opt_step else None,
                        overfit=True,
                    )

                # Update progress bar with timing info
                # Update every 10 iterations to avoid too frequent updates
                if t % 10 == 0:
                    timing_info = timing_wrapper.format_times()
                    if timing_info:
                        pbar.set_description(f"Overfit batch | {timing_info}")

                if t % 100 == 0:
                    checkpoint.save()

            cprint("Done testing overfit batch.", "green")
            return

        eval_after_epoch(agent, test_loader, logger, opt_step=0)
        epochs = args.train_iter // len(train_loader)
        timing_wrapper = TimingWrapper()

        for epoch in tqdm(range(epochs), desc="Offline epochs"):
            pbar = tqdm(train_loader, desc="Offline batches")
            for t, raw_batch in enumerate(pbar):
                with timing_wrapper.time("postprocess"):
                    batch = agent.postprocess(raw_batch)

                opt_step = epoch * len(train_loader) + t
                with timing_wrapper.time("train"):
                    train_step(
                        agent,
                        batch,
                        t,
                        opt_step=opt_step if args.opt_step else None,
                    )

                # Update progress bar with timing info
                # Update every 10 iterations to avoid too frequent updates
                if t % 10 == 0:
                    timing_info = timing_wrapper.format_times()
                    if timing_info:
                        pbar.set_description(f"Offline batches | {timing_info}")

            eval_after_epoch(agent, test_loader, logger, opt_step=opt_step)
            checkpoint.save()
    else:
        batch = next(train_dataset)
        batch = agent.postprocess(batch)
        # inspect_batch_unittest(batch, config)
        state = None

        timing_wrapper = TimingWrapper()
        pbar = tqdm(range(args.train_iter), desc="Offline training")

        for t in pbar:
            with timing_wrapper.time("get_batch"):
                batch = next(train_dataset)

            with timing_wrapper.time("postprocess"):
                batch = agent.postprocess(batch)

            if t == 0:
                print("Batch:")
                for k, v in batch.items():
                    print(f"{k} {v.shape}")

            with timing_wrapper.time("train"):
                train_step(agent, batch, t, state)

            # Update progress bar with timing info
            if t % 10 == 0:  # Update every 10 iterations to avoid too frequent updates
                timing_info = timing_wrapper.format_times()
                if timing_info:
                    pbar.set_description(f"Offline training | {timing_info}")

            if t % 1000 == 0:
                checkpoint.save()

    cprint("Offline training done.", "green")


def eval_offline_wm(agent: Agent, replay: GenericReplay, logger: Logger, config):
    args = config.run
    logdir = Path(config.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_save = Clock(args.save_every)
    step = logger.step

    # metrics = Metrics()
    checkpoint = Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.step = step
    checkpoint.agent = agent

    if args.from_checkpoint != "":
        checkpoint.load(args.from_checkpoint)
    timer = Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("logger", logger, ["write"])
    should_save(step)

    print(f"Ckpt has step {checkpoint.step.value}")

    if args.torch_dataloader:
        dataset = Dataset(replay)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=numpy_collate,
            num_workers=8,
        )
    else:
        dataset = agent.dataset(replay.dataset)

    def eval_step(agent: Agent, batch):
        report = agent.report(batch)
        logger.add(report, prefix="report")
        logger.write(fps=True)

    if args.torch_dataloader:
        batch = next(iter(dataloader))
        batch = agent.postprocess(batch)
        inspect_batch_unittest(batch, config)

        timing_wrapper = TimingWrapper()
        pbar = tqdm(dataloader, desc="Offline eval batches")

        for i, raw_batch in enumerate(pbar):
            with timing_wrapper.time("postprocess"):
                batch = agent.postprocess(raw_batch)

            with timing_wrapper.time("eval"):
                eval_step(agent, batch)

            # Update progress bar with timing info
            if i % 10 == 0:  # Update every 10 iterations
                timing_info = timing_wrapper.format_times()
                if timing_info:
                    pbar.set_description(f"Offline eval batches | {timing_info}")
    else:
        raise NotImplementedError

    cprint("Offline eval done.", "green")
