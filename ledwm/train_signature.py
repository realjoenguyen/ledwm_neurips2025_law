def canonical_reward_weight_keys(config):
    """Return fixed reward-weight keys when a task has a known train signature."""
    if config.replay.imbalance != "balanced_weight":
        return None
    task = str(config.task)
    if task.startswith("messenger_"):
        task = task[len("messenger_") :]
    if task == "s1":
        return (-1.0, 1.0)
    if task in ("s2", "lwm_easy", "lwm_medium", "lwm_hard"):
        return (-1.0, -0.5, 1.5)
    if task == "s3":
        return (-2.0, -1.5, -1.0, -0.5, 1.5)
    return None
