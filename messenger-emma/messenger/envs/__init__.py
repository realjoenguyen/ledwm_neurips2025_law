__all__ = ["TwoEnvWrapper", "StageOne", "StageTwo", "StageThree"]


def __getattr__(name):
    if name == "TwoEnvWrapper":
        from messenger.envs.TwoEnvWrapper import TwoEnvWrapper

        globals()[name] = TwoEnvWrapper
        return TwoEnvWrapper
    if name == "StageOne":
        from messenger.envs.stage_one import StageOne

        globals()[name] = StageOne
        return StageOne
    if name == "StageTwo":
        from messenger.envs.stage_two import StageTwo

        globals()[name] = StageTwo
        return StageTwo
    if name == "StageThree":
        from messenger.envs.stage_three import StageThree

        globals()[name] = StageThree
        return StageThree
    raise AttributeError(name)
