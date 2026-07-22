from collections import defaultdict
import itertools
import random
import copy
import numpy as np
from ledwm.embodied.core.config import Config
from ledwm.embodied.run.smoothing import ReplayEps


class EntityAug(ReplayEps):
    def __init__(self, replay, sigma=0, zero_first=True, config: "Config" = None):
        super().__init__(replay, sigma, zero_first, config)

    def _add_eps_to_replay(self, worker, training=True):
        last_step = self.current_eps_trans[worker][-1]
        assert last_step["reward"] != 0

        # find the critical entity
        hit_pos_id = None
        ne = last_step["entity_ids"].shape[0]
        assert ne == last_step["entity_pos"].shape[0]

        # add the current eps regardless of entity aug
        # TODO optimize this
        eps_stream_temp = copy.deepcopy(self.current_eps_trans[worker])
        reward_stream_temp = copy.deepcopy(self.reward_buffer[worker])
        super()._add_eps_to_replay(worker, training=training)

        for pos_id, entity_pos in enumerate(last_step["entity_pos"]):
            if np.array_equal(entity_pos[:2], last_step["avatar_pos"][0][:2]):
                hit_pos_id = pos_id
                hit_pos = entity_pos
                break

        # other ids except hit_pos_id: range(3) - hit_pos_id
        # entity_mask = np.ones(ne, dtype=bool)
        if hit_pos_id is not None:
            other_ids = [i for i in range(ne) if i != hit_pos_id]
        else:
            return
        #     other_ids = list(range(ne))

        # eps_stream = self.current_eps_trans[worker].copy()
        # if hit then can take until len(other_ids) else need at least 1 other id alive
        for subset_size in range(1, len(other_ids) + (hit_pos_id is not None)):
            # subset_size = random.randint(1, len(other_ids))
            # Size can be from 0 to the length of other_ids
            # random_subset = random.sample(other_ids, subset_size)  # (d,)
            if hit_pos_id is None:
                assert subset_size < len(other_ids)

            # take all subset instead
            subsets = list(itertools.combinations(other_ids, subset_size))
            for subset in subsets:
                random_subset = list(subset)
                # entity_mask[random_subset] = 0

                # step['entity_ids']: (Ne,)
                # step['entity_pos']: (Ne, 3)
                eps_stream_aug = copy.deepcopy(eps_stream_temp)
                for i, step in enumerate(eps_stream_aug):
                    eps_stream_aug[i]["entity_ids"][random_subset] = 0
                    eps_stream_aug[i]["entity_pos"][random_subset, :] = 0

                super()._add_eps_to_replay(
                    worker,
                    eps_stream=eps_stream_aug,
                    reward_stream=reward_stream_temp,
                    training=training,
                    aug=True,
                )
