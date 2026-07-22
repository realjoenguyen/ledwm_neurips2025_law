from typing import Union
from messenger.envs.base import Position
from messenger.envs.config import Entity
import numpy as np


AVATAR_KEYS = ["with_message.1", "no_message.1"]


def num_entities(obs):
    # except AVATAR_KEYS
    entities = [key for key in obs.keys() if key not in AVATAR_KEYS]
    return len(entities)


DEAD_ID = 0
DEAD_POS = 10


class DiscreteEntities:
    def __init__(
        self,
        key2order=None,
        is_avatar=False,
    ):
        self.is_avatar = is_avatar
        if not is_avatar:
            assert key2order is not None, key2order
            num_entities = len(key2order)
            self.key2order = key2order
        else:
            num_entities = 1
            self.key2order = None
            # self.key2order = {k: 0 for k in AVATAR_KEYS}

        self.ids = np.ones(num_entities, dtype=int) * DEAD_ID
        self.pos = np.ones((num_entities, 2), dtype=int) * DEAD_POS  # x, y
        self.manual_ids = np.ones(num_entities, dtype=int) * (-1)

    def add(
        self,
        entity: "Union[Entity, int]",
        position: "Position",
        manual_id: int = None,  # type: ignore
        key=None,  # key in self.entity2order enemy.1, goal.1, message.1
    ):
        if hasattr(entity, "id"):
            entity_id = entity.id
        else:
            entity_id = entity

        if manual_id is not None:
            assert isinstance(manual_id, int), manual_id
        if self.is_avatar:
            assert key is None, key
            assert manual_id is None, manual_id

        entity_order = 0
        if key is not None:
            # assert key in self.key2order, (key, self.key2order)
            assert self.key2order is not None, (key, self.key2order)
            entity_order = self.key2order[key]

        if isinstance(entity, Entity):
            assert entity_id > 0, entity_id
        else:
            assert entity_id >= 0, entity_id

        self.ids[entity_order] = entity_id
        self.pos[entity_order] = [position.x, position.y]

        if manual_id is not None:
            self.manual_ids[entity_order] = manual_id
