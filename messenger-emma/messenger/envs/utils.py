"""
Utilites for the environments
"""

import json
from functools import lru_cache
from pathlib import Path
from messenger.envs.config import NPCS, Game


def get_entity(name: str):
    """
    Get the Entity object for the entity with
    name name.
    """
    for entity in NPCS:
        if entity.name == name:
            return entity
    raise Exception("entity not found.")


def get_game(game_tuple):
    """
    Take a tuple of strings (enemy, message, goal) and get the
    corresponding Game object.
    """
    enemy_name, message_name, goal_name = game_tuple
    enemy = get_entity(enemy_name)
    message = get_entity(message_name)
    goal = get_entity(goal_name)
    return Game(enemy=enemy, message=message, goal=goal)


@lru_cache(maxsize=None)
def _json_from_path(path):
    with Path(path).open(mode="r") as json_file:
        return json.load(json_file)


def json_from_path(json_path):
    return _json_from_path(str(Path(json_path)))


@lru_cache(maxsize=None)
def _text_from_path(path):
    return Path(path).read_text()


def text_from_path(path):
    return _text_from_path(str(Path(path)))


@lru_cache(maxsize=None)
def _games_from_json(json_path, split):
    games = _json_from_path(json_path)
    return tuple(get_game(game) for game in games[split])


def games_from_json(json_path, split: str):
    """
    Convert game strings in games.json to Game namedtuples
    """
    json_path = str(Path(json_path))
    # Preserve the historical mutable list container while sharing the immutable
    # Game/Entity namedtuples and parsed JSON within each spawned worker.
    return list(_games_from_json(json_path, split))
