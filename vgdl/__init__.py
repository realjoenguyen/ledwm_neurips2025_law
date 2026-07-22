import os
import warnings

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
warnings.filterwarnings(
    "ignore",
    "pkg_resources is deprecated as an API.*",
    UserWarning,
    "pygame.pkgdata",
)

from .registration import registry
from .parser import VGDLParser

# __all__ = ['VGDLParser', 'ontology', 'registry']

from .core import SpriteRegistry
from .core import BasicGame, BasicGameLevel, GameState
from .core import Action, ACTION
from .core import Avatar, VGDLSprite, Immutable, Physics, Termination, Effect, FunctionalEffect
