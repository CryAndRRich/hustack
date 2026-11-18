from __future__ import annotations

import copy
from pathlib import Path

import gin
import numpy as np

from smart_buildings import evaluate
from smart_buildings.smart_control.environment import hybrid_action_environment


def load_env(config_gin: Path, gin_overrides: dict | None = None):
    if gin_overrides:
        gin.clear_config()
        with gin.unlock_config():
            gin.parse_config_file(str(config_gin))
            for key, value in gin_overrides.items():
                gin.bind_parameter(key, value)
        env = hybrid_action_environment.HybridActionEnvironment()
    else:
        env = evaluate.load_hybrid_environment(Path(config_gin))
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    meta = evaluate.create_building_meta_data(env, Path(config_gin).parent)
    return env, meta


def _find_unpicklable(root, type_names=("SuperLU",)) -> list:
    found, seen, stack = [], set(), [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (np.ndarray, str, bytes, int, float, bool, type(None))):
            continue
        seen.add(id(obj))
        if type(obj).__name__ in type_names:
            found.append(obj)
            continue
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        else:
            d = getattr(obj, "__dict__", None)
            if d is not None:
                stack.extend(d.values())
            for slot in getattr(type(obj), "__slots__", ()):
                if hasattr(obj, slot):
                    stack.append(getattr(obj, slot))
    return found


def snapshot(*objs):
    memo = {id(lu): lu for lu in _find_unpicklable(objs)}
    return copy.deepcopy(objs, memo)
