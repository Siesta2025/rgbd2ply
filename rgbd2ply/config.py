"""Configuration loader for rgbd2ply.

Loads config.yaml from the rgbd2ply package directory, then applies overrides
from environment variables (RGBD2PLY_*). All access is via attribute-style dot paths.

Usage:
    from rgbd2ply.config import cfg
    print(cfg.paths.project_root)
    print(cfg.defaults.thr)
"""

import os
from pathlib import Path

import yaml as _yaml


class _ConfigStore:
    """Attribute-accessible nested dict loaded from YAML + env overrides."""

    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        v = self._data.get(name)
        if v is None:
            raise AttributeError(name)
        if isinstance(v, dict):
            return _ConfigStore(v)
        return v

    def __repr__(self):
        return repr(self._data)

    def get(self, name, default=None):
        try:
            return getattr(self, name)
        except AttributeError:
            return default


def _env_override(data, prefix="RGBD2PLY_"):
    """Walk flattened env vars like RGBD2PLY_PATHS_PROJECT_ROOT back into data."""
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("_")
        if len(parts) < 2:
            continue
        # Navigate into nested dict
        d = data
        for part in parts[:-1]:
            if part in d and isinstance(d[part], dict):
                d = d[part]
            else:
                d = None
                break
        if d is not None and parts[-1] in d:
            env_val = val
            # Cast to type of existing value
            existing = d[parts[-1]]
            if isinstance(existing, bool):
                env_val = val.lower() in ("1", "true", "yes")
            elif isinstance(existing, int):
                env_val = int(val)
            elif isinstance(existing, float):
                env_val = float(val)
            d[parts[-1]] = env_val


# Singleton config instance
_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(_CONFIG_PATH) as f:
    _data = _yaml.safe_load(f)
_env_override(_data)
cfg = _ConfigStore(_data)


def reload_config(path=None):
    """Reload config from path (or default config.yaml), re-applying env overrides."""
    global cfg
    src = path or _CONFIG_PATH
    with open(src) as f:
        data = _yaml.safe_load(f)
    _env_override(data)
    cfg = _ConfigStore(data)
    return cfg
