"""Build SAM3 auto-concepts from the label registry.

Converts object_registry.json → auto_concepts.json. Each enabled label gets its
first prompt by default. Alternative prompts stay in the registry for sweep.py.

Usage (module):
    from rgbd2ply.concepts import build_concepts, load_registry
"""

import json
from pathlib import Path

try:
    from .config import cfg
except ImportError:
    from config import cfg


def load_registry(path: str | Path | None = None) -> dict:
    """Load object_registry.json. Returns the full dict (has 'labels' key)."""
    src = Path(path) if path else Path(cfg.paths.registry)
    with open(src) as f:
        return json.load(f)


def build_concepts(registry_path: str | Path | None = None) -> list[dict]:
    """Generate SAM3 concept list from the registry.

    Each entry: {"label": int, "name": str, "prompt": str, "max_instances": int}.
    Objects are sorted first, hand last (wins overlap in paint order).
    """
    reg = load_registry(registry_path)
    concepts = []
    for item in reg.get("labels", []):
        if not item.get("auto", True):
            continue
        prompts = item.get("prompts") or [item.get("name_en", "")]
        if not prompts:
            continue
        concepts.append({
            "label": int(item["id"]),
            "name": item.get("key", f"label_{item['id']}"),
            "prompt": prompts[0],
            "max_instances": int(item.get("max_instances", 1)),
        })
    # Hand (label 1) always last so it wins overlaps
    concepts.sort(key=lambda c: c["label"] == 1)
    return concepts


def save_concepts(concepts: list[dict], out_path: str | Path):
    """Write concepts list to a JSON file."""
    data = {"concepts": concepts}
    Path(out_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )


# Standalone usage: python -m rgbd2ply run <source> --steps concepts
