"""Where the dataset and the scene index live."""

import json
import os
from pathlib import Path
from typing import Any

from longtail.data.pandaset import PandaSetScene

# ASSUMPTION A-35: the committed scene index lives in the package, since /data/ is the git-ignored dataset dir
INDEX_PATH = Path(__file__).with_name("index.json")


def data_root() -> Path:
    return Path(os.environ.get("LONGTAIL_DATA_DIR", "./data"))


def pandaset_root() -> Path:
    # ASSUMPTION A-33: PandaSet sequences live at $LONGTAIL_DATA_DIR/pandaset/<sequence id>/
    return data_root() / "pandaset"


def load_index(path: Path = INDEX_PATH) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def open_scene(scene_id: str, root: Path | None = None) -> PandaSetScene:
    return PandaSetScene(root or pandaset_root(), scene_id)
