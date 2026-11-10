"""
Locations of the data and results.

Every location has a default inside the repository and can be changed with an environment
variable PRHCL_<NAME> (e.g. PRHCL_DATA_DIR) or in a local_paths.env file in the repository
root.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATHS_FILE = ROOT / "local_paths.env"


def _read_local_paths(path):
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                name, _, value = line.partition("=")
                values[name.strip()] = value.strip().strip("\"'")
    return values


_LOCAL = _read_local_paths(LOCAL_PATHS_FILE)


def _path(name, default):
    """PRHCL_<name> from the environment, else from local_paths.env, else the default."""
    value = os.environ.get(f"PRHCL_{name}", _LOCAL.get(f"PRHCL_{name}"))
    return Path(value).expanduser() if value else Path(default)


# datasets
DATA_DIR = _path("DATA_DIR", ROOT / "data")
COCO_DIR = _path("COCO_DIR", DATA_DIR / "coco")  # annotations/captions_<split>2017.json, images/<split>2017/
CC3M_RECAPTIONED_DIR = _path("CC3M_RECAPTIONED_DIR", DATA_DIR / "cc3m_recaptioned")  # cc3m-train-XXXX.tar shards
DOCCI_DIR = _path("DOCCI_DIR", DATA_DIR / "docci")  # docci_descriptions.jsonlines, images/
DOCCI_WITH_SUMMARIES_DIR = _path("DOCCI_WITH_SUMMARIES_DIR", DATA_DIR / "docci_with_summaries")  # generate_docci_summaries.py

# results
RESULTS_DIR = _path("RESULTS_DIR", ROOT / "results")
FEATURES_DIR = _path("FEATURES_DIR", RESULTS_DIR / "features")
ALIGNMENT_DIR = _path("ALIGNMENT_DIR", RESULTS_DIR / "alignment")
PLOTS_DIR = _path("PLOTS_DIR", RESULTS_DIR / "plots")
BENCHMARK_DIR = _path("BENCHMARK_DIR", ROOT / "benchmark_results")  # lm-eval json files


if __name__ == "__main__":
    # print locations for shell scripts, e.g. ALIGNMENT_DIR=$(python -m src.paths ALIGNMENT_DIR)
    import sys

    names = sys.argv[1:] or [n for n in dir() if n.isupper() and n.endswith(("_DIR", "ROOT"))]
    for name in names:
        print(globals()[name] if len(sys.argv) > 1 else f"{name}={globals()[name]}")
