# Local repository paths - no external dependencies
# Get the script directory (repo root)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PATH_TO_SRC_DATA=$REPO_ROOT/data
export PATH_TO_SAMPLED_DATA=$REPO_ROOT/data
export PATH_TO_GENERAL_BACKBONE=$REPO_ROOT/data
export PATH_TO_FEATURE=$REPO_ROOT/data
export PATH_TO_PRETRAINED_MODEL=$REPO_ROOT/data
export PATH_TO_LOG=$REPO_ROOT/logs