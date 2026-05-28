#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate ${CONDA_ENV:-qwenvln}

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN must be set in the environment." >&2
  exit 1
fi

REPO_ID=${REPO_ID:-InternRobotics/InternData-N1}
REVISION=${REVISION:-v0.5-mini}
SCENE_PATH=${SCENE_PATH:?Set SCENE_PATH, for example vln_n1/traj_data/<dataset>/<scene>}
LOCAL_DIR=${LOCAL_DIR:-/mnt/nas5/xiangchen/VLNData/InternData-N1}

python - "$REPO_ID" "$REVISION" "$SCENE_PATH" "$LOCAL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, revision, scene_path, local_dir = sys.argv[1:]
scene_path = scene_path.strip("/")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision=revision,
    local_dir=local_dir,
    token=True,
    allow_patterns=[
        f"{scene_path}/data/**",
        f"{scene_path}/meta/**",
        f"{scene_path}/videos/**",
    ],
)
print(f"Downloaded {scene_path} into {local_dir}")
PY
