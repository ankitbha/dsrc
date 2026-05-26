#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/ab9738/dsrc}"
OVERLAY_TEMPLATE="${OVERLAY_TEMPLATE:-/share/apps/overlay-fs-ext3/overlay-25GB-500K.ext3.gz}"
OVERLAY="${OVERLAY:-${WORKDIR}/dsrc_gpu_env.ext3}"
SIF="${SIF:-${WORKDIR}/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif}"
SING_BIN="${SING_BIN:-/share/apps/apptainer/bin/singularity}"
GPU_BURN="${GPU_BURN:-${WORKDIR}/gpu_burn.py}"
INSTALL_CHECK="${INSTALL_CHECK:-${WORKDIR}/env_scripts/env_setup_scripts/install_check.py}"
FORCE="${FORCE:-0}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/env_scripts/env_setup_scripts/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/setup_dsrc_gpu_env_$(date +%Y%m%d_%H%M%S).log}"
JOB_ID_FILE="${LOG_FILE%.log}.jobid"

mkdir -p "${LOG_DIR}"

status() {
  echo "$*"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    status "[error] ${label} not found: ${path}"
    exit 1
  fi
}

require_file "${OVERLAY_TEMPLATE}" "overlay template"
require_file "${SIF}" "SIF"
require_file "${GPU_BURN}" "gpu burn script"
require_file "${INSTALL_CHECK}" "install check script"

status "[info] writing setup log to ${LOG_FILE}"
status "[info] requesting one L40S GPU node for overlay setup"

if {
  set -x
  rm -f "${JOB_ID_FILE}"

  if [[ -e "${OVERLAY}" ]]; then
    if [[ "${FORCE}" == "1" ]]; then
      rm -f "${OVERLAY}"
    else
      echo "[error] overlay already exists: ${OVERLAY}" >&2
      echo "[info] rerun with FORCE=1 to recreate it from scratch" >&2
      exit 1
    fi
  fi

  echo "[info] creating fresh overlay: ${OVERLAY}"
  gzip -dc "${OVERLAY_TEMPLATE}" > "${OVERLAY}"

  srun \
    --nodes=1 \
    --cpus-per-task=4 \
    --mem=32GB \
    --time=6:00:00 \
    --partition=l40s_public \
    --gres=gpu:l40s:1 \
    --account=torch_pr_633_general \
    bash -lc "
set -euo pipefail

RUNTIME_BASE=\"\${SLURM_TMPDIR:-/tmp}/\${USER}_appt_\${SLURM_JOB_ID:-\$\$}\"
mkdir -p \"\${RUNTIME_BASE}\"/{tmp,cache,session}
export APPTAINER_TMPDIR=\"\${RUNTIME_BASE}/tmp\"
export APPTAINER_CACHEDIR=\"\${RUNTIME_BASE}/cache\"
export APPTAINER_SESSIONDIR=\"\${RUNTIME_BASE}/session\"
export TMPDIR=\"\${RUNTIME_BASE}/tmp\"
export XDG_RUNTIME_DIR=\"\${RUNTIME_BASE}/session\"

echo \"[info] setup job \${SLURM_JOB_ID:-N/A} on node \$(hostname)\"
echo \"\${SLURM_JOB_ID:-N/A}\" > \"${JOB_ID_FILE}\"
nvidia-smi

\"${SING_BIN}\" exec --nv \
  --fakeroot \
  --overlay \"${OVERLAY}:rw\" \
  \"${SIF}\" \
  /bin/bash -lc '
set -euo pipefail

export HOME=/ext3/home
export TMPDIR=/ext3/tmp
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p \"\${HOME}\" \"\${TMPDIR}\"

MINIFORGE=/ext3/miniforge3
ENV_SH=/ext3/env.sh
INSTALLER=\"\${TMPDIR}/Miniforge3-Linux-x86_64.sh\"
BURN_PID=\"\"

cleanup() {
  if [[ -n \"\${BURN_PID}\" ]] && kill -0 \"\${BURN_PID}\" >/dev/null 2>&1; then
    echo \"[info] stopping gpu_burn.py pid \${BURN_PID}\"
    kill \"\${BURN_PID}\" || true
    wait \"\${BURN_PID}\" || true
  fi
}
trap cleanup EXIT

echo \"[info] installing Miniforge into \${MINIFORGE}\"
wget --no-check-certificate -O \"\${INSTALLER}\" \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash \"\${INSTALLER}\" -b -p \"\${MINIFORGE}\"
rm -f \"\${INSTALLER}\"

source \"\${MINIFORGE}/etc/profile.d/conda.sh\"
conda activate base
conda install -y python=3.11

cat > \"\${ENV_SH}\" <<\"EOF\"
#!/bin/bash

unset -f which 2>/dev/null || true

source /ext3/miniforge3/etc/profile.d/conda.sh
conda activate base
export PATH=/ext3/miniforge3/bin:\$PATH
export SDL_VIDEODRIVER=dummy
export PYOPENGL_PLATFORM=egl
export MPLBACKEND=Agg
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
EOF
chmod +x \"\${ENV_SH}\"
source \"\${ENV_SH}\"

python -m pip install --upgrade pip setuptools wheel

echo \"[info] installing CUDA 11.8 PyTorch wheels\"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python - <<\"PY\"
import torch
print(\"torch:\", torch.__version__)
print(\"cuda available:\", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(\"CUDA is not available to PyTorch\")
print(\"cuda device:\", torch.cuda.get_device_name(torch.cuda.current_device()))
PY

echo \"[info] starting gpu_burn.py while remaining packages install\"
python \"${GPU_BURN}\" &
BURN_PID=\$!
sleep 10
if ! kill -0 \"\${BURN_PID}\" >/dev/null 2>&1; then
  echo \"[error] gpu_burn.py did not stay running\" >&2
  wait \"\${BURN_PID}\" || true
  exit 1
fi

echo \"[info] installing highway-env and RL dependencies into /ext3/miniforge3 base\"
python -m pip install \
  highway-env gymnasium pettingzoo supersuit \
  stable-baselines3 sb3-contrib \
  numpy scipy pandas matplotlib pyyaml tqdm networkx numba \
  tensorboard wandb rich h5py pyarrow \
  pygame imageio imageio-ffmpeg moviepy opencv-python-headless

echo \"[info] checking overlay capacity\"
du -sh /ext3
find /ext3 | wc -l

echo \"[info] running install check\"
source \"\${ENV_SH}\"
python \"${INSTALL_CHECK}\"
'
"
} >>"${LOG_FILE}" 2>&1; then
  if [[ -f "${JOB_ID_FILE}" ]]; then
    status "[info] slurm job id: $(cat "${JOB_ID_FILE}")"
  fi
  status "[info] setup complete: ${OVERLAY}"
  status "[info] log: ${LOG_FILE}"
  status "[info] run read-only verification with:"
  status "${SING_BIN} exec --nv --overlay ${OVERLAY}:ro ${SIF} /bin/bash -lc 'source /ext3/env.sh; python ${INSTALL_CHECK}'"
else
  if [[ -f "${JOB_ID_FILE}" ]]; then
    status "[info] slurm job id: $(cat "${JOB_ID_FILE}")"
  fi
  status "[error] setup failed; see log: ${LOG_FILE}"
  exit 1
fi
