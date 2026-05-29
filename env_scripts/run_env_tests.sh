#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/ab9738/dsrc}"
SIF="${SIF:-${WORKDIR}/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${WORKDIR}/dsrc_gpu_env.ext3}"
SING_BIN="${SING_BIN:-/share/apps/apptainer/bin/singularity}"
TEST_SCRIPT="${TEST_SCRIPT:-${WORKDIR}/env_scripts/env_tests/test_highway_env_integration.py}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/env_scripts/env_tests/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/env_tests_$(date +%Y%m%d_%H%M%S).log}"
USE_SRUN="${USE_SRUN:-1}"
RUN_INSTALL_CHECK="${RUN_INSTALL_CHECK:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

mkdir -p "${LOG_DIR}"

for path in "${SIF}" "${OVERLAY}" "${TEST_SCRIPT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[error] required file not found: ${path}" >&2
    exit 1
  fi
done

echo "[info] writing env test log to ${LOG_FILE}"

run_in_container() {
  local use_nv_flag="$1"
  local use_fakeroot="$2"
  "${SING_BIN}" exec ${use_nv_flag} ${use_fakeroot} \
    --overlay "${OVERLAY}:ro" \
    "${SIF}" \
    /bin/bash -lc "
set -euo pipefail
source /ext3/env.sh
cd \"${WORKDIR}\"
echo \"[info] python: \$(which python)\"
if [[ \"${RUN_INSTALL_CHECK}\" == \"1\" ]]; then
  REQUIRE_CUDA=\"${REQUIRE_CUDA}\" python -u \"${WORKDIR}/env_scripts/env_setup_scripts/install_check.py\"
fi
python -u \"${TEST_SCRIPT}\"
"
}

if {
  if [[ "${USE_SRUN}" == "1" ]]; then
    srun \
      --nodes=1 \
      --cpus-per-task=4 \
      --mem=16GB \
      --time=00:30:00 \
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

echo \"[info] env test job \${SLURM_JOB_ID:-N/A} on node \$(hostname)\"
nvidia-smi
$(declare -f run_in_container)
run_in_container --nv --fakeroot
"
  else
    echo "[info] running env tests without srun"
    if [[ "${REQUIRE_CUDA}" == "1" ]]; then
      run_in_container --nv --fakeroot
    else
      run_in_container "" ""
    fi
  fi
} >"${LOG_FILE}" 2>&1; then
  echo "[info] env tests passed"
  echo "[info] log: ${LOG_FILE}"
else
  echo "[error] env tests failed; see log: ${LOG_FILE}" >&2
  exit 1
fi
