#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/ab9738/dsrc}"
SIF="${SIF:-${WORKDIR}/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif}"
OVERLAY="${OVERLAY:-${WORKDIR}/dsrc_gpu_env.ext3}"
SING_BIN="${SING_BIN:-/share/apps/apptainer/bin/singularity}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/env_scripts/env_setup_scripts/logs}"
JOB_NAME="${JOB_NAME:-dsrc_pytest_overlay}"
POLL_SECONDS="${POLL_SECONDS:-30}"
SBATCH_SCRIPT="${LOG_DIR}/update_test_overlay_on_node_$(date +%Y%m%d_%H%M%S).sbatch"
JOB_LOG="${SBATCH_SCRIPT%.sbatch}.log"

mkdir -p "${LOG_DIR}"

for path in "${SIF}" "${OVERLAY}" "${WORKDIR}/env_scripts/run_env_tests.sh"; do
  if [[ ! -f "${path}" ]]; then
    echo "[error] required file not found: ${path}" >&2
    exit 1
  fi
done

cat >"${SBATCH_SCRIPT}" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=torch_pr_633_general
#SBATCH --output=${JOB_LOG}

set -euo pipefail

WORKDIR="${WORKDIR}"
SIF="${SIF}"
OVERLAY="${OVERLAY}"
SING_BIN="${SING_BIN}"

RUNTIME_BASE="\${SLURM_TMPDIR:-/tmp}/\${USER}_appt_\${SLURM_JOB_ID:-\$\$}"
mkdir -p "\${RUNTIME_BASE}"/{tmp,cache,session}
export APPTAINER_TMPDIR="\${RUNTIME_BASE}/tmp"
export APPTAINER_CACHEDIR="\${RUNTIME_BASE}/cache"
export APPTAINER_SESSIONDIR="\${RUNTIME_BASE}/session"
export TMPDIR="\${RUNTIME_BASE}/tmp"
export XDG_RUNTIME_DIR="\${RUNTIME_BASE}/session"

echo "[info] update/test job \${SLURM_JOB_ID:-N/A} on node \$(hostname)"
nvidia-smi

echo "[info] installing pytest into writable overlay"
"\${SING_BIN}" exec --nv --fakeroot \\
  --overlay "\${OVERLAY}:rw" \\
  "\${SIF}" \\
  /bin/bash -lc 'source /ext3/env.sh; python -m pip install pytest'

echo "[info] verifying pytest and simulation imports from read-only overlay"
"\${SING_BIN}" exec --nv \\
  --overlay "\${OVERLAY}:ro" \\
  "\${SIF}" \\
  /bin/bash -lc 'source /ext3/env.sh; python -m pytest --version; python -c "import pytest, gymnasium, highway_env, numpy"'

echo "[info] running repo pytest suite"
"\${SING_BIN}" exec --nv \\
  --overlay "\${OVERLAY}:ro" \\
  "\${SIF}" \\
  /bin/bash -lc 'source /ext3/env.sh; cd /scratch/ab9738/dsrc; python -m pytest tests'

echo "[info] running project interface validation"
"\${SING_BIN}" exec --nv \\
  --overlay "\${OVERLAY}:ro" \\
  "\${SIF}" \\
  /bin/bash -lc 'source /ext3/env.sh; cd /scratch/ab9738/dsrc; python scripts/validate_project_interface.py'

echo "[info] running existing environment integration tests without nested Slurm"
cd "\${WORKDIR}"
USE_SRUN=0 RUN_INSTALL_CHECK=1 REQUIRE_CUDA=1 bash env_scripts/run_env_tests.sh

echo "[info] overlay update and all tests completed"
SBATCH

JOB_ID="$(sbatch --parsable "${SBATCH_SCRIPT}")"
echo "[info] submitted job: ${JOB_ID}"
echo "[info] sbatch script: ${SBATCH_SCRIPT}"
echo "[info] job log: ${JOB_LOG}"

printed_running=0
while true; do
  queue_records="$(squeue -h -j "${JOB_ID}" -o "%.18i %.9P %.30j %.8T %.10M %.9l %.6D %R" 2>/dev/null || true)"
  if [[ -n "${queue_records}" ]]; then
    echo "             JOBID PARTITION                           NAME    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)"
    echo "${queue_records}"
    state="$(squeue -h -j "${JOB_ID}" -o "%T" 2>/dev/null | head -n 1 || true)"
    if [[ "${state}" == "RUNNING" && "${printed_running}" == "0" ]]; then
      echo "[info] job ${JOB_ID} is RUNNING; tests are executing on the allocated L40S node"
      echo "[info] tail progress with: tail -f ${JOB_LOG}"
      printed_running=1
    fi
    sleep "${POLL_SECONDS}"
    continue
  fi
  break
done

echo "[info] job left the queue; sacct summary:"
sacct -j "${JOB_ID}" --format=JobID,State,ExitCode,Elapsed,NodeList

final_state="$(sacct -n -P -j "${JOB_ID}" --format=State | head -n 1 | cut -d'|' -f1 | tr -d ' ')"
final_exit="$(sacct -n -P -j "${JOB_ID}" --format=ExitCode | head -n 1 | cut -d'|' -f1 | tr -d ' ')"
if [[ "${final_state}" != "COMPLETED" || "${final_exit}" != "0:0" ]]; then
  echo "[error] job ${JOB_ID} did not complete successfully; see ${JOB_LOG}" >&2
  exit 1
fi

echo "[info] job ${JOB_ID} completed successfully"
echo "[info] log: ${JOB_LOG}"
