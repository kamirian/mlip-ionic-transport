#!/bin/bash
# Submit one SLURM job per study manifest (analysis only; reads existing trajectories).
# Usage (from the project root on the cluster): bash scripts/submit_all.sh [manifest ...]
set -euo pipefail
ACCOUNT=${ACCOUNT:-YOUR_ACCOUNT}
ENV_ACTIVATE=${ENV_ACTIVATE:-$HOME/scratch/MACE_env/bin/activate}
manifests=("$@")
[ ${#manifests[@]} -eq 0 ] && manifests=(manifests/*.json)
for m in "${manifests[@]}"; do
  name=$(basename "$m" .json)
  sbatch <<EOS
#!/bin/bash
#SBATCH --job-name=iondiff_${name}
#SBATCH -A ${ACCOUNT}
#SBATCH -p standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH --output=logs/${name}_%j.out
source ${ENV_ACTIVATE}
cd $(pwd)
python scripts/run_study.py ${m} results/${name}.json --workers 16
EOS
done
