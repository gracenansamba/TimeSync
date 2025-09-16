#!/bin/bash

#SBATCH --job-name=huygens-bench

#SBATCH --account=askjellum

#SBATCH --partition=batch-warp

#SBATCH --nodes=6

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1

#SBATCH --mem-per-cpu=4G

#SBATCH --time=02:00:00



# --- config you likely tweak ---

TRIALS=100

PROCS=(2 4 6)                     # sweep of MPI processes

EXE=./huygen_inbuilt_SVMg         # new binary we built

# --------------------------------



# Load MPI (adjust if your env differs)

spack load openmpi@4.1.6



# Output CSVs

echo "mode,processes,per_rank_doubles,total_doubles,trials,avg_latency_s" > results_strong.csv

echo "mode,processes,per_rank_doubles,total_doubles,trials,avg_latency_s" > results_weak.csv



# -------- helper to run one case and append a CSV row --------

run_case () {

  local mode="$1"    # "barrier" or "svm"

  local p="$2"       # processes

  local per_rank="$3"

  local total="$4"

  local csv="$5"



  echo ">> Running: mode=$mode, procs=$p, per-rank=$per_rank, total=$total, trials=$TRIALS"



  # Grab program output. We tee to a log per run for later inspection.

  local log="run_${mode}_p${p}_n${per_rank}.log"

  OUTPUT=$(srun -n "$p" --exclusive "$EXE" --mode="$mode" --trials="$TRIALS" --size="$per_rank" 2>&1 | tee "$log")



  # Parse the reported average latency (format: 'Avg Allreduce latency ...: <num> s')

  AVG=$(echo "$OUTPUT" | awk '/Avg Allreduce latency/ {print $(NF-1); exit}')

  [[ -z "$AVG" ]] && AVG="NA"



  echo "${mode},${p},${per_rank},${total},${TRIALS},${AVG}" >> "$csv"

}



# ==================== STRONG SCALING ====================

# Fixed global size; per-rank = GLOBAL / P

GLOBAL=160000

for p in "${PROCS[@]}"; do

  per_rank=$(( GLOBAL / p ))

  # Barrier vs SVM-deadline

  run_case barrier "$p" "$per_rank" "$GLOBAL" results_strong.csv

  run_case svm     "$p" "$per_rank" "$GLOBAL" results_strong.csv

done



# ================= “WEAK-SCALING STYLE” FOR COLLECTIVES =================

# Fixed per-rank size; total grows with P

PER_RANK_FIXED=8

for p in "${PROCS[@]}"; do

  total=$(( PER_RANK_FIXED * p ))

  run_case barrier "$p" "$PER_RANK_FIXED" "$total" results_weak.csv

  run_case svm     "$p" "$PER_RANK_FIXED" "$total" results_weak.csv

done



echo "Done."

echo "Strong-scaling results -> results_strong.csv"

echo "Weak-style collective results -> results_weak.csv"


