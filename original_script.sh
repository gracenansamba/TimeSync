#!/bin/bash

#SBATCH --job-name=huygens-weak

#SBATCH --account=askjellum          # REQUIRED on this cluster

#SBATCH --partition=batch-warp        # or 'debug' (<=30m) / 'any-interactive' (<=2h)

#SBATCH --nodes=6                     # matches PROCS=(6) below

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1

#SBATCH --mem-per-cpu=4G

#SBATCH --time=02:00:00



set -euo pipefail



# Fresh results files

echo "sync_mode,processes,array_size,doubles_per_process,sync_time,avg_time" > results_d_1.csv

echo "sync_mode,processes,array_size,drift_check,drift_seconds" > drift_results_1.csv

echo "sync_mode,processes,array_size,drift_check,drift_rate" > drift_rate_results_1.csv



# MPI env

module purge

spack load openmpi@4.1.6
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Parameters

TRIALS=100

BASE_ARRAY_SIZE=100000

PROCS=(2 4 6) # IMPORTANT: with --nodes=6 and --ntasks-per-node=1, p must be <= 6

EXE=./huygen_timed



# Main loop

for sync in 0 1; do

  for p in "${PROCS[@]}"; do

    size=$((BASE_ARRAY_SIZE * p))

    echo "Running: sync=$sync, procs=$p, total_array=$size (=$BASE_ARRAY_SIZE per rank)"



    # Use exactly the tasks you allocated: 1 per node x 6 nodes => up to 6 tasks

    OUTPUT=$(srun -n "$p" --exclusive "$EXE" "$sync" "$BASE_ARRAY_SIZE" 2>&1 || true)



    SYNC_TIME=$(echo "$OUTPUT" | awk '/Synchronization took/ {print $4; exit}')

    [[ -z "${SYNC_TIME:-}" ]] && SYNC_TIME="N/A"



    AVG_TIME=$(echo "$OUTPUT" | awk '/Avg MPI_Allreduce time/ {print $5; exit}')

    [[ -z "${AVG_TIME:-}" ]] && AVG_TIME="N/A"



    echo "$([[ $sync -eq 0 ]] && echo 'ClockSync' || echo 'BarrierOnly'),$p,$size,$BASE_ARRAY_SIZE,$SYNC_TIME,$AVG_TIME" >> results_d_1.csv



    echo "$OUTPUT" | grep -F "Drift check" | while read -r line; do

      DRIFT_CHECK=$(echo "$line" | awk '{print $3}' | tr -d ':')

      DRIFT_SECONDS=$(echo "$line" | awk '{print $6}')

      echo "$([[ $sync -eq 0 ]] && echo 'ClockSync' || echo 'BarrierOnly'),$p,$size,$DRIFT_CHECK,$DRIFT_SECONDS" >> drift_results_1.csv

    done



    echo "$OUTPUT" | grep -F "Estimated drift rate" | while read -r line; do

      DRIFT_CHECK=$(echo "$line" | awk '{print $6}' | tr -d ':')

      DRIFT_RATE=$(echo "$line" | awk '{print $(NF-1)}')

      echo "$([[ $sync -eq 0 ]] && echo 'ClockSync' || echo 'BarrierOnly'),$p,$size,$DRIFT_CHECK,$DRIFT_RATE" >> drift_rate_results_1.csv

    done

  done

done



echo "Sweep completed. Results saved to results_d_1.csv, drift_results_1.csv, drift_rate_results_1.csv"


