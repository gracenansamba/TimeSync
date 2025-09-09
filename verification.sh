#!/bin/bash
# SBATCH -n 512
#SBATCH -N 8
# SBATCH -N 1
#SBATCH --ntasks-per-node=1
# SBATCH --time 120
#SBATCH --time=01:30:00

# rm -f results_d.csv
echo "sync_mode,processes,array_size,doubles_per_process,sync_time,avg_time" > results_d_1.csv

# rm -f drift_results.csv
echo "sync_mode,processes,array_size,drift_check,drift_seconds" > drift_results_1.csv

# rm -f drift_rate_results.csv
echo "sync_mode,processes,array_size,drift_check,drift_rate" > drift_rate_results_1.csv

# Load MPI module
module load openmpi/4.1.2

# Parameters
TRIALS=100
GLOBAL=160000                   # total doubles to reduce (fixed)
PROCS=(8)
EXE=./huygen_timed

# Main loop (strong scaling)
for sync in 0 1; do
  for p in "${PROCS[@]}"; do
    BASE_ARRAY_SIZE=$(( GLOBAL / p ))   # per-rank problem size
    TOTAL=$GLOBAL                       # total is fixed
    echo "Running: sync=$sync, procs=$p, total=$TOTAL, per-rank=$BASE_ARRAY_SIZE"

        OUTPUT=$(srun -n $p --exclusive $EXE $sync $BASE_ARRAY_SIZE 2>&1)

        SYNC_TIME=$(echo "$OUTPUT" | awk '/Synchronization took/ {print $4}')
        [[ -z "$SYNC_TIME" ]] && SYNC_TIME="N/A"

        AVG_TIME=$(echo "$OUTPUT" | awk '/Avg MPI_Allreduce time/ {print $5}')
        [[ -z "$AVG_TIME" ]] && AVG_TIME="N/A"

        echo "$([[ $sync -eq 0 ]] && echo 'ClockSync' || echo 'BarrierOnly'),$p,$size,$BASE_ARRAY_SIZE,$SYNC_TIME,$AVG_TIME" >> results_d_1.csv
        echo "$OUTPUT" | grep "Drift check" | while read -r line; do
            DRIFT_CHECK=$(echo "$line" | awk '{print $3}' | tr -d ':')
            DRIFT_SECONDS=$(echo "$line" | awk '{print $6}')
            echo "$([[ $sync -eq 0 ]] && echo 'ClockSync' || echo 'BarrierOnly'),$p,$size,$DRIFT_CHECK,$DRIFT_SECONDS" >> drift_results_1.csv
        done
	 echo "$OUTPUT" | grep "Estimated drift rate" | while read -r line; do
            DRIFT_CHECK=$(echo "$line" | awk '{print $6}' | tr -d ':')
            DRIFT_RATE=$(echo "$line" | awk '{print $(NF-1)}')
            echo "$([[ $sync -eq 0 ]] && echo 'ClockSync' || echo 'BarrierOnly'),$p,$size,$DRIFT_CHECK,$DRIFT_RATE" >> drift_rate_results_1.csv
        done
    done
done

echo "Sweep completed. Results saved to results_d.csv, drift_results.csv and drift_rate_results.csv"
