#!/bin/bash
# SBATCH -n 512
#SBATCH -N 6
# SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time 120

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
BASE_ARRAY_SIZE=100000  
# PROCS=(16 32 64 128 256 512)  # weak scaling list
# PROCS=(16 32 64)
PROCS=(6)
EXE=./huygen_timed

# Main loop
for sync in 0 1; do
    for p in "${PROCS[@]}"; do
        size=$((BASE_ARRAY_SIZE * p))

        echo "Running: sync=$sync, procs=$p, total_array=$size (=$BASE_ARRAY_SIZE per rank)"

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

