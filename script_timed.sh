#!/bin/bash
#SBATCH -N 1
#SBATCH --time 120

# Optional: Clean up old results
rm -f results_d.csv
echo "sync_mode,processes,array_size,doubles_per_process,sync_time,avg_time" > results_d.csv

rm -f drift_results.csv
echo "sync_mode,processes,array_size,drift_check,drift_seconds" > drift_results.csv

module load openmpi/4.1.2

# Experiment sweep
TRIALS=100
ARRAY_SIZES=(131072 262144 524288 1000000 2000000)  # 128K to 2M doubles per rank
PROCS=(2 4 8 16 32)  # up to 112 for single-node on Dane (we used 32 only)

EXE=./huygen_timed

for sync in 0 1; do
    for size in "${ARRAY_SIZES[@]}"; do
        for p in "${PROCS[@]}"; do
            echo "Running: sync=$sync, procs=$p, base_array=$size"

            OUTPUT=$(srun -n $p --exclusive $EXE $sync $size 2>&1)

            # Capture synchronization time
            SYNC_TIME=$(echo "$OUTPUT" | awk '/Synchronization took/ {print $4}')
            [[ -z "$SYNC_TIME" ]] && SYNC_TIME="N/A"

            # Capture average Allreduce time (5th field is the time value)
            AVG_TIME=$(echo "$OUTPUT" | awk '/Avg MPI_Allreduce time/ {print $5}')
            [[ -z "$AVG_TIME" ]] && AVG_TIME="N/A"

            # Save to results_d.csv
            echo "$([[ $sync -eq 0 ]] && echo 'ClockSync' || echo 'BarrierOnly'),$p,$size,$((size * p)),$SYNC_TIME,$AVG_TIME" >> results_d.csv

            # Extract and save drift measurements
            echo "$OUTPUT" | grep "Drift check" | while read -r line; do
                DRIFT_CHECK=$(echo "$line" | awk '{print $3}' | tr -d ':')
                DRIFT_SECONDS=$(echo "$line" | awk '{print $6}')
                echo "$([[ $sync -eq 0 ]] && echo 'ClockSync' || echo 'BarrierOnly'),$p,$size,$DRIFT_CHECK,$DRIFT_SECONDS" >> drift_results.csv
            done

        done
    done
done

echo "Sweep completed. Results saved to results_d.csv and drift_results.csv"

