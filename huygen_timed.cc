#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define NUM_PROBES 10
#define NUM_TRIALS 100
#define MASTER 0
#define DRIFT_MEASUREMENTS 10  // how many times to check drift
#define DRIFT_INTERVAL 0.1     // seconds between drift checks

double get_local_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1.0e-9;
}

void compute_drift(double *timestamps, double *received, int count, double *alpha, double *beta) {
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (int i = 0; i < count; i++) {
        sum_x += timestamps[i];
        sum_y += received[i];
        sum_xx += timestamps[i] * timestamps[i];
        sum_xy += timestamps[i] * received[i];
    }
    double denominator = (count * sum_xx - sum_x * sum_x);
    if (fabs(denominator) < 1e-10) {
        *alpha = 0.0;
        *beta = 0.0;
        return;
    }
    *alpha = (count * sum_xy - sum_x * sum_y) / denominator;
    *beta = (sum_y - (*alpha) * sum_x) / count;
    if (isnan(*alpha)) *alpha = 0.0;
    if (isnan(*beta)) *beta = 0.0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int use_sync = 1;
    if (argc > 1 && atoi(argv[1]) == 1) {
        use_sync = 0;
    }

    int base_array_size = 1000000;
    if (argc > 2) {
        base_array_size = atoi(argv[2]);
    }

    int array_size = base_array_size * size;

    if (rank == MASTER) {
        printf("Running with %d processes\n", size);
        printf("Each process reducing %d doubles (%.2f MB)\n",
               array_size, array_size * sizeof(double) / 1e6);
    }

    double local_clock = get_local_time() + ((rank == 0) ? 0.05 : -0.05);

    /* ### Measure Synchronization Time ****/
    double sync_start = MPI_Wtime();

    if (use_sync) {
        double timestamps[NUM_PROBES], received_times[NUM_PROBES];
        int target = (rank + 1) % size;
        int source = (rank - 1 + size) % size;
        for (int i = 0; i < NUM_PROBES; i++) {
            timestamps[i] = get_local_time();
            MPI_Send(&timestamps[i], 1, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
            MPI_Recv(&received_times[i], 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        double alpha, beta;
        compute_drift(timestamps, received_times, NUM_PROBES, &alpha, &beta);

        double global_alphas[size];
        MPI_Gather(&alpha, 1, MPI_DOUBLE, global_alphas, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

        double avg_alpha = 0.0;
        if (rank == MASTER) {
            for (int i = 0; i < size; i++) avg_alpha += global_alphas[i];
            avg_alpha /= size;
        }
        MPI_Bcast(&avg_alpha, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
        local_clock -= avg_alpha;
    } else {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double sync_end = MPI_Wtime();
    double sync_duration = sync_end - sync_start;

    if (rank == MASTER) {
        printf("[%s] Synchronization took %.9f seconds\n",
               use_sync ? "ClockSync" : "BarrierOnly", sync_duration);
    }

    /* ### Start Drift Trackin ****/
    for (int i = 0; i < DRIFT_MEASUREMENTS; i++) {
        double local_now = MPI_Wtime();
        double global_min, global_max;

        MPI_Allreduce(&local_now, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_now, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == MASTER) {
            double drift = global_max - global_min;
            printf("Drift check %d: drift = %.9f seconds\n", i, drift);
        }

        MPI_Barrier(MPI_COMM_WORLD);  
	struct timespec req = { (int)DRIFT_INTERVAL, (DRIFT_INTERVAL - (int)DRIFT_INTERVAL) * 1e9 };
        nanosleep(&req, NULL);     }

    /* ### Normal Allreduce Benchmark ****/
    double *local_array = (double *) malloc(array_size * sizeof(double));
    double *global_array = (double *) malloc(array_size * sizeof(double));

    for (int i = 0; i < array_size; i++) {
        local_array[i] = rank + 1;
    }

    double total_time = 0.0;
    for (int i = 0; i < NUM_TRIALS; i++) {
        double start = MPI_Wtime();
        MPI_Allreduce(local_array, global_array, array_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double end = MPI_Wtime();
        total_time += (end - start);
    }

    free(local_array);
    free(global_array);

    double avg_time = total_time / NUM_TRIALS;
    double global_avg_time;
    MPI_Reduce(&avg_time, &global_avg_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
        global_avg_time /= size;
        printf("[%s] Avg MPI_Allreduce time: %.9f seconds (over %d trials)\n",
               use_sync ? "ClockSync" : "BarrierOnly", global_avg_time, NUM_TRIALS);
    }

    MPI_Finalize();
    return 0;
}

