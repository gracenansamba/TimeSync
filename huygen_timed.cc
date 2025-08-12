/*
Part 1:

One-time synchronization phase: processes exchange timestamps in a ring and use linear regression to estimate:
Alpha(drift rate): how fast a process clock deviates
Beta(offset): fixed time difference between clocks
Clock correction: each process adjusts its time estimate to align with the system average.
Drift tracking: regularly checks how desynchronized processes become over time.
Periodic resynchronization after K iterations (like resetting clocks once they fall out of sync).

Part 2 
Added the SVR 

*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>


#define NUM_PROBES 20
#define NUM_TRIALS 100
#define MASTER 0
#define DRIFT_MEASUREMENTS 10  
#define DRIFT_INTERVAL 0.1     

#define MAX_RANKS 512
#define MAX_RESYNC 20
double skew_table[MAX_RANKS][MAX_RESYNC];  // Global lookup table

void load_skew_table() {
    FILE *fp = fopen("svr_skew_table.csv", "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open skew table.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char line[128];
    fgets(line, sizeof(line), fp); // skip header

    int r, c;
    double skew;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "%d,%d,%lf", &r, &c, &skew) == 3) {
            if (r < MAX_RANKS && c < MAX_RESYNC)
                skew_table[r][c] = skew;
        }
    }
    fclose(fp);
}


double get_local_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1.0e-9;
}

void compute_clocksync_regression(double *timestamps, double *received, int count, double *alpha, double *beta) {
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


void resync_clocks(double *local_clock, int rank, int size, int resync_count) {
    double timestamps[NUM_PROBES], received_times[NUM_PROBES];
    int target = (rank + 1) % size;
    int source = (rank - 1 + size) % size;

   /* for (int i = 0; i < NUM_PROBES; i++) {
        timestamps[i] = get_local_time();
        MPI_Send(&timestamps[i], 1, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
        MPI_Recv(&received_times[i], 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }*/
    
double epsilon = 1e-6; // You can experiment with this value

    
char filename[128];
sprintf(filename, "logs_7/probes_p%d_rank%d.csv", size, rank);
FILE *fp = fopen(filename, "a");

fseek(fp, 0, SEEK_END);
long fsize = ftell(fp);
if (fsize == 0) {
    fprintf(fp, "rank,resync_count,tx_a,rx_b,tx_b,rx_a\n");
}
    
for (int i = 0; i < NUM_PROBES; i++) {
    double tx_a = 0, rx_b = 0, tx_b = 0, rx_a = 0;

    // Rank A (sender)
    if (rank % 2 == 0) {
        tx_a = get_local_time();
        MPI_Send(&tx_a, 1, MPI_DOUBLE, target, 0, MPI_COMM_WORLD);
        MPI_Recv(&rx_b, 1, MPI_DOUBLE, target, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        rx_a = get_local_time();

        // You already sent tx_a and received rx_b
        // But tx_b is known only to the receiver
        // Option 1 (simple workaround): Receive tx_b as payload extension
        MPI_Recv(&tx_b, 1, MPI_DOUBLE, target, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Apply the epsilon test
        double discrepancy = fabs((rx_a - rx_b) - (tx_b - tx_a));
        //if (discrepancy < epsilon) {
            // Save only pure probes
          fprintf(fp, "%d,%d,%.9f,%.9f,%.9f,%.9f\n", rank, resync_count, tx_a, rx_b, tx_b, rx_a);
        //}

    } else {  // Rank B (receiver)
        MPI_Recv(&tx_a, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        rx_b = get_local_time();
        tx_b = get_local_time();

        MPI_Send(&rx_b, 1, MPI_DOUBLE, source, 1, MPI_COMM_WORLD);
        MPI_Send(&tx_b, 1, MPI_DOUBLE, source, 2, MPI_COMM_WORLD);  // Send tx_b as extra packet
    }
}
   
    
    
/*
    // save time probes for each rank 
    //char filename[64];
    char filename[128];
    sprintf(filename, "logs_6/probes_p%d_rank%d.csv", size, rank);
    FILE *fp = fopen(filename, "a");  // append mode (keeps multiple resyncs)

    fseek(fp, 0, SEEK_END);
	long fsize = ftell(fp);
	if (fsize == 0) {
    		fprintf(fp, "rank,resync_count,local_time,received_time\n");
	}


    for (int i = 0; i < NUM_PROBES; i++) {
        fprintf(fp, "%d,%d,%.9f,%.9f\n",rank, resync_count, timestamps[i], received_times[i]);
    }
    fclose(fp);

 // SVR   
    if (resync_count < MAX_RESYNC && rank < MAX_RANKS) {
    double predicted_skew = skew_table[rank][resync_count];
    *local_clock -= predicted_skew;
    }
*/
/* //remove my original regressoion 
    double alpha, beta;
    compute_clocksync_regression(timestamps, received_times, NUM_PROBES, &alpha, &beta);

    double global_alphas[size], global_betas[size];
    MPI_Gather(&alpha, 1, MPI_DOUBLE, global_alphas, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Gather(&beta, 1, MPI_DOUBLE, global_betas, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    double avg_alpha = 0.0, avg_beta = 0.0;
    if (rank == MASTER) {
        for (int i = 0; i < size; i++) {
            avg_alpha += global_alphas[i];
            avg_beta  += global_betas[i];
        }
        avg_alpha /= size;
        avg_beta  /= size;
    }

    MPI_Bcast(&avg_alpha, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&avg_beta, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    *local_clock = (*local_clock * (1.0 - avg_alpha)) - avg_beta;
    */
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;

    if (rank == 0) {
        mkdir("logs_7", 0777);
    }
    int K=10;
    int resync_count = 0;
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

    
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("Rank %d is running on node %s\n", rank, hostname);
    
    /* ### Measure Synchronization Time ****/
    double sync_start = MPI_Wtime();

    if (use_sync) {
        load_skew_table(); 
    	resync_clocks(&local_clock, rank, size, resync_count);
	}
	else {
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
        double previous_drift = 0.0;

        MPI_Allreduce(&local_now, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_now, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (rank == MASTER) {
            double drift = global_max - global_min;
            printf("Drift check %d: drift = %.9f seconds\n", i, drift);
	    if (i > 0) {
        	double drift_rate = (drift - previous_drift) / DRIFT_INTERVAL;
        	printf("Estimated drift rate at check %d: %.9f seconds/second\n", i, drift_rate);
    	     }
    	   previous_drift = drift;
        }

        MPI_Barrier(MPI_COMM_WORLD);  
        struct timespec req = { 
		(int)DRIFT_INTERVAL,
	       	(long)((DRIFT_INTERVAL - (int)DRIFT_INTERVAL) * 1e9) };
       		 nanosleep(&req, NULL);    
        }

    /* ### Allreduce Benchmark ****/
    double *local_array = (double *) malloc(array_size * sizeof(double));
    double *global_array = (double *) malloc(array_size * sizeof(double));

    for (int i = 0; i < array_size; i++) {
        local_array[i] = rank + 1;
    }

    double total_time = 0.0;
  
    for (int i = 0; i < NUM_TRIALS; i++) {
    if (use_sync && i % K == 0 && i > 0) {
	resync_count++;
        resync_clocks(&local_clock, rank, size,resync_count);
    }

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

