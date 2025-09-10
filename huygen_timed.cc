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

Part 3
Added the SVM model and used it to correct time 

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
#define TAG_TXA 0
#define TAG_RXB 1
#define TAG_TXB 2
#define TAG_RXA 3
#define MAX_RANKS 512
#define MAX_RESYNC 20

// #define MAX_RESYNC 256
static double g_s[MAX_RESYNC], g_c[MAX_RESYNC];
static unsigned char g_have[MAX_RESYNC];  // 1 if row present

static inline double map_B_to_A(double tB, int k){
    if (k < 0 || k >= MAX_RESYNC || !g_have[k]) return tB; // no-op if missing
    return (tB - g_c[k]) / g_s[k];
}

double get_local_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    //clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1.0e-9;
}
/*
static inline double aligned_now(int k) {
    double t_local = get_local_time();              // SAME base as your probes
    if (k >= 0 && k < MAX_RESYNC && g_have[k]) {
        return (t_local - g_c[k]) / g_s[k];         // t_aligned = (t - c)/s
    }
    return t_local;                                  // fallback if model missing
}
*/
static inline double aligned_now(int k) {
    double t = get_local_time();
    if (k >= 0 && k < MAX_RESYNC && g_have[k]) {
        double s = g_s[k], c = g_c[k];
        // Reject bogus/unstable models
        if (!isfinite(s) || !isfinite(c) || s <= 0.5 || s >= 1.5) {
            return t; // fallback to raw time
        }
        return (t - c) / s;
    }
    return t;
}




static double g_epsilon = 5e-6;  // seconds (default 5 µs)



/* this part is for the SVR formula but didnt yield speedup in the collective operation*/
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


/* This is the Least square method*/

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
    double epsilon = 1e-6; 
    char filename[128];
    sprintf(filename, "logs-9-9-rwork/probes_p%d_rank%d.csv", size, rank);
    FILE *fp = fopen(filename, "a");
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);

    if (ftell(fp) == 0) {
        fprintf(fp, "rank,resync_count,tx_a,rx_b,tx_b,rx_a,keep_ab,keep_ba\n");
    }

    double prev_tx_a = 0.0, prev_rx_b = 0.0;  // A->B 
    double prev_tx_b = 0.0, prev_rx_a = 0.0;  // B->A 

    for (int i = 0; i < NUM_PROBES; i++) {
        double tx_a = 0.0, rx_b = 0.0, tx_b = 0.0, rx_a = 0.0;
        int keep_ab = 1, keep_ba = 1;  // first probe defaults to keep

        if ((rank % 2) == 0) {
            tx_a = get_local_time();
            MPI_Send(&tx_a, 1, MPI_DOUBLE, target, TAG_TXA, MPI_COMM_WORLD);
            MPI_Recv(&rx_b, 1, MPI_DOUBLE, target, TAG_RXB, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            rx_a = get_local_time();  // when A receives B's rx_b back

            MPI_Recv(&tx_b, 1, MPI_DOUBLE, target, TAG_TXB, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&rx_a, 1, MPI_DOUBLE, target, TAG_RXA, MPI_COMM_WORLD);
	
	    if (g_have[resync_count]) {
           	 rx_b = map_B_to_A(rx_b, resync_count);
            	tx_b = map_B_to_A(tx_b, resync_count);
            }

            if (i > 0) {
                double d_txAB = tx_a - prev_tx_a;
                double d_rxAB = rx_b - prev_rx_b;
                keep_ab = (fabs(d_rxAB - d_txAB) <= g_epsilon);

                double d_txBA = tx_b - prev_tx_b;
                double d_rxBA = rx_a - prev_rx_a;
                keep_ba = (fabs(d_rxBA - d_txBA) <= g_epsilon);
            }

            prev_tx_a = tx_a;
            prev_rx_b = rx_b;
            prev_tx_b = tx_b;
            prev_rx_a = rx_a;

        } else {
            MPI_Recv(&tx_a, 1, MPI_DOUBLE, source, TAG_TXA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            rx_b = get_local_time();

            tx_b = get_local_time();
            MPI_Send(&rx_b, 1, MPI_DOUBLE, source, TAG_RXB, MPI_COMM_WORLD);
            MPI_Send(&tx_b, 1, MPI_DOUBLE, source, TAG_TXB, MPI_COMM_WORLD);

            MPI_Recv(&rx_a, 1, MPI_DOUBLE, source, TAG_RXA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (i > 0) {
                double d_txAB = tx_a - prev_tx_a;
                double d_rxAB = rx_b - prev_rx_b;
                keep_ab = (fabs(d_rxAB - d_txAB) <= g_epsilon);

                double d_txBA = tx_b - prev_tx_b;
                double d_rxBA = rx_a - prev_rx_a;
                keep_ba = (fabs(d_rxBA - d_txBA) <= g_epsilon);
            }

            prev_tx_a = tx_a; prev_rx_b = rx_b;
            prev_tx_b = tx_b; prev_rx_a = rx_a;
        }

        fprintf(fp, "%d,%d,%.9f,%.9f,%.9f,%.9f,%d,%d\n",
                rank, resync_count, tx_a, rx_b, tx_b, rx_a, keep_ab, keep_ba);
    }

fflush(fp);
fclose(fp);    
    
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


static int load_sc_csv(const char *path){
    for (int i=0;i<MAX_RESYNC;i++) 
        g_have[i]=0;
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    int rows=0, r; double s, c;
    while (fscanf(fp, "%d,%lf,%lf", &r, &s, &c) == 3){
        if (0 <= r && r < MAX_RESYNC){
            g_s[r] = s; g_c[r] = c; g_have[r] = 1; rows++;
        }
    }
    fclose(fp);
    return rows;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;

    int K=10;
    int resync_count = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    const char *sc_path = (argc > 3) ? argv[3] : "svm_sc.csv";
    int sc_rows = 0;
    if (rank == 0) {
        sc_rows = load_sc_csv(sc_path);  // fills g_s[], g_c[], g_have[]
        printf("Loaded %d rows from %s\n", sc_rows, sc_path);
    }
    // share to everyone so all ranks can map B->A the same way

    MPI_Bcast(g_s,    MAX_RESYNC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(g_c,    MAX_RESYNC, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(g_have, MAX_RESYNC, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    
    if (rank == 0) { mkdir("logs_align", 0777); }
    MPI_Barrier(MPI_COMM_WORLD);

    static FILE *align_fp = NULL;
    {
        char fn[128];
        snprintf(fn, sizeof(fn), "logs_align/align_rank%d.csv", rank);
        align_fp = fopen(fn, "a");
        if (align_fp) {
        // Add header only if empty file
            fseek(align_fp, 0, SEEK_END);
            if (ftell(align_fp) == 0) {
                fprintf(align_fp, "resync,trial,lag_pre,waited,lag_post,timed_out\n");
                fflush(align_fp);
            }
        }
    }

    
    
    
    if (rank == 0) {
        mkdir("logs-9-9-rwork", 0777);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    int use_sync = 1;
    if (argc > 1 && atoi(argv[1]) == 1) {
        use_sync = 0;
    }

    int base_array_size = 1000000;
    if (argc > 2) {
        base_array_size = atoi(argv[2]);
    }

    //int array_size = base_array_size * size;
    int array_size = base_array_size;
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
     //   load_skew_table(); 
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
        //double local_now = MPI_Wtime();
        double local_now = use_sync ? aligned_now(resync_count) : MPI_Wtime(); // when use_sync is on, the min/max drift you print is measured on the aligned clock

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

        
    if (use_sync) {
    // 1) each rank’s current time on the shared timeline
    double me = aligned_now(resync_count);

    // 2) measure instantaneous skew on the aligned clock
    double t_max, t_min;
    MPI_Allreduce(&me, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&me, &t_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    double skew = t_max - t_min;             // seconds

    // 3) choose headroom = skew + guard, clamped
    const double GUARD     = 50e-6;          // 50 µs
    const double HEAD_MIN  = 50e-6;          // 50 µs
    const double HEAD_MAX  = 500e-6;         // 500 µs
    double head = skew + GUARD;
    if (head < HEAD_MIN) head = HEAD_MIN;
    if (head > HEAD_MAX) head = HEAD_MAX;

    double t_start = t_max + head;

    // 4) wait: short sleeps + short spin, with a small timeout
    struct timespec nap = {0, 20000};        // 20 µs nominal
    const double TIMEOUT = 2e-3;             // 2 ms
    double t0 = aligned_now(resync_count);
    int timed_out = 0;

    while (1) {
        double now = aligned_now(resync_count);
        if (now >= t_start) break;
        if ((t_start - now) > 1e-4) {        // >100 µs left → nap
            nanosleep(&nap, NULL);
        } else {
            // final short spin (avoid scheduler latency)
            while ((now = aligned_now(resync_count)) < t_start) { /* spin a bit */ }
            break;
        }
        if (aligned_now(resync_count) - t0 > TIMEOUT) { timed_out = 1; break; }
    }

    // 5) light logging (rank 0, first few trials)
    static int log_trials = 3;
    if (rank == 0 && log_trials > 0) {
        double waited = aligned_now(resync_count) - t0;
        printf("align: skew=%.0f us, head=%.0f us, waited=%.0f us%s\n",
               1e6*skew, 1e6*head, 1e6*waited, timed_out ? " (timeout)" : "");
        --log_trials;
    }
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

