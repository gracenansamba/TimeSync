// huygen_inbuilt_SVM.c

// Compare MPI_Allreduce latency with Barrier vs SVM-deadline start alignment.

// Requires: MPI + LIBLINEAR built from https://github.com/cjlin1/liblinear

//

// Build (objects you already have):

//   mpicxx -O2 -std=c++11 huygen_inbuilt_SVM.c \

//     -I/work/projects/askjellum/gnansamba42/liblinear \

//     /work/projects/askjellum/gnansamba42/liblinear/linear.o \

//     /work/projects/askjellum/gnansamba42/liblinear/newton.o \

//     /work/projects/askjellum/gnansamba42/liblinear/blas/blas.a \

//     -lm -o huygen_inbuilt_SVMg

//

// If you make a static archive once:

//   ar rcs /work/projects/askjellum/gnansamba42/liblinear/liblinear.a \

//     /work/projects/askjellum/gnansamba42/liblinear/linear.o \

//     /work/projects/askjellum/gnansamba42/liblinear/newton.o \

//     /work/projects/askjellum/gnansamba42/liblinear/blas/blas.a

//   mpicxx -O2 -std=c++11 huygen_inbuilt_SVM.c \

//     -I/work/projects/askjellum/gnansamba42/liblinear \

//     /work/projects/askjellum/gnansamba42/liblinear/liblinear.a -lm \

//     -o huygen_inbuilt_SVMg

//

// Run examples:

//   mpirun -n 2 ./huygen_inbuilt_SVMg --mode=barrier --trials=10 --size=8

//   mpirun -n 2 ./huygen_inbuilt_SVMg --mode=svm     --trials=10 --size=8



#include <mpi.h>

#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <math.h>

#include <time.h>

#include <stdint.h>

#include "linear.h"  // from liblinear



// ------------------- Tunables -------------------

#ifndef NUM_PROBES

#define NUM_PROBES 64          // coded probes per resync window

#endif



#ifndef RESYNC_EVERY

#define RESYNC_EVERY 25        // resync every N trials (and at trial 0)

#endif



#ifndef EPSILON_PROBE

#define EPSILON_PROBE 2e-5     // 20 microseconds tolerance to keep a probe pair

#endif



#ifndef GUARD_SEC

#define GUARD_SEC 1e-4         // 100 microseconds guard for deadline mode

#endif

// ------------------------------------------------



typedef enum { MODE_BARRIER=0, MODE_SVM=1 } start_mode_t;



typedef struct { double x, y; int label; } Sample;



// Silence liblinear's optimizer prints (optional)

static void quiet_print(const char* s) { (void)s; }



static inline double get_local_time_sec(void) {

    struct timespec ts;

#if defined(CLOCK_MONOTONIC_RAW)

    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

#else

    clock_gettime(CLOCK_MONOTONIC, &ts);

#endif

    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;

}



static inline void abs_sleep_until(double t_local_abs) {

    if (t_local_abs <= 0) return;

    struct timespec ts;

    ts.tv_sec  = (time_t)t_local_abs;

    ts.tv_nsec = (long)((t_local_abs - ts.tv_sec) * 1e9);

#if defined(CLOCK_MONOTONIC_RAW)

    clock_nanosleep(CLOCK_MONOTONIC_RAW, TIMER_ABSTIME, &ts, NULL);

#else

    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL);

#endif

}



// Train a linear SVM and return separator y = alpha*x + beta.

// We center x by its window mean and scale y from seconds to microseconds

// to improve numerical stability, then convert back.

static void fit_alpha_beta_liblinear(const Sample *S, int N, double *alpha, double *beta) {

    // Center x; scale y

    double x0 = 0.0;

    for (int i = 0; i < N; ++i) x0 += S[i].x;

    x0 = (N > 0 ? x0 / N : 0.0);

    const double sx = 1.0;   // keep x scale

    const double sy = 1e6;   // seconds -> microseconds



    struct problem  prob = (struct problem){0};

    struct parameter param = (struct parameter){0};



    prob.l   = N;

    prob.n   = 2;        // features: x', y'

    prob.bias = 1.0;     // liblinear-managed bias

    prob.y   = (double*)malloc(N * sizeof(double));

    prob.x   = (struct feature_node**)malloc(N * sizeof(struct feature_node*));

    if (!prob.y || !prob.x) { fprintf(stderr, "OOM\n"); MPI_Abort(MPI_COMM_WORLD, 1); }



    for (int i = 0; i < N; ++i) {

        prob.y[i] = (double)S[i].label;   // +1 / -1

        double xp = (S[i].x - x0) * sx;

        double yp =  S[i].y * sy;

        struct feature_node *row = (struct feature_node*)malloc(3 * sizeof(*row));

        row[0].index = 1; row[0].value = xp;

        row[1].index = 2; row[1].value = yp;

        row[2].index = -1;

        prob.x[i] = row;

    }



    // Linear SVM (primal). Bias is regularized by default when bias>0.

    param.solver_type = L2R_L2LOSS_SVC;

    param.C  = 1.0;

    param.eps = 0.1;



    const char *err = check_parameter(&prob, &param);

    if (err) { fprintf(stderr, "liblinear param error: %s\n", err); MPI_Abort(MPI_COMM_WORLD, 2); }



    // Quiet internal prints

    set_print_string_function(quiet_print);



    struct model *mdl = train(&prob, &param);



    // Hyperplane in scaled space: wX * x' + wY * y' + b' = 0

    const double *w = mdl->w;

    int nfeat = get_nr_feature(mdl); // 2

    double wX = w[0];

    double wY = w[1];

    double bprime  = (mdl->bias > 0) ? w[nfeat] * mdl->bias : 0.0;



    // y' = a' x' + b'

    double denom = (fabs(wY) > 1e-20 ? wY : (wY < 0 ? -1e-20 : 1e-20));

    double aprime = -wX / denom;

    double bprime_line  = -bprime / denom;



    // Back to original units

    double a = (aprime * sx) / sy;

    double b = (bprime_line) / sy - a * x0;



    *alpha = a;

    *beta  = b;



    free_and_destroy_model(&mdl);

    for (int i = 0; i < N; ++i) free(prob.x[i]);

    free(prob.x); free(prob.y);

}



// Ring-coded probe exchange; build SVM samples in-memory

static int do_resync_and_collect_samples(int rank, int size, Sample *out, int out_cap) {

    const int target = (rank + 1) % size;

    const int source = (rank - 1 + size) % size;



    double prev_tx_a = 0.0, prev_rx_b = 0.0, prev_tx_b = 0.0, prev_rx_a = 0.0;

    int have_prev = 0;



    double tx_a=0, rx_b=0, tx_b=0, rx_a=0;

    int keep_ab=1, keep_ba=1;



    int m = 0;



    enum { TAG_TXA=100, TAG_RXB=101, TAG_TXB=102, TAG_RXA=103 };



    for (int i = 0; i < NUM_PROBES; i++) {

        keep_ab = 1; keep_ba = 1; // first pair kept by default



        if ((rank % 2) == 0) {

            // Even ranks initiate A->B, then complete B->A

            tx_a = get_local_time_sec();

            MPI_Send(&tx_a, 1, MPI_DOUBLE, target, TAG_TXA, MPI_COMM_WORLD);

            MPI_Recv(&rx_b, 1, MPI_DOUBLE, target, TAG_RXB, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            rx_a = get_local_time_sec();

            MPI_Recv(&tx_b, 1, MPI_DOUBLE, target, TAG_TXB, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Send(&rx_a, 1, MPI_DOUBLE, target, TAG_RXA, MPI_COMM_WORLD);

        } else {

            // Odd ranks respond

            MPI_Recv(&tx_a, 1, MPI_DOUBLE, source, TAG_TXA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            rx_b = get_local_time_sec();

            tx_b = get_local_time_sec();

            MPI_Send(&rx_b, 1, MPI_DOUBLE, source, TAG_RXB, MPI_COMM_WORLD);

            MPI_Send(&tx_b, 1, MPI_DOUBLE, source, TAG_TXB, MPI_COMM_WORLD);

            MPI_Recv(&rx_a, 1, MPI_DOUBLE, source, TAG_RXA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }



        if (have_prev) {

            double d_txAB = tx_a - prev_tx_a;

            double d_rxAB = rx_b - prev_rx_b;

            double d_txBA = tx_b - prev_tx_b;

            double d_rxBA = rx_a - prev_rx_a;

            keep_ab = (fabs(d_rxAB - d_txAB) <= EPSILON_PROBE);

            keep_ba = (fabs(d_rxBA - d_txBA) <= EPSILON_PROBE);

        }



        if (keep_ab && m < out_cap) out[m++] = (Sample){ .x = tx_a, .y = (rx_b - tx_a), .label = +1 };

        if (keep_ba && m < out_cap) out[m++] = (Sample){ .x = rx_a, .y = (tx_b - rx_a), .label = -1 };



        prev_tx_a = tx_a; prev_rx_b = rx_b;

        prev_tx_b = tx_b; prev_rx_a = rx_a;

        have_prev = 1;

    }



    return m;

}



// Convert (alpha,beta) to inverse aligned_now form: (t - c)/s

static inline void alpha_beta_to_inverse(double alpha, double beta, double *s_inv, double *c_inv) {

    double s_direct = 1.0 + alpha; // t_ref = s_direct * t_local + beta

    if (fabs(s_direct) < 1e-12) s_direct = (s_direct < 0 ? -1e-12 : 1e-12);

    *s_inv = 1.0 / s_direct;

    *c_inv = -beta * (*s_inv);

}



static void parse_args(int argc, char **argv,

                       start_mode_t *mode, int *trials, int *array_size, int *resync_every) {

    *mode = MODE_BARRIER; *trials = 200; *array_size = 8; *resync_every = RESYNC_EVERY;

    for (int i=1;i<argc;i++) {

        if (!strncmp(argv[i],"--mode=",7)) {

            const char *v = argv[i]+7;

            if (!strcmp(v,"barrier")) *mode = MODE_BARRIER;

            else if (!strcmp(v,"svm")) *mode = MODE_SVM;

        } else if (!strncmp(argv[i],"--trials=",9)) {

            *trials = atoi(argv[i]+9);

        } else if (!strncmp(argv[i],"--size=",7)) {

            *array_size = atoi(argv[i]+7);

        } else if (!strncmp(argv[i],"--resync=",9)) {

            *resync_every = atoi(argv[i]+9);

        }

    }

}



int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);



    // Quiet liblinear messages on all ranks

    set_print_string_function(quiet_print);



    int rank=0, size=1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);



    start_mode_t mode;

    int NUM_TRIALS, ARRAY_SIZE, RESYNC_N;

    parse_args(argc, argv, &mode, &NUM_TRIALS, &ARRAY_SIZE, &RESYNC_N);



    if (rank==0) {

        printf("Mode=%s, trials=%d, array_size=%d doubles (%.3f KB), resync_every=%d, probes=%d\n",

               (mode==MODE_BARRIER?"barrier":"svm-deadline"), NUM_TRIALS, ARRAY_SIZE,

               ARRAY_SIZE*sizeof(double)/1024.0, RESYNC_N, NUM_PROBES);

    }



    // Benchmark buffers

    double *local = (double*)malloc(ARRAY_SIZE*sizeof(double));

    double *global= (double*)malloc(ARRAY_SIZE*sizeof(double));

    for (int i=0;i<ARRAY_SIZE;i++) local[i] = (double)(rank + 1);

    memset(global, 0, ARRAY_SIZE*sizeof(double));



    // Model storage per window

    double s_inv_curr = 1.0, c_inv_curr = 0.0; // (t - c)/s

    int    model_ok_curr = 0;

    int    resync_count = -1;



    double total_max = 0.0;



    for (int t = 0; t < NUM_TRIALS; ++t) {

        // RESYNC window at t==0 and every RESYNC_N trials (ONLY in SVM mode)

        if (mode == MODE_SVM && (t == 0 || (RESYNC_N > 0 && (t % RESYNC_N) == 0))) {

            resync_count++;

            Sample *S = (Sample*)malloc(2*NUM_PROBES*sizeof(Sample));

            int M = do_resync_and_collect_samples(rank, size, S, 2*NUM_PROBES);



            double alpha_local=0.0, beta_local=0.0;

            if (M >= 4) fit_alpha_beta_liblinear(S, M, &alpha_local, &beta_local);

            free(S);



            double alpha_sum=0.0, beta_sum=0.0;

            MPI_Allreduce(&alpha_local, &alpha_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            MPI_Allreduce(&beta_local,  &beta_sum,  1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            double alpha_bar = alpha_sum / size;

            double beta_bar  = beta_sum  / size;



            alpha_beta_to_inverse(alpha_bar, beta_bar, &s_inv_curr, &c_inv_curr);



            model_ok_curr = isfinite(s_inv_curr) && isfinite(c_inv_curr)

                         && (s_inv_curr > 0.5) && (s_inv_curr < 1.5);



            if (rank==0) {

                printf("[resync %d] M≈%d, alpha_bar=%.3e, beta_bar=%.3e, s_inv=%.6f, c_inv=%.6e, ok=%d\n",

                       resync_count, M, alpha_bar, beta_bar, s_inv_curr, c_inv_curr, model_ok_curr);

            }

        } else if (mode != MODE_SVM && (t == 0 || (RESYNC_N > 0 && (t % RESYNC_N) == 0))) {

            // In barrier mode, ensure we don't accidentally try SVM deadline

            model_ok_curr = 0;

        }



        // ----------------- Pre-step before Allreduce -----------------

        if (mode == MODE_BARRIER || !model_ok_curr) {

            MPI_Barrier(MPI_COMM_WORLD); // baseline or fallback

        } else {

            // SVM-deadline: broadcast a reference start time and absolute-sleep locally

            double t_local_now = get_local_time_sec();

            double now_ref = (t_local_now - c_inv_curr) / s_inv_curr;  // aligned_now



            double t_start_ref = 0.0;

            if (rank == 0) t_start_ref = now_ref + GUARD_SEC;

            MPI_Bcast(&t_start_ref, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);



            double t_start_local = s_inv_curr * t_start_ref + c_inv_curr; // back to local

            // small safety clamp

            double now2 = get_local_time_sec();

            if (t_start_local < now2) t_start_local = now2 + 50e-6;

            if (t_start_local - now2 > 0.1) t_start_local = now2 + 0.1;

            abs_sleep_until(t_start_local);

        }



        // ----------------- Timed collective -----------------

        double t0 = MPI_Wtime();

        MPI_Allreduce(local, global, ARRAY_SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();

        double trial = t1 - t0;



        // Max across ranks = true collective latency

        double trial_max = 0.0;

        MPI_Reduce(&trial, &trial_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank==0) total_max += trial_max;

    }



    if (rank==0) {

        double avg_max = total_max / NUM_TRIALS;

        printf("[RESULT] mode=%s, ranks=%d, size=%d doubles (%.3f KB), trials=%d\n"

               "         Avg Allreduce latency (max across ranks): %.9f s\n",

               (mode==MODE_BARRIER?"barrier":"svm-deadline"),

               size, ARRAY_SIZE, ARRAY_SIZE*sizeof(double)/1024.0, NUM_TRIALS, avg_max);

    }



    free(local); free(global);

    MPI_Finalize();

    return 0;

}


