#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <errno.h>

#define REPETICIONES 5
#define UMBRAL_TASK 50000  // aumentado para reducir overhead de tareas
#define INSERTION_SORT_THRESHOLD 64  // para subarreglos pequeños usar insertion sort

/* Comparador (para qsort si se necesitara) */
int comparar_double(const void *a, const void *b) {
    double x = *(double *)a;
    double y = *(double *)b;
    return (x > y) - (x < y);
}

/* Insertion sort para subarreglos pequeños */
void insertion_sort(double *arr, long left, long right) {
    for (long i = left + 1; i <= right; ++i) {
        double key = arr[i];
        long j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

/* QuickSort paralelo con OpenMP (divide & conquer con tareas).
   Cambios: si el subarreglo es pequeño, usar insertion_sort. */
void quicksort_parallel(double *arr, long left, long right, int profundidad) {
    if (left >= right) return;
    long len = right - left + 1;
    if (len <= INSERTION_SORT_THRESHOLD) {
        insertion_sort(arr, left, right);
        return;
    }

    long i = left, j = right;
    double pivote = arr[left + (right - left) / 2];
    while (i <= j) {
        while (arr[i] < pivote) i++;
        while (arr[j] > pivote) j--;
        if (i <= j) {
            double tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++; j--;
        }
    }

    if (profundidad > 0 && (right - left) > UMBRAL_TASK) {
        #pragma omp task shared(arr) firstprivate(left, j, profundidad)
        quicksort_parallel(arr, left, j, profundidad - 1);

        #pragma omp task shared(arr) firstprivate(i, right, profundidad)
        quicksort_parallel(arr, i, right, profundidad - 1);

        #pragma omp taskwait
    } else {
        if (left < j) quicksort_parallel(arr, left, j, 0);
        if (i < right) quicksort_parallel(arr, i, right, 0);
    }
}

/* Merge de dos arreglos ordenados (devuelve nuevo buffer que debe liberarse) */
double *merge_two(const double *a, long na, const double *b, long nb) {
    double *out = malloc((na + nb) * sizeof(double));
    if (!out) return NULL;
    long ia = 0, ib = 0, k = 0;
    while (ia < na && ib < nb) {
        if (a[ia] <= b[ib]) out[k++] = a[ia++];
        else out[k++] = b[ib++];
    }
    while (ia < na) out[k++] = a[ia++];
    while (ib < nb) out[k++] = b[ib++];
    return out;
}

/* Lectura rápida de CSV usando getline + strtod */
double *read_csv_fast(const char *filename, long *out_total) {
    FILE *f = fopen(filename, "r");
    if (!f) return NULL;

    char *line = NULL;
    size_t len = 0;
    long capacity = 1024;
    long total = 0;
    double *arr = malloc(capacity * sizeof(double));
    if (!arr) { fclose(f); return NULL; }

    while (getline(&line, &len, f) != -1) {
        char *endptr = NULL;
        errno = 0;
        double val = strtod(line, &endptr);
        if (endptr == line) continue; // línea vacía o no numérica
        if (errno != 0 && val == 0.0) continue;
        if (total >= capacity) {
            long newcap = capacity * 2;
            double *tmp = realloc(arr, newcap * sizeof(double));
            if (!tmp) { free(arr); free(line); fclose(f); return NULL; }
            arr = tmp;
            capacity = newcap;
        }
        arr[total++] = val;
    }

    free(line);
    fclose(f);
    *out_total = total;
    return arr;
}

int main(int argc, char *argv[]) {
    int rank, size;
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init failed\n");
        return 1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Se requieren al menos 2 procesos (1 maestro + N esclavos)\n");
        MPI_Finalize();
        return 1;
    }

    /* Configurar OpenMP: fijar número de hilos detectado por entorno */
    int omp_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(omp_threads);
    int max_depth = (int)log2((omp_threads > 0) ? omp_threads : 1);
    if (max_depth < 1) max_depth = 1;

    double tiempo_total_prom = 0.0;

    for (int rep = 0; rep < REPETICIONES; ++rep) {
        double t0 = MPI_Wtime();
        double *datos = NULL;   // solo en rank 0 contiene todo
        long total = 0;
        double t_io_start = 0.0, t_io_end = 0.0;

        if (rank == 0) {
            t_io_start = MPI_Wtime();
            datos = read_csv_fast("dataset.csv", &total);
            t_io_end = MPI_Wtime();
            if (!datos) {
                fprintf(stderr, "Error: no se pudo leer dataset.csv (rank 0)\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            printf("[Rep %d] Dataset cargado: %ld elementos\n", rep + 1, total);
            printf("Memoria usada (maestro, solo datos): %.2f MB\n",
                   (total * sizeof(double)) / (1024.0 * 1024.0));
            printf("I/O (lectura) tomó: %.6f s\n", t_io_end - t_io_start);
        }

        /* Broadcast total a todos */
        MPI_Bcast(&total, 1, MPI_LONG, 0, MPI_COMM_WORLD);
        if (total == 0) {
            if (rank == 0) fprintf(stderr, "Dataset vacío\n");
            MPI_Finalize();
            return 1;
        }

        /* Construir counts/displs (long) y versiones int para MPI_Scatterv/Gatherv */
        int k = size;
        long *counts = malloc(k * sizeof(long));
        long *displs = malloc(k * sizeof(long));
        if (!counts || !displs) MPI_Abort(MPI_COMM_WORLD, 1);

        long base = total / k;
        int rem = (int)(total % k);
        for (int i = 0; i < k; ++i) counts[i] = base + (i < rem ? 1 : 0);
        displs[0] = 0;
        for (int i = 1; i < k; ++i) displs[i] = displs[i - 1] + counts[i - 1];

        int *scounts = malloc(k * sizeof(int));
        int *sdispls = malloc(k * sizeof(int));
        if (!scounts || !sdispls) MPI_Abort(MPI_COMM_WORLD, 1);
        for (int i = 0; i < k; ++i) {
            if (counts[i] > INT_MAX) {
                if (rank == 0) fprintf(stderr, "Error: chunk demasiado grande para int\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            scounts[i] = (int)counts[i];
            sdispls[i] = (int)displs[i];
        }

        /* Reserva local */
        long local_n = counts[rank];
        double *local = malloc((local_n > 0 ? local_n : 1) * sizeof(double));
        if (!local) MPI_Abort(MPI_COMM_WORLD, 1);

        /* Scatterv desde root a todos */
        double t_scatter_start = MPI_Wtime();
        MPI_Scatterv(datos, scounts, sdispls, MPI_DOUBLE, local, (int)local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        double t_scatter_end = MPI_Wtime();

        /* Ordenamiento local con OpenMP */
        double t_sort_start = MPI_Wtime();
        #pragma omp parallel
        {
            #pragma omp single
            quicksort_parallel(local, 0, (local_n > 0 ? local_n - 1 : 0), max_depth);
        }
        double t_sort_end = MPI_Wtime();

        /* --- NUEVO: recolección mediante Gatherv al root y merge seguro allí --- */
        double t_merge_start = MPI_Wtime();

        /* En root reservamos buffer para recibir todos los datos ordenados */
        double *gather_buf = NULL;
        if (rank == 0) {
            gather_buf = malloc((total > 0 ? total : 1) * sizeof(double));
            if (!gather_buf && total > 0) MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Todos envían su bloque local ordenado al root */
        MPI_Gatherv(local, (int)local_n, MPI_DOUBLE,
                    gather_buf, scounts, sdispls, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        /* Root hace merge iterativo de las porciones ya ordenadas */
        double *cur_data = NULL;
        long cur_n = 0;
        int owns_data = 0;
        if (rank == 0) {
            /* Inicializar cur_data con la porción del rank 0 que ya está en gather_buf */
            if (scounts[0] > 0) {
                cur_data = malloc(scounts[0] * sizeof(double));
                if (!cur_data && scounts[0] > 0) MPI_Abort(MPI_COMM_WORLD, 1);
                memcpy(cur_data, gather_buf + sdispls[0], scounts[0] * sizeof(double));
                cur_n = scounts[0];
                owns_data = 1;
            } else {
                /* Si por alguna razón scounts[0]==0, empezar con vacío */
                cur_data = NULL;
                cur_n = 0;
                owns_data = 0;
            }

            /* Merge iterativo con las porciones 1..k-1 */
            for (int src = 1; src < k; ++src) {
                long src_n = counts[src];
                if (src_n == 0) continue;
                double *src_ptr = gather_buf + sdispls[src];

                /* merge cur_data (cur_n) con src_ptr (src_n) -> merged */
                double *merged = NULL;
                if (cur_n == 0) {
                    /* simplemente copiar src_ptr */
                    merged = malloc(src_n * sizeof(double));
                    if (!merged && src_n > 0) MPI_Abort(MPI_COMM_WORLD, 1);
                    memcpy(merged, src_ptr, src_n * sizeof(double));
                    /* liberar cur_data si existía */
                    if (owns_data && cur_data) { free(cur_data); cur_data = NULL; owns_data = 0; }
                    cur_data = merged;
                    cur_n = src_n;
                    owns_data = 1;
                } else {
                    merged = merge_two(cur_data, cur_n, src_ptr, src_n);
                    if (!merged) { fprintf(stderr, "merge_two devolvió NULL en root\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
                    /* liberar antiguo cur_data */
                    if (owns_data && cur_data) { free(cur_data); cur_data = NULL; owns_data = 0; }
                    cur_data = merged;
                    cur_n = cur_n + src_n;
                    owns_data = 1;
                }
            }
        }

        /* Root ya tiene cur_data con todos los datos ordenados (si quieres puedes usarlo aquí) */

        double t_merge_end = MPI_Wtime();

        /* Recolección de tiempos (máximos por etapa) */
        double scatter_time = t_scatter_end - t_scatter_start;
        double sort_time = t_sort_end - t_sort_start;
        double merge_time = t_merge_end - t_merge_start;
        double io_time = (rank == 0) ? (t_io_end - t_io_start) : 0.0;

        double max_scatter, max_sort, max_merge, max_io;
        MPI_Reduce(&scatter_time, &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sort_time, &max_sort, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&merge_time, &max_merge, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&io_time, &max_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        /* Memoria local y suma total */
        double local_mem = (local_n * sizeof(double)) / (1024.0 * 1024.0);
        double sum_mem = 0.0;
        MPI_Reduce(&local_mem, &sum_mem, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double dur_total = MPI_Wtime() - t0;
            tiempo_total_prom += dur_total;
            printf("Rep %d: tiempos (s): I/O=%.6f scatter=%.6f sort_max=%.6f merge_max=%.6f total=%.6f\n",
                   rep + 1, max_io, max_scatter, max_sort, max_merge, dur_total);
            printf("Memoria total (suma procesos): %.2f MB\n", sum_mem);
        }

        /* Liberaciones finales: solo lo que se posea */
        if (rank == 0) {
            if (datos) { free(datos); datos = NULL; }
            if (gather_buf) { free(gather_buf); gather_buf = NULL; }
            if (owns_data && cur_data) { free(cur_data); cur_data = NULL; owns_data = 0; }
        } else {
            /* otros ranks no deben liberar cur_data (no lo tienen) */
        }

        if (local) { free(local); local = NULL; }

        free(counts); counts = NULL;
        free(displs); displs = NULL;
        free(scounts); scounts = NULL;
        free(sdispls); sdispls = NULL;
    } // fin repeticiones

    if (rank == 0) {
        double promedio = tiempo_total_prom / REPETICIONES;
        printf("\n=== Resultado final ===\n");
        printf("Procesos (MPI): %d\n", size);
        printf("Hilos por proceso (OpenMP): %d\n", omp_get_max_threads());
        printf("Tiempo promedio (sobre %d reps): %.6f s\n", REPETICIONES, promedio);
        printf("======================\n");

        FILE *f = fopen("resultados_mpi_openmp.csv", "a");
        if (f) {
            fprintf(f, "%d,%d,%.6f\n", size, omp_get_max_threads(), promedio);
            fclose(f);
        }
    }

    MPI_Finalize();
    return 0;
}
