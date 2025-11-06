#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <limits.h>

#define REPETICIONES 5
#define UMBRAL_TASK 10000  // tamaño mínimo para crear tareas (ajustable)

/* Comparador (no usado por quicksort_parallel pero útil) */
int comparar_double(const void *a, const void *b) {
    double x = *(double *)a;
    double y = *(double *)b;
    return (x > y) - (x < y);
}

/* QuickSort paralelo con OpenMP (divide & conquer con tareas) */
void quicksort_parallel(double *arr, long left, long right, int profundidad) {
    if (left >= right) return;
    long i = left, j = right;
    double pivote = arr[(left + right) / 2];
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

/* Merge de dos arreglos ordenados */
double *merge_two(const double *a, long na, const double *b, long nb) {
    double *out = malloc((na + nb) * sizeof(double));
    if (!out) return NULL;
    long i = 0, j = 0, k = 0;
    while (i < na && j < nb) {
        if (a[i] <= b[j]) out[k++] = a[i++];
        else out[k++] = b[j++];
    }
    while (i < na) out[k++] = a[i++];
    while (j < nb) out[k++] = b[j++];
    return out;
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

    int omp_threads = omp_get_max_threads();
    int max_depth = (int)log2((omp_threads > 0) ? omp_threads : 1);
    if (max_depth < 1) max_depth = 1;

    double tiempo_total_prom = 0.0;

    for (int rep = 0; rep < REPETICIONES; ++rep) {
        double t0 = MPI_Wtime();
        double *datos = NULL;
        long total = 0;
        double t_io_start = 0.0, t_io_end = 0.0;

        if (rank == 0) {
            t_io_start = MPI_Wtime();
            FILE *f = fopen("dataset.csv", "r");
            if (!f) {
                fprintf(stderr, "Error: no se pudo abrir dataset.csv\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            double tmp;
            while (fscanf(f, "%lf", &tmp) == 1) total++;
            rewind(f);

            datos = malloc(total * sizeof(double));
            if (!datos) {
                fprintf(stderr, "Error: malloc datos falló (total=%ld)\n", total);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            for (long i = 0; i < total; ++i) {
                if (fscanf(f, "%lf", &datos[i]) != 1) {
                    fprintf(stderr, "Error leyendo dato %ld\n", i);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
            fclose(f);
            t_io_end = MPI_Wtime();

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

        /* Construir counts/displs (long) y versiones int para MPI_Scatterv */
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

        /* Scatterv */
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

        /* Merge en árbol: cada proceso mantiene cur_data y owns_data */
        double *cur_data = local;
        long cur_n = local_n;
        int owns_data = 1; // inicialmente cada proceso "posee" local

        double t_merge_start = MPI_Wtime();
        for (int step = 1; step < size; step <<= 1) {
            if (rank % (2 * step) == 0) {
                int partner = rank + step;
                if (partner < size) {
                    /* recibir count (MPI_LONG) */
                    long recv_count = 0;
                    MPI_Recv(&recv_count, 1, MPI_LONG, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (recv_count < 0) { fprintf(stderr, "recv_count negativo\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

                    /* reservar buffer y recibir datos */
                    double *recv_buf = malloc((recv_count > 0 ? recv_count : 1) * sizeof(double));
                    if (!recv_buf && recv_count > 0) MPI_Abort(MPI_COMM_WORLD, 1);
                    MPI_Recv(recv_buf, (int)recv_count, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    /* merge cur_data + recv_buf -> merged */
                    double *merged = merge_two(cur_data, cur_n, recv_buf, recv_count);
                    if (!merged) { fprintf(stderr, "merge_two devolvió NULL\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

                    /* liberar antiguos buffers con control */
                    if (owns_data && cur_data) {
                        free(cur_data);
                        cur_data = NULL;
                        owns_data = 0;
                    }
                    if (recv_buf) free(recv_buf);

                    /* adoptar merged */
                    cur_data = merged;
                    cur_n = cur_n + recv_count;
                    owns_data = 1; // ahora poseemos merged
                }
            } else {
                int partner = rank - step;
                /* enviar cur_n y cur_data a partner, luego abandonar */
                MPI_Send(&cur_n, 1, MPI_LONG, partner, 0, MPI_COMM_WORLD);
                MPI_Send(cur_data, (int)cur_n, MPI_DOUBLE, partner, 1, MPI_COMM_WORLD);
                /* después de enviar, liberamos si poseemos buffer */
                if (owns_data && cur_data) {
                    free(cur_data);
                    cur_data = NULL;
                    owns_data = 0;
                }
                cur_n = 0;
                break;
            }
        }
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
        }
        if (owns_data && cur_data) {
            free(cur_data);
            cur_data = NULL;
            owns_data = 0;
        }

        free(counts); free(displs); free(scounts); free(sdispls);
        // local fue liberado por la lógica de owns_data; no liberar aquí.
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
