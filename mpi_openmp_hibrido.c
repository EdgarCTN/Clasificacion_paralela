/* mpi_openmp_hibrido_mejorado_parallel_io.c
   Lectura paralela de CSV con MPI I/O + OpenMP hybrid sorting
   Mantiene: insertion sort para subarreglos pequeños, UMBRAL_TASK elevado, etc.
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <errno.h>

#define REPETICIONES 5
#define UMBRAL_TASK 50000
#define INSERTION_SORT_THRESHOLD 64
/* activa prints de depuración por proceso */
// #define DEBUG 1

/* comparador para qsort si se necesita */
int comparar_double(const void *a, const void *b) {
    double x = *(double *)a;
    double y = *(double *)b;
    return (x > y) - (x < y);
}

/* insertion sort para subarreglos pequeños */
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

/* quicksort paralelo con OpenMP y threshold/insertion */
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

/* merge de dos arrays ordenados -> nuevo buffer */
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

/* Lectura paralela de CSV con MPI I/O
   Cada proceso recibe su rango de bytes, ajusta para no partir líneas,
   parsea números (strtod) y devuelve un arreglo con sus valores.
   out_count contiene la cantidad de elementos leídos por el proceso.
*/
double *read_csv_parallel(const char *filename, long *out_count, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    MPI_File fh;
    MPI_Offset file_size = 0;
    if (MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        if (rank == 0) fprintf(stderr, "Error: no se pudo abrir %s con MPI_File_open\n", filename);
        MPI_Barrier(comm);
        return NULL;
    }
    MPI_File_get_size(fh, &file_size);
    if (file_size == 0) {
        MPI_File_close(&fh);
        *out_count = 0;
        return NULL;
    }

    /* calcular rango de bytes para cada proceso */
    MPI_Offset chunk = file_size / nprocs;
    MPI_Offset start = rank * chunk;
    MPI_Offset end = (rank == nprocs - 1) ? (file_size - 1) : (start + chunk - 1);
    MPI_Offset read_len = end - start + 1;

    /* leer un poco extra (1 KiB) al final para coger la línea completa */
    const MPI_Offset EXTRA = 1024;
    MPI_Offset alloc_len = (size_t)(read_len + EXTRA + 1); // +1 para terminador
    char *buf = malloc((size_t)alloc_len);
    if (!buf) {
        MPI_File_close(&fh);
        fprintf(stderr, "rank %d: malloc falló en read_csv_parallel\n", rank);
        return NULL;
    }
    memset(buf, 0, (size_t)alloc_len);

    /* leer en paralelo desde offset start */
    MPI_Status st;
    MPI_Offset read_offset = start;
    MPI_File_read_at_all(fh, read_offset, buf, (int)(read_len + EXTRA), MPI_CHAR, &st);
    /* Nota: si (read_len + EXTRA) excede file bounds, MPI devuelve menos, pero el buffer está term'd por memset */

    /* determinar inicio real (si no es rank 0, saltar hasta después del primer '\n') */
    size_t local_begin = 0;
    if (start != 0) {
        /* buscar primer '\n' en buf */
        char *p = memchr(buf, '\n', (size_t)read_len + (size_t)EXTRA);
        if (p) {
            local_begin = (size_t)(p - buf) + 1; /* comienza después del '\n' */
        } else {
            /* no encontramos nueva línea en el rango leído: archivo con líneas muy largas -> fallback: procesar todo */
            local_begin = 0;
        }
    } else {
        local_begin = 0;
    }

    /* determinar fin real (si no es último rank, recortar hasta último '\n' dentro del buffer) */
    size_t local_end = (size_t)(read_len + EXTRA); /* exclusivo */
    if (rank != nprocs - 1) {
        /* buscar última '\n' en el área [0, read_len+EXTRA) */
        char *last_nl = NULL;
        char *p = buf;
        char *endptr = buf + (size_t)(read_len + EXTRA);
        while (p < endptr) {
            char *q = memchr(p, '\n', (size_t)(endptr - p));
            if (!q) break;
            last_nl = q;
            p = q + 1;
        }
        if (last_nl) {
            local_end = (size_t)(last_nl - buf) + 1; /* exclusivo, después del '\n' */
        } else {
            /* no encontramos '\n': líneas muy largas o chunk demasiado pequeño; procesar todo */
            local_end = (size_t)(read_len + EXTRA);
        }
    } else {
        /* último proceso procesa hasta el final del lo leído */
        local_end = (size_t)(read_len + EXTRA);
    }

    if (local_end <= local_begin) {
        /* nada que procesar */
        free(buf);
        MPI_File_close(&fh);
        *out_count = 0;
        return NULL;
    }

    /* ahora parsear líneas entre buf+local_begin y buf+local_end */
    char *parse_ptr = buf + local_begin;
    char *parse_end = buf + local_end;
    size_t capacity = 1024;
    size_t count = 0;
    double *arr = malloc(capacity * sizeof(double));
    if (!arr) {
        free(buf);
        MPI_File_close(&fh);
        fprintf(stderr, "rank %d: malloc arr falló\n", rank);
        return NULL;
    }

    while (parse_ptr < parse_end) {
        /* buscar próxima '\n' */
        char *nl = memchr(parse_ptr, '\n', (size_t)(parse_end - parse_ptr));
        size_t linelen = nl ? (size_t)(nl - parse_ptr) : (size_t)(parse_end - parse_ptr);
        if (linelen == 0) {
            /* línea vacía -> avanzar */
            if (!nl) break;
            parse_ptr = nl + 1;
            continue;
        }

        /* copiamos la línea a un buffer temporal no-terminado y terminamos para strtod */
        char *linebuf = malloc(linelen + 1);
        if (!linebuf) {
            free(arr); free(buf); MPI_File_close(&fh);
            fprintf(stderr, "rank %d: malloc linebuf falló\n", rank);
            return NULL;
        }
        memcpy(linebuf, parse_ptr, linelen);
        linebuf[linelen] = '\0';

        char *endptr = NULL;
        errno = 0;
        double v = strtod(linebuf, &endptr);
        if (endptr != linebuf && !(errno != 0 && v == 0.0)) {
            if (count >= capacity) {
                size_t nc = capacity * 2;
                double *tmp = realloc(arr, nc * sizeof(double));
                if (!tmp) {
                    free(linebuf); free(arr); free(buf); MPI_File_close(&fh);
                    fprintf(stderr, "rank %d: realloc arr falló\n", rank);
                    return NULL;
                }
                arr = tmp;
                capacity = nc;
            }
            arr[count++] = v;
        }
        free(linebuf);

        if (!nl) break;
        parse_ptr = nl + 1;
    }

    free(buf);
    MPI_File_close(&fh);

    /* compactar arr al tamaño final */
    if (count == 0) {
        free(arr);
        arr = NULL;
    } else {
        double *tmp = realloc(arr, count * sizeof(double));
        if (tmp) arr = tmp;
    }
    *out_count = (long)count;

#ifdef DEBUG
    fprintf(stderr, "rank %d: read bytes [%lld..%lld], local_begin=%zu local_end=%zu, elements=%ld\n",
            rank, (long long)start, (long long)end, local_begin, local_end, (long)*out_count);
#endif

    return arr;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Se requieren al menos 2 procesos\n");
        MPI_Finalize();
        return 1;
    }

    /* Configurar OpenMP */
    int omp_threads = omp_get_max_threads();
    omp_set_dynamic(0);
    omp_set_num_threads(omp_threads);
    int max_depth = (int)log2((omp_threads > 0) ? omp_threads : 1);
    if (max_depth < 1) max_depth = 1;

    double tiempo_total_prom = 0.0;

    for (int rep = 0; rep < REPETICIONES; ++rep) {
        double t0 = MPI_Wtime();
        long local_count = 0;
        double *local_data = NULL;
        double t_io_start = MPI_Wtime();

        /* lectura paralela CSV: cada proceso obtiene su array local */
        local_data = read_csv_parallel("dataset.csv", &local_count, MPI_COMM_WORLD);
        double t_io_end = MPI_Wtime();

        /* contar y sumar los totales globales de elementos */
        long local_count_long = local_count;
        long total = 0;
        MPI_Allreduce(&local_count_long, &total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("[Rep %d] total elementos detectados: %ld\n", rep + 1, total);
            printf("I/O (lectura paralela) tomó: %.6f s\n", t_io_end - t_io_start);
        }

        if (total == 0) {
            if (rank == 0) fprintf(stderr, "Dataset vacío\n");
            MPI_Finalize();
            return 1;
        }

        /* Construir counts/displs por elementos (usaremos total/size reparto equitativo)
           Nota: aquí podemos usar counts/displs basados en local_count reales para Gatherv/Scatters posteriores.
           Vamos a distribuir de forma justa por conteo calculado localmente (ya leído), pero para evitar
           discrepancias usaremos counts calculados por root basados en total. */
        int k = size;
        long *counts = malloc(k * sizeof(long));
        long *displs = malloc(k * sizeof(long));
        if (!counts || !displs) MPI_Abort(MPI_COMM_WORLD, 1);

        long base = total / k;
        int rem = (int)(total % k);
        for (int i = 0; i < k; ++i) counts[i] = base + (i < rem ? 1 : 0);
        displs[0] = 0;
        for (int i = 1; i < k; ++i) displs[i] = displs[i - 1] + counts[i - 1];

        /* Pero local_data tiene su propio conteo; debemos remapear: cada proceso
           podría tener distinto local_count. Para continuar con el mismo pipeline
           que antes, haremos un MPI_Scatterv desde un root virtual que reconstruye
           el arreglo global: en nuestra versión actual, mejor hacer un paso de
           Allgatherv para unir las lecturas en root y luego proceder como antes.
           Para simplicidad y seguridad (evitar double-free), haremos:
             - Cada proceso ordena su bloque local.
             - Hacemos MPI_Gatherv de los bloques locales ordenados al root.
        */

        /* Ordenar localmente */
        double t_sort_start = MPI_Wtime();
        if (local_count > 0) {
            #pragma omp parallel
            {
                #pragma omp single
                quicksort_parallel(local_data, 0, (local_count > 0 ? local_count - 1 : 0), max_depth);
            }
        }
        double t_sort_end = MPI_Wtime();

        /* Preparar arrays int para MPI_Gatherv */
        int *scounts = malloc(k * sizeof(int));
        int *sdispls = malloc(k * sizeof(int));
        if (!scounts || !sdispls) MPI_Abort(MPI_COMM_WORLD, 1);

        /* reunir los counts locales en root para construir scounts/sdispls */
        long *all_counts = NULL;
        if (rank == 0) all_counts = malloc(k * sizeof(long));
        MPI_Gather(&local_count_long, 1, MPI_LONG, all_counts, 1, MPI_LONG, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            long running = 0;
            for (int i = 0; i < k; ++i) {
                if (all_counts[i] > INT_MAX) { fprintf(stderr, "chunk demasiado grande\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
                scounts[i] = (int)all_counts[i];
                sdispls[i] = (int)running;
                running += all_counts[i];
            }
        }

        /* Root reserva buffer para recibir todos los bloques ordenados */
        double *gather_buf = NULL;
        if (rank == 0) {
            long total_elems = 0;
            for (int i = 0; i < k; ++i) total_elems += (all_counts ? all_counts[i] : 0);
            if (total_elems != total) {
                /* En teoría deben coincidir (Allreduce antes), pero si hay diferencia, usar total */
                total_elems = total;
            }
            gather_buf = malloc((total_elems > 0 ? total_elems : 1) * sizeof(double));
            if (!gather_buf && total_elems > 0) MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Hacemos Gatherv de los datos ordenados al root (cada proceso envía local_count elementos) */
        MPI_Gatherv(local_data, (int)local_count, MPI_DOUBLE,
                    gather_buf, scounts, sdispls, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        /* Root hace merge iterativo de las porciones ya ordenadas */
        double *cur_data = NULL;
        long cur_n = 0;
        int owns_data = 0;
        double t_merge_start = MPI_Wtime();
        if (rank == 0) {
            /* empezar con porción 0 */
            if (scounts[0] > 0) {
                cur_data = malloc((size_t)scounts[0] * sizeof(double));
                if (!cur_data && scounts[0] > 0) MPI_Abort(MPI_COMM_WORLD, 1);
                memcpy(cur_data, gather_buf + sdispls[0], (size_t)scounts[0] * sizeof(double));
                cur_n = scounts[0];
                owns_data = 1;
            }
            for (int src = 1; src < k; ++src) {
                long src_n = (long)scounts[src];
                if (src_n == 0) continue;
                double *src_ptr = gather_buf + sdispls[src];

                if (cur_n == 0) {
                    cur_data = malloc((size_t)src_n * sizeof(double));
                    if (!cur_data && src_n > 0) MPI_Abort(MPI_COMM_WORLD, 1);
                    memcpy(cur_data, src_ptr, (size_t)src_n * sizeof(double));
                    cur_n = src_n;
                    owns_data = 1;
                } else {
                    double *merged = merge_two(cur_data, cur_n, src_ptr, src_n);
                    if (!merged) { fprintf(stderr, "merge_two devolvió NULL\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
                    if (owns_data && cur_data) { free(cur_data); cur_data = NULL; owns_data = 0; }
                    cur_data = merged;
                    cur_n = cur_n + src_n;
                    owns_data = 1;
                }
            }
        }
        double t_merge_end = MPI_Wtime();

        /* Recolección de tiempos (máximos por etapa) */
        double scatter_time = 0.0; /* no usamos scatter en este pipeline */
        double sort_time = t_sort_end - t_sort_start;
        double merge_time = t_merge_end - t_merge_start;
        double io_time = t_io_end - t_io_start;

        double max_scatter=0.0, max_sort=0.0, max_merge=0.0, max_io=0.0;
        MPI_Reduce(&scatter_time, &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&sort_time, &max_sort, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&merge_time, &max_merge, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&io_time, &max_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        /* Memoria local y suma total */
        double local_mem = ((double)local_count * sizeof(double)) / (1024.0 * 1024.0);
        double sum_mem = 0.0;
        MPI_Reduce(&local_mem, &sum_mem, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double dur_total = MPI_Wtime() - t0;
            tiempo_total_prom += dur_total;
            printf("Rep %d: tiempos (s): I/O=%.6f scatter=%.6f sort_max=%.6f merge_max=%.6f total=%.6f\n",
                   rep + 1, max_io, max_scatter, max_sort, max_merge, dur_total);
            printf("Memoria total (suma procesos): %.2f MB\n", sum_mem);
        }

        /* liberaciones */
        if (local_data) { free(local_data); local_data = NULL; }
        if (rank == 0) {
            if (gather_buf) { free(gather_buf); gather_buf = NULL; }
            if (cur_data && owns_data) { free(cur_data); cur_data = NULL; owns_data = 0; }
            if (all_counts) { free(all_counts); all_counts = NULL; }
        }
        free(counts); free(displs); free(scounts); free(sdispls);
    } /* fin repeticiones */

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
