#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <sys/resource.h>

/*
 * Programa: mpi.c
 * Propósito: Implementar un esquema "Divide y Vencerás" en paralelo usando MPI.
 *
 * Este programa:
 *  - Carga un dataset numérico desde "dataset.csv" (solo el proceso maestro).
 *  - Divide el arreglo entre los procesos MPI (esquema maestro-esclavo).
 *  - Cada proceso ordena su segmento con qsort() (divide y vencerás local).
 *  - El maestro recolecta y combina los datos ordenados.
 *  - Mide el tiempo total y muestra estadísticas (memoria, min, max, promedio).
 *
 * Compilación: mpicc mpi.c -o mpi -O2
 * Ejecución:   mpirun -np 4 ./mpi
 *
 * Sistema: Ubuntu/Linux
 */

int comparar(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

double tiempo_actual() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int verificar_orden(const double *datos, long total) {
    for (long i = 1; i < total; i++) {
        if (datos[i] < datos[i - 1]) return 0;
    }
    return 1;
}

void fusionar(const double *a, long na, const double *b, long nb, double *resultado) {
    long i = 0, j = 0, k = 0;
    while (i < na && j < nb) {
        if (a[i] < b[j]) resultado[k++] = a[i++];
        else resultado[k++] = b[j++];
    }
    while (i < na) resultado[k++] = a[i++];
    while (j < nb) resultado[k++] = b[j++];
}

// Calcular uso de memoria (en MB)
double memoria_usada_MB() {
    struct rusage uso;
    getrusage(RUSAGE_SELF, &uso);
#ifdef __linux__
    return uso.ru_maxrss / 1024.0; 
#else
    return uso.ru_maxrss;
#endif
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *datos = NULL;
    double *subdatos = NULL;
    long total = 0;
    double inicio_global, fin_global;

    if (rank == 0) {
        FILE *archivo = fopen("dataset.csv", "r");
        if (!archivo) {
            fprintf(stderr, "Error: no se pudo abrir dataset.csv\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double temp;
        while (fscanf(archivo, " %lf,", &temp) == 1) total++;
        rewind(archivo);

        datos = (double *)malloc(total * sizeof(double));
        if (!datos) {
            fprintf(stderr, "Error: no se pudo asignar memoria\n");
            fclose(archivo);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (long i = 0; i < total; i++) {
            if (fscanf(archivo, " %lf,", &datos[i]) != 1) {
                fprintf(stderr, "Error al leer dato %ld\n", i);
                free(datos);
                fclose(archivo);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        fclose(archivo);

        // Calcular estadísticas iniciales
        double min = datos[0], max = datos[0], suma = 0.0;
        for (long i = 0; i < total; i++) {
            if (datos[i] < min) min = datos[i];
            if (datos[i] > max) max = datos[i];
            suma += datos[i];
        }
        double promedio = suma / total;
        double memoria = memoria_usada_MB();

        printf("==========================================\n");
        printf("Datos cargados: %ld\n", total);
        printf("Valor mínimo: %.2f\n", min);
        printf("Valor máximo: %.2f\n", max);
        printf("Promedio: %.2f\n", promedio);
        printf("Memoria usada: %.2f MB\n", memoria);
        printf("\nProcesos MPI: %d\n", size);
        printf("Dividiendo datos entre procesos...\n");
        printf("==========================================\n");
    }

    // Difundir total
    MPI_Bcast(&total, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    long tam_local = total / size;
    subdatos = (double *)malloc(tam_local * sizeof(double));
    if (!subdatos) {
        fprintf(stderr, "Error: no se pudo asignar memoria en proceso %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Enviar partes
    MPI_Scatter(datos, tam_local, MPI_DOUBLE, subdatos, tam_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    inicio_global = tiempo_actual();

    qsort(subdatos, tam_local, sizeof(double), comparar);

    MPI_Gather(subdatos, tam_local, MPI_DOUBLE, datos, tam_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double *temp = (double *)malloc(total * sizeof(double));
        long paso = tam_local;
        for (int p = 1; p < size; p++) {
            fusionar(datos, paso * p, &datos[p * tam_local], tam_local, temp);
            for (long i = 0; i < (paso * (p + 1)); i++)
                datos[i] = temp[i];
        }
        free(temp);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    fin_global = tiempo_actual();

    if (rank == 0) {
        double tiempo = fin_global - inicio_global;
        double memoria = memoria_usada_MB();

        printf("Tiempo total de ordenamiento paralelo: %.6f segundos\n", tiempo);
        printf("Memoria usada: %.2f MB\n", memoria);

        if (verificar_orden(datos, total))
            printf("Verificación: datos correctamente ordenados \n");
        else
            printf("Error: datos no ordenados \n");

        FILE *res = fopen("resultados_mpi.csv", "a");
        if (res) {
            fprintf(res, "%d,%.6f,%.2f\n", size, tiempo, memoria);
            fclose(res);
        }

        printf("==========================================\n");
        free(datos);
    }

    free(subdatos);
    MPI_Finalize();
    return 0;
}
