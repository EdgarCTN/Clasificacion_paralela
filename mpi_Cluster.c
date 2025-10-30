#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define REPETICIONES 5

// Función de comparación para qsort
int comparar(const void *a, const void *b) {
    double x = *(double *)a;
    double y = *(double *)b;
    return (x > y) - (x < y);
}

// Función para combinar dos arreglos ordenados
void merge(double *a, int n, double *b, int m, double *resultado) {
    int i = 0, j = 0, k = 0;
    while (i < n && j < m) {
        if (a[i] < b[j])
            resultado[k++] = a[i++];
        else
            resultado[k++] = b[j++];
    }
    while (i < n) resultado[k++] = a[i++];
    while (j < m) resultado[k++] = b[j++];
}

int main(int argc, char *argv[]) {
    int rank, size;
    double tiempo_total = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "Error: se requieren al menos 2 procesos (1 maestro + N esclavos)\n");
        MPI_Finalize();
        return 1;
    }

    for (int rep = 0; rep < REPETICIONES; rep++) {
        double inicio_global, fin_global;
        inicio_global = MPI_Wtime();

        double *datos = NULL;
        long total = 0;

        if (rank == 0) {
            FILE *archivo = fopen("dataset.csv", "r");
            if (archivo == NULL) {
                fprintf(stderr, "Error: no se pudo abrir dataset.csv\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            double valor;
            while (fscanf(archivo, "%lf", &valor) == 1)
                total++;
            rewind(archivo);

            datos = (double *)malloc(total * sizeof(double));
            for (long i = 0; i < total; i++)
                fscanf(archivo, "%lf", &datos[i]);
            fclose(archivo);

            double memoria_usada = (total * sizeof(double)) / (1024.0 * 1024.0);
            printf("[Iteración %d] Dataset cargado: %ld elementos\n", rep + 1, total);
            printf("Memoria usada: %.2f MB\n", memoria_usada);
        }

        MPI_Bcast(&total, 1, MPI_LONG, 0, MPI_COMM_WORLD);

        long chunk = total / size;
        double *subdatos = (double *)malloc(chunk * sizeof(double));

        // Enviar datos a cada proceso
        MPI_Scatter(datos, chunk, MPI_DOUBLE, subdatos, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Cada proceso ordena su parte
        qsort(subdatos, chunk, sizeof(double), comparar);

        // Recolectar los resultados en el maestro
        double *resultado_final = NULL;
        if (rank == 0) resultado_final = (double *)malloc(total * sizeof(double));

        MPI_Gather(subdatos, chunk, MPI_DOUBLE, resultado_final, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            // Combinar todas las partes (simplemente combinar progresivamente)
            for (int i = 1; i < size; i++) {
                merge(resultado_final, chunk * i, resultado_final + chunk * i, chunk,
                      resultado_final);
            }

            fin_global = MPI_Wtime();
            double duracion = fin_global - inicio_global;
            tiempo_total += duracion;

            printf("Iteración %d completada en %.6f segundos\n", rep + 1, duracion);
            free(resultado_final);
            free(datos);
        }

        free(subdatos);
    }

    if (rank == 0) {
        double tiempo_promedio = tiempo_total / REPETICIONES;
        printf("\n=====================================\n");
        printf("MPI Maestro–Esclavo finalizado\n");
        printf("Procesos totales: %d\n", size);
        printf("Tiempo promedio: %.6f segundos\n", tiempo_promedio);
        printf("=====================================\n");

        FILE *res = fopen("resultados_mpi.csv", "a");
        if (res) {
            fprintf(res, "%d,%.6f\n", size, tiempo_promedio);
            fclose(res);
        } else {
            fprintf(stderr, "Advertencia: no se pudo guardar resultados_mpi.csv\n");
        }
    }

    MPI_Finalize();
    return 0;
}
