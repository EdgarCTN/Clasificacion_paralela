/*
 * ===============================================================
 *  PROGRAMA: mpi_maestro_esclavo.c
 *  PROPÓSITO:
 *     Implementar un esquema Maestro–Esclavo (Master–Slave) en MPI
 *     para realizar el ordenamiento paralelo de un dataset numérico
 *     usando QuickSort y una fusión jerárquica.
 *
 *  DESCRIPCIÓN GENERAL:
 *     - El proceso Maestro (rank 0) lee todo el archivo dataset.csv.
 *     - Divide los datos y los distribuye entre los procesos Esclavos.
 *     - Cada Esclavo ordena su subconjunto de datos localmente.
 *     - El Maestro recolecta los resultados y realiza la fusión final.
 *     - Se repite el proceso varias veces para calcular un promedio.
 *
 * ===============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define REPETICIONES 5  // Número de veces que se repite para promediar el tiempo

// ---------------------------------------------------------------
// Función comparadora usada por qsort
// ---------------------------------------------------------------
int comparar(const void *a, const void *b) {
    double x = *(double *)a;
    double y = *(double *)b;
    return (x > y) - (x < y);
}

// ---------------------------------------------------------------
// Función de fusión: combina dos arreglos ordenados en uno nuevo
// ---------------------------------------------------------------
void merge(double *a, long n, double *b, long m, double *resultado) {
    long i = 0, j = 0, k = 0;
    while (i < n && j < m) {
        if (a[i] < b[j]) resultado[k++] = a[i++];
        else resultado[k++] = b[j++];
    }
    while (i < n) resultado[k++] = a[i++];
    while (j < m) resultado[k++] = b[j++];
}

// ---------------------------------------------------------------
// Verifica si el arreglo está ordenado (control de calidad)
// ---------------------------------------------------------------
int verificar_orden(double *arr, long n) {
    for (long i = 1; i < n; i++) {
        if (arr[i] < arr[i - 1]) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size;
    double tiempo_total = 0.0;

    MPI_Init(&argc, &argv);               // Inicializa el entorno MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtiene el ID del proceso actual
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Obtiene el número total de procesos

    // Validación: se requiere al menos un maestro y un esclavo
    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "Error: se requieren al menos 2 procesos (1 maestro + N esclavos)\n");
        MPI_Finalize();
        return 1;
    }

    // ---------------------------------------------------------------
    // Repetimos el experimento varias veces para promediar el tiempo
    // ---------------------------------------------------------------
    for (int rep = 0; rep < REPETICIONES; rep++) {
        double inicio_global = MPI_Wtime();  // Inicio del cronómetro global

        double *datos = NULL;
        long total = 0;

        // ===========================================================
        // ROL DEL MAESTRO (rank 0)
        // ===========================================================
        if (rank == 0) {
            FILE *archivo = fopen("dataset.csv", "r");
            if (archivo == NULL) {
                fprintf(stderr, "Error: no se pudo abrir dataset.csv\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Contar cuántos elementos hay
            double valor;
            while (fscanf(archivo, "%lf", &valor) == 1)
                total++;
            rewind(archivo);

            // Reservar memoria y cargar todos los datos
            datos = (double *)malloc(total * sizeof(double));
            for (long i = 0; i < total; i++)
                fscanf(archivo, "%lf", &datos[i]);
            fclose(archivo);

            double memoria_usada = (total * sizeof(double)) / (1024.0 * 1024.0);
            printf("\n[Iteración %d] Dataset cargado: %ld elementos\n", rep + 1, total);
            printf("Memoria usada: %.2f MB\n", memoria_usada);
        }

        // ===========================================================
        // DIFUSIÓN DE INFORMACIÓN (Maestro → Esclavos)
        // El maestro informa a todos cuántos datos hay en total
        // ===========================================================
        MPI_Bcast(&total, 1, MPI_LONG, 0, MPI_COMM_WORLD);

        long base = total / size;
        long resto = total % size;

        // Cálculo de cuántos elementos tendrá cada proceso
        int *cuentas = (int *)malloc(size * sizeof(int));
        int *desplazamientos = (int *)malloc(size * sizeof(int));

        for (int i = 0; i < size; i++) {
            cuentas[i] = (int)(base + (i == size - 1 ? resto : 0));
            desplazamientos[i] = (i == 0) ? 0 : desplazamientos[i - 1] + cuentas[i - 1];
        }

        // Cada proceso (maestro y esclavos) reserva su buffer local
        double *subdatos = (double *)malloc(cuentas[rank] * sizeof(double));

        // ===========================================================
        // DISTRIBUCIÓN DE TRABAJO (Scatterv)
        // El maestro reparte fragmentos del arreglo a cada esclavo
        // ===========================================================
        MPI_Scatterv(datos, cuentas, desplazamientos, MPI_DOUBLE,
                     subdatos, cuentas[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // ===========================================================
        // ROL DE LOS ESCLAVOS
        // Cada proceso ordena su subconjunto de datos localmente
        // ===========================================================
        qsort(subdatos, cuentas[rank], sizeof(double), comparar);

        // ===========================================================
        // RECOLECCIÓN DE RESULTADOS (Gatherv)
        // Los esclavos devuelven los datos ordenados al maestro
        // ===========================================================
        double *resultado_final = NULL;
        if (rank == 0)
            resultado_final = (double *)malloc(total * sizeof(double));

        MPI_Gatherv(subdatos, cuentas[rank], MPI_DOUBLE,
                    resultado_final, cuentas, desplazamientos, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // ===========================================================
        // ROL DEL MAESTRO: FUSIÓN FINAL
        // Combina los fragmentos ya ordenados de todos los esclavos
        // ===========================================================
        if (rank == 0) {
            double *temp = (double *)malloc(total * sizeof(double));
            long acumulado = cuentas[0];

            for (int i = 1; i < size; i++) {
                merge(resultado_final, acumulado,
                      resultado_final + acumulado, cuentas[i], temp);
                acumulado += cuentas[i];

                // Copiar la parte fusionada de nuevo
                for (long k = 0; k < acumulado; k++)
                    resultado_final[k] = temp[k];
            }
            free(temp);

            // -------------------------------------------------------
            // Verificación del resultado y tiempo total
            // -------------------------------------------------------
            double fin_global = MPI_Wtime();
            double duracion = fin_global - inicio_global;
            tiempo_total += duracion;

            if (!verificar_orden(resultado_final, total))
                printf("⚠️  Advertencia: el arreglo final no está completamente ordenado.\n");

            printf("Iteración %d completada en %.6f segundos\n", rep + 1, duracion);

            free(resultado_final);
            free(datos);
        }

        // Liberar memoria temporal
        free(subdatos);
        free(cuentas);
        free(desplazamientos);
    }

    // ===============================================================
    // PROMEDIO FINAL Y REPORTE
    // Solo el maestro imprime y guarda los resultados
    // ===============================================================
    if (rank == 0) {
        double tiempo_promedio = tiempo_total / REPETICIONES;
        printf("\n=====================================\n");
        printf("MPI Maestro–Esclavo (Optimizado)\n");
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
