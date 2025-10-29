#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#define REPETICIONES 5  // Número de veces que se repite el ordenamiento

/*
 * Programa: openmp.c
 * Propósito: Medir el desempeño del ordenamiento paralelo (QuickSort) usando OpenMP.
 *
 * Este programa:
 *  - Lee "dataset.csv" (una lista de valores numéricos separados por saltos de línea).
 *  - Calcula estadísticas básicas (mínimo, máximo, promedio).
 *  - Ordena los datos en paralelo con OpenMP varias veces para obtener un tiempo promedio.
 *  - Verifica que los datos se hayan ordenado correctamente.
 *  - Guarda los resultados (N, tiempo promedio, min, max, promedio) en "resultados.csv".
 *
 * Grupo 5 – Programación Paralela
 * Sistema: Ubuntu/Linux
 */

int comparar(const void *a, const void *b) {
    double x = *(double *)a;
    double y = *(double *)b;
    return (x > y) - (x < y);
}

// Función para medir tiempo de alta precisión
double tiempo_actual() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Verificar orden ascendente
int verificar_orden(double *datos, long total) {
    for (long i = 1; i < total; i++) {
        if (datos[i] < datos[i - 1]) return 0;
    }
    return 1;
}

// QuickSort paralelo con OpenMP (divide y vencerás)
void quicksort_parallel(double *arr, long left, long right, int profundidad) {
    if (left >= right) return;

    double pivote = arr[(left + right) / 2];
    long i = left, j = right;

    while (i <= j) {
        while (arr[i] < pivote) i++;
        while (arr[j] > pivote) j--;
        if (i <= j) {
            double tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++;
            j--;
        }
    }

    // Crear tareas solo si hay suficiente profundidad y tamaño del arreglo
    if (profundidad > 0 && (right - left) > 10000) {
        #pragma omp task shared(arr) firstprivate(left, j, profundidad)
        quicksort_parallel(arr, left, j, profundidad - 1);

        #pragma omp task shared(arr) firstprivate(i, right, profundidad)
        quicksort_parallel(arr, i, right, profundidad - 1);

        #pragma omp taskwait
    } else {
        quicksort_parallel(arr, left, j, 0);
        quicksort_parallel(arr, i, right, 0);
    }
}

int main() {
    FILE *archivo;
    double *datos;
    long total = 0;
    double valor, min = 1e9, max = -1e9, suma = 0.0;

    archivo = fopen("dataset.csv", "r");
    if (archivo == NULL) {
        fprintf(stderr, "Error: no se pudo abrir dataset.csv\n");
        return 1;
    }

    // Contar cantidad de datos
    while (fscanf(archivo, "%lf", &valor) == 1) {
        total++;
    }
    rewind(archivo);

    if (total == 0) {
        fprintf(stderr, "Error: el dataset está vacío o mal formateado.\n");
        fclose(archivo);
        return 1;
    }

    datos = (double *)malloc(total * sizeof(double));
    if (datos == NULL) {
        fprintf(stderr, "Error: no se pudo asignar memoria (%.2f MB requeridos)\n",
                total * sizeof(double) / (1024.0 * 1024.0));
        fclose(archivo);
        return 1;
    }

    // Leer datos y calcular estadísticas básicas
    for (long i = 0; i < total; i++) {
        if (fscanf(archivo, "%lf", &datos[i]) != 1) {
            fprintf(stderr, "Error al leer dato %ld\n", i);
            free(datos);
            fclose(archivo);
            return 1;
        }
        if (datos[i] < min) min = datos[i];
        if (datos[i] > max) max = datos[i];
        suma += datos[i];
    }
    fclose(archivo);

    double promedio = suma / total;

    printf("==========================================\n");
    printf("Datos cargados: %ld\n", total);
    printf("Valor mínimo: %.2f\n", min);
    printf("Valor máximo: %.2f\n", max);
    printf("Promedio: %.2f\n", promedio);
    printf("Memoria usada: %.2f MB\n", total * sizeof(double) / (1024.0 * 1024.0));
    printf("Repeticiones para promedio: %d\n", REPETICIONES);
    printf("Hilos OpenMP detectados: %d\n", omp_get_max_threads());
    printf("==========================================\n");

    // Profundidad máxima adaptativa
    int max_depth = (int)log2(omp_get_max_threads());
    if (max_depth < 1) max_depth = 1;

    double tiempo_total = 0.0;

    for (int r = 0; r < REPETICIONES; r++) {
        double *copia = (double *)malloc(total * sizeof(double));
        if (copia == NULL) {
            fprintf(stderr, "Error: no se pudo asignar memoria para copia\n");
            free(datos);
            return 1;
        }

        for (long i = 0; i < total; i++) copia[i] = datos[i];

        double inicio = tiempo_actual();

        #pragma omp parallel
        {
            #pragma omp single
            {
                quicksort_parallel(copia, 0, total - 1, max_depth);
            }
        }

        double fin = tiempo_actual();
        tiempo_total += (fin - inicio);
        free(copia);
    }

    double tiempo_promedio = tiempo_total / REPETICIONES;
    printf("\nTiempo promedio (QuickSort OpenMP): %.6f segundos\n", tiempo_promedio);

    // Verificación de orden
    double *verif = (double *)malloc(total * sizeof(double));
    for (long i = 0; i < total; i++) verif[i] = datos[i];
    #pragma omp parallel
    {
        #pragma omp single
        quicksort_parallel(verif, 0, total - 1, max_depth);
    }
    int orden_ok = verificar_orden(verif, total);
    free(verif);

    if (orden_ok)
        printf("Verificación: datos correctamente ordenados\n");
    else
        printf("Verificación: error, datos no están ordenados\n");

    // Guardar resultados
    FILE *resultados = fopen("resultados.csv", "a");
    if (resultados) {
        fprintf(resultados, "%ld,%.6f,%.2f,%.2f,%.2f\n",
                total, tiempo_promedio, min, max, promedio);
        fclose(resultados);
        printf("\nResultados guardados en 'resultados.csv'\n");
    } else {
        fprintf(stderr, "Advertencia: no se pudo escribir en resultados.csv\n");
    }

    free(datos);
    printf("==========================================\n");
    return 0;
}
