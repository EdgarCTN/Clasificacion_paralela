#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define REPETICIONES 5  // Número de veces que se repite el ordenamiento para promedio

/*
 * Programa: secuencial_final_v2.c
 * Propósito: Medir el desempeño del ordenamiento secuencial (QuickSort)
 * sobre un dataset numérico grande en C.
 *
 * Este programa:
 *  - Lee "dataset.csv" (una lista de valores numéricos separados por saltos de línea).
 *  - Calcula estadísticas básicas (mínimo, máximo, promedio).
 *  - Ordena los datos con qsort() varias veces para obtener un tiempo promedio.
 *  - Verifica que los datos se hayan ordenado correctamente.
 *  - Guarda los resultados (N, tiempo promedio, min, max, promedio) en "resultados.csv".
 *
 * Grupo 5 – Programación Paralela
 * Sistema: Ubuntu/Linux
 */

// Función para comparar dos elementos (usada por qsort)
int comparar(const void *a, const void *b) {
    double x = *(double *)a;
    double y = *(double *)b;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

// Medir tiempo real en segundos (alta precisión)
double tiempo_actual() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Verifica que los datos estén ordenados de forma ascendente
int verificar_orden(double *datos, long total) {
    for (long i = 1; i < total; i++) {
        if (datos[i] < datos[i - 1]) return 0; // No está ordenado
    }
    return 1;
}

int main() {
    FILE *archivo;
    double *datos;
    long total = 0;
    double valor, min = 1e9, max = -1e9, suma = 0.0;

    // Abrir el archivo con los datos
    archivo = fopen("dataset.csv", "r");
    if (archivo == NULL) {
        fprintf(stderr, "Error: no se pudo abrir el archivo dataset.csv\n");
        return 1;
    }

    // Contar cuántos datos contiene el archivo
    while (fscanf(archivo, "%lf", &valor) == 1) {
        total++;
    }
    rewind(archivo);

    if (total == 0) {
        fprintf(stderr, "Error: el archivo dataset.csv está vacío o tiene formato incorrecto.\n");
        fclose(archivo);
        return 1;
    }

    // Reservar memoria dinámica según la cantidad de datos
    datos = (double *)malloc(total * sizeof(double));
    if (datos == NULL) {
        fprintf(stderr, "Error: no se pudo asignar memoria (%.2f MB requeridos)\n",
                total * sizeof(double) / (1024.0 * 1024.0));
        fclose(archivo);
        return 1;
    }

    // Leer los datos y calcular estadísticas básicas
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

    // Medir el tiempo promedio de ordenamiento con qsort()
    double tiempo_total = 0.0;
    for (int r = 0; r < REPETICIONES; r++) {
        // Duplicar los datos originales para no ordenar el mismo arreglo
        double *copia = (double *)malloc(total * sizeof(double));
        if (copia == NULL) {
            fprintf(stderr, "Error: no se pudo asignar memoria para copia\n");
            free(datos);
            return 1;
        }
        for (long i = 0; i < total; i++) copia[i] = datos[i];

        double inicio = tiempo_actual();
        qsort(copia, total, sizeof(double), comparar);
        double fin = tiempo_actual();

        tiempo_total += (fin - inicio);
        free(copia);
    }

    double tiempo_promedio = tiempo_total / REPETICIONES;
    printf("\nTiempo promedio (QuickSort secuencial): %.6f segundos\n", tiempo_promedio);

    // Verificar orden en la última copia ordenada (solo una vez)
    double *verif = (double *)malloc(total * sizeof(double));
    for (long i = 0; i < total; i++) verif[i] = datos[i];
    qsort(verif, total, sizeof(double), comparar);
    int orden_ok = verificar_orden(verif, total);
    free(verif);

    if (orden_ok)
        printf("Verificación: datos correctamente ordenados\n");
    else
        printf("Verificación: error, datos no están ordenados\n");

    // Guardar resultados en formato CSV
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
