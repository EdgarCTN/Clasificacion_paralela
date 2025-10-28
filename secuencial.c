#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Programa: secuencial_final.c
 * Propósito: Medir el desempeño del ordenamiento secuencial (QuickSort)
 * sobre un dataset numérico grande en C.
 * 
 * Este programa:
 *  - Lee un archivo "dataset.csv" con una lista de valores numéricos.
 *  - Calcula estadísticas básicas (mínimo, máximo, promedio).
 *  - Ordena los datos con qsort() y mide el tiempo de ejecución.
 *  - Verifica que los datos se hayan ordenado correctamente.
 *  - Guarda los resultados de tiempo en "resultados.csv" para análisis posterior.
 *
 * Autor: Edgar Tejeda (2025)
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
    double inicio, fin;

    // Abrir el archivo con los datos
    archivo = fopen("dataset.csv", "r");
    if (archivo == NULL) {
        fprintf(stderr, "Error: no se pudo abrir el archivo dataset.csv\n");
        return 1;
    }

    // Contar cuántos datos contiene el archivo
    while (fscanf(archivo, " %lf,", &valor) == 1) {
        total++;
    }
    rewind(archivo); // Regresar al inicio del archivo

    // Reservar memoria dinámica según la cantidad de datos
    datos = (double *)malloc(total * sizeof(double));
    if (datos == NULL) {
        fprintf(stderr, "Error: no se pudo asignar memoria\n");
        fclose(archivo);
        return 1;
    }

    // Leer los datos y calcular estadísticas básicas
    for (long i = 0; i < total; i++) {
        if (fscanf(archivo, " %lf,", &datos[i]) != 1) {
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

    printf("==========================================\n");
    printf("Datos cargados: %ld\n", total);
    printf("Valor mínimo: %.2f\n", min);
    printf("Valor máximo: %.2f\n", max);
    printf("Promedio: %.2f\n", suma / total);
    printf("Memoria usada: %.2f MB\n", total * sizeof(double) / (1024.0 * 1024.0));

    // Medir el tiempo de ordenamiento con qsort (QuickSort secuencial)
    inicio = tiempo_actual();
    qsort(datos, total, sizeof(double), comparar);
    fin = tiempo_actual();

    double tiempo = fin - inicio;
    printf("\nTiempo de ordenamiento (QuickSort secuencial): %.6f segundos\n", tiempo);

    // Verificar que los datos estén correctamente ordenados
    if (verificar_orden(datos, total)) {
        printf("Verificación: datos correctamente ordenados \n");
    } else {
        printf("Verificación: error, datos no están ordenados \n");
    }

    // Guardar datos ordenados (opcional)
    FILE *salida = fopen("ordenado.csv", "w");
    if (salida) {
        for (long i = 0; i < total; i++) {
            fprintf(salida, "%.6f\n", datos[i]);
        }
        fclose(salida);
    }

    // Guardar los resultados en formato CSV (para análisis posterior)
    FILE *resultados = fopen("resultados.csv", "a");
    if (resultados) {
        fprintf(resultados, "%ld,%.6f\n", total, tiempo);
        fclose(resultados);
    }

    free(datos);
    printf("==========================================\n");
    return 0;
}
