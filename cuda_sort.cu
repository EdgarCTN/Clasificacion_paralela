#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <windows.h>
#include <psapi.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define REPETICIONES 5

/*
 * Programa: cuda_sort.cu (VERSIÓN CON GUARDADO GARANTIZADO)
 * Compilación: nvcc cuda_sort.cu -o cuda_sort.exe -O3 -std=c++17
 * Ejecución: cuda_sort.exe
 */

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

double tiempo_actual() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}

double memoria_usada_MB() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    return pmc.WorkingSetSize / (1024.0 * 1024.0);
}

int verificar_orden(double *datos, long total) {
    for (long i = 1; i < total; i++) {
        if (datos[i] < datos[i - 1]) {
            return 0;
        }
    }
    return 1;
}

void thrust_sort(double *d_data, long n) {
    thrust::device_ptr<double> dev_ptr(d_data);
    thrust::sort(dev_ptr, dev_ptr + n);
}

int main() {
    FILE *archivo;
    double *datos_host;
    long total = 0;
    double valor, min_val = 1e9, max_val = -1e9, suma = 0.0;

    archivo = fopen("dataset.csv", "r");
    if (archivo == NULL) {
        fprintf(stderr, "Error: no se pudo abrir dataset.csv\n");
        return 1;
    }

    while (fscanf(archivo, "%lf", &valor) == 1) total++;
    rewind(archivo);

    datos_host = (double *)malloc(total * sizeof(double));
    if (datos_host == NULL) {
        fprintf(stderr, "Error: memoria insuficiente\n");
        fclose(archivo);
        return 1;
    }

    for (long i = 0; i < total; i++) {
        fscanf(archivo, "%lf", &datos_host[i]);
        if (datos_host[i] < min_val) min_val = datos_host[i];
        if (datos_host[i] > max_val) max_val = datos_host[i];
        suma += datos_host[i];
    }
    fclose(archivo);

    double promedio = suma / total;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("==========================================\n");
    printf("CUDA Parallel Sort - Windows\n");
    printf("==========================================\n");
    printf("Datos cargados: %ld\n", total);
    printf("Valor minimo: %.2f\n", min_val);
    printf("Valor maximo: %.2f\n", max_val);
    printf("Promedio: %.2f\n", promedio);
    printf("Memoria host: %.2f MB\n", (total * sizeof(double)) / (1024.0 * 1024.0));
    printf("\nGPU: %s\n", prop.name);
    printf("Memoria GPU: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Algoritmo: Thrust Sort (optimizado)\n");
    printf("Repeticiones: %d\n", REPETICIONES);
    printf("==========================================\n");

    double *d_data;
    size_t gpu_mem_size = total * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_data, gpu_mem_size));

    double tiempo_total = 0.0;
    double tiempo_copia_total = 0.0;

    for (int rep = 0; rep < REPETICIONES; rep++) {
        double inicio_copia = tiempo_actual();
        CUDA_CHECK(cudaMemcpy(d_data, datos_host, gpu_mem_size, cudaMemcpyHostToDevice));
        double fin_copia = tiempo_actual();
        tiempo_copia_total += (fin_copia - inicio_copia);

        double inicio = tiempo_actual();
        thrust_sort(d_data, total);
        CUDA_CHECK(cudaDeviceSynchronize());
        double fin = tiempo_actual();

        double tiempo_iter = fin - inicio;
        tiempo_total += tiempo_iter;
        printf("Iteracion %d: %.6f segundos (GPU) + %.6f s (copia) = %.6f s total\n", 
               rep + 1, tiempo_iter, fin_copia - inicio_copia, 
               tiempo_iter + (fin_copia - inicio_copia));
    }

    CUDA_CHECK(cudaMemcpy(datos_host, d_data, gpu_mem_size, cudaMemcpyDeviceToHost));

    double tiempo_promedio = tiempo_total / REPETICIONES;
    double tiempo_copia_promedio = tiempo_copia_total / REPETICIONES;
    double memoria_final = memoria_usada_MB();

    printf("\n==========================================\n");
    printf("RESULTADOS FINALES\n");
    printf("==========================================\n");
    printf("Tiempo promedio (solo GPU): %.6f segundos\n", tiempo_promedio);
    printf("Tiempo promedio (copia H->D): %.6f segundos\n", tiempo_copia_promedio);
    printf("Tiempo total promedio: %.6f segundos\n", tiempo_promedio + tiempo_copia_promedio);
    printf("Memoria usada (host): %.2f MB\n", memoria_final);

    if (verificar_orden(datos_host, total)) {
        printf("\nVerificacion: datos correctamente ordenados [OK]\n");
    } else {
        printf("\nVerificacion: ERROR - datos NO ordenados [FALLO]\n");
    }

    // ===== GUARDAR DATASET ORDENADO =====
    printf("\n==========================================\n");
    printf("GUARDANDO DATASET ORDENADO...\n");
    printf("==========================================\n");
    
    FILE *ordenado = fopen("dataset_ordenado.csv", "w");
    if (ordenado == NULL) {
        fprintf(stderr, "ERROR CRITICO: No se pudo crear dataset_ordenado.csv\n");
        fprintf(stderr, "Verifica permisos de escritura\n");
    } else {
        printf("Archivo 'dataset_ordenado.csv' creado exitosamente\n");
        printf("Escribiendo %ld elementos...\n", total);
        
        long escritos = 0;
        for (long i = 0; i < total; i++) {
            fprintf(ordenado, "%.0f\n", datos_host[i]);
            escritos++;
            
            if ((i + 1) % 250000 == 0) {
                printf("  -> %ld de %ld elementos (%.1f%%)\n", 
                       i + 1, total, (i + 1) * 100.0 / total);
                fflush(stdout);
            }
        }
        
        fclose(ordenado);
        
        printf("\n[EXITO] Dataset ordenado guardado: %ld elementos\n", escritos);
        printf("Archivo: dataset_ordenado.csv\n");
        printf("Primeros 10: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f\n",
               datos_host[0], datos_host[1], datos_host[2], datos_host[3], datos_host[4],
               datos_host[5], datos_host[6], datos_host[7], datos_host[8], datos_host[9]);
        printf("Ultimos 10: %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f\n",
               datos_host[total-10], datos_host[total-9], datos_host[total-8], 
               datos_host[total-7], datos_host[total-6], datos_host[total-5],
               datos_host[total-4], datos_host[total-3], datos_host[total-2], datos_host[total-1]);
    }

    // ===== GUARDAR ESTADÍSTICAS =====
    FILE *resultados = fopen("resultados_cuda.csv", "a");
    if (resultados) {
        fprintf(resultados, "%ld,%.6f,%.6f,%.2f,%.2f,%.2f,%s\n",
                total, tiempo_promedio, tiempo_copia_promedio,
                min_val, max_val, promedio, prop.name);
        fclose(resultados);
        printf("\nEstadisticas guardadas en 'resultados_cuda.csv'\n");
    }

    printf("==========================================\n");

    cudaFree(d_data);
    free(datos_host);

    return 0;
}