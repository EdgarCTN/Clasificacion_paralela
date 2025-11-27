#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define REPETICIONES 3
#define CHUNK_SIZE_MB 8

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

std::vector<double> leer_csv(const char* archivo) {
    FILE* f = fopen(archivo, "r");
    if (!f) {
        fprintf(stderr, "Error abriendo %s\n", archivo);
        exit(1);
    }
    std::vector<double> datos;
    double val;
    while (fscanf(f, "%lf", &val) == 1) datos.push_back(val);
    fclose(f);
    return datos;
}

// Calcular desorden (inversiones) de un chunk
int count_inversions_sample(double *arr, size_t n, size_t sample_size = 100) {
    if (n < sample_size) sample_size = n;
    
    int inversions = 0;
    size_t step = n / sample_size;
    if (step < 1) step = 1;
    
    for (size_t i = 0; i < n - step; i += step) {
        for (size_t j = i + step; j < n; j += step) {
            if (arr[i] > arr[j]) inversions++;
        }
    }
    
    return inversions;
}

enum TipoRama {
    RAMA_CPU_SIMPLE,   // CPU single-thread (baja complejidad)
    RAMA_CPU_PARALELO, // CPU multi-thread (complejidad media)
    RAMA_GPU           // GPU (alta complejidad)
};

struct ChunkInfo {
    size_t offset;
    size_t size;
    TipoRama rama;
    int complejidad;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Uso: %s <dataset.csv> [num_hilos]\n", argv[0]);
        return 1;
    }

    const char* archivo = argv[1];
    int num_hilos = (argc >= 3) ? atoi(argv[2]) : omp_get_max_threads();
    if (num_hilos < 1) num_hilos = 1;
    
    omp_set_num_threads(num_hilos);
    
    printf("Esquema de Ramificacion Pura CUDA+OpenMP\n");
    printf("OpenMP activo con %d hilos\n", num_hilos);

    // Cargar datos originales
    auto datos_originales = leer_csv(archivo);
    size_t N = datos_originales.size();
    size_t chunk_size = (CHUNK_SIZE_MB * 1024 * 1024) / sizeof(double);
    size_t num_chunks = (N + chunk_size - 1) / chunk_size;
    
    printf("Datos cargados: %zu\n", N);
    printf("Chunks: %zu, chunk_size = %zu\n\n", num_chunks, chunk_size);

    double tiempo_total = 0.0;
    double tiempo_analisis_total = 0.0;
    double tiempo_ejecucion_total = 0.0;
    
    int chunks_cpu_simple_total = 0;
    int chunks_cpu_paralelo_total = 0;
    int chunks_gpu_total = 0;

    for (int rep = 0; rep < REPETICIONES; rep++) {
        printf("Iteracion %d...\n", rep + 1);
        
        // Copia fresca cada iteración
        std::vector<double> datos = datos_originales;
        std::vector<ChunkInfo> chunks_info(num_chunks);
        
        double t0_total = omp_get_wtime();
        
        // ==== FASE 1: ANALISIS Y CLASIFICACION (CPU paralelo) ====
        double t0_analisis = omp_get_wtime();
        
        int chunks_cpu_simple = 0;
        int chunks_cpu_paralelo = 0;
        int chunks_gpu = 0;
        
        #pragma omp parallel for num_threads(num_hilos) \
            reduction(+:chunks_cpu_simple,chunks_cpu_paralelo,chunks_gpu)
        for (int c = 0; c < (int)num_chunks; c++) {
            size_t offset = c * chunk_size;
            size_t size = (offset + chunk_size > N) ? (N - offset) : chunk_size;
            
            chunks_info[c].offset = offset;
            chunks_info[c].size = size;
            
            double *chunk_ptr = datos.data() + offset;
            
            // Medir complejidad del chunk (número de inversiones)
            int inversions = count_inversions_sample(chunk_ptr, size);
            chunks_info[c].complejidad = inversions;
            
            // RAMIFICACION PURA: Clasificar según complejidad
            // TODOS los chunks se procesan, ninguno se salta
            if (inversions < 20) {
                // Baja complejidad → CPU single-thread
                chunks_info[c].rama = RAMA_CPU_SIMPLE;
                chunks_cpu_simple++;
            } else if (inversions < 50) {
                // Complejidad media → CPU multi-thread
                chunks_info[c].rama = RAMA_CPU_PARALELO;
                chunks_cpu_paralelo++;
            } else {
                // Alta complejidad → GPU
                chunks_info[c].rama = RAMA_GPU;
                chunks_gpu++;
            }
        }
        
        double t1_analisis = omp_get_wtime();
        double tiempo_analisis = t1_analisis - t0_analisis;
        tiempo_analisis_total += tiempo_analisis;
        
        chunks_cpu_simple_total += chunks_cpu_simple;
        chunks_cpu_paralelo_total += chunks_cpu_paralelo;
        chunks_gpu_total += chunks_gpu;
        
        // ==== FASE 2: EJECUCION EN PARALELO (RAMIFICACION PURA) ====
        double t0_ejecucion = omp_get_wtime();
        
        // Preparar buffer GPU si hay chunks GPU
        double *d_buffer = nullptr;
        size_t max_gpu_chunk_size = 0;
        
        if (chunks_gpu > 0) {
            for (size_t c = 0; c < num_chunks; c++) {
                if (chunks_info[c].rama == RAMA_GPU && chunks_info[c].size > max_gpu_chunk_size) {
                    max_gpu_chunk_size = chunks_info[c].size;
                }
            }
            CUDA_CHECK(cudaMalloc(&d_buffer, max_gpu_chunk_size * sizeof(double)));
        }
        
        // RAMIFICACION: CPU y GPU trabajan EN PARALELO
        // Usamos OpenMP sections para ejecutar ambas ramas simultáneamente
        #pragma omp parallel sections num_threads(2)
        {
            // ===== RAMA CPU =====
            #pragma omp section
            {
                // Sub-rama 1: CPU_SIMPLE (secuencial)
                for (size_t c = 0; c < num_chunks; c++) {
                    if (chunks_info[c].rama == RAMA_CPU_SIMPLE) {
                        double *chunk_ptr = datos.data() + chunks_info[c].offset;
                        std::sort(chunk_ptr, chunk_ptr + chunks_info[c].size);
                    }
                }
                
                // Sub-rama 2: CPU_PARALELO (OpenMP nested)
                #pragma omp parallel for num_threads(num_hilos)
                for (int c = 0; c < (int)num_chunks; c++) {
                    if (chunks_info[c].rama == RAMA_CPU_PARALELO) {
                        double *chunk_ptr = datos.data() + chunks_info[c].offset;
                        std::sort(chunk_ptr, chunk_ptr + chunks_info[c].size);
                    }
                }
            }
            
            // ===== RAMA GPU =====
            #pragma omp section
            {
                if (chunks_gpu > 0) {
                    // Procesar todos los chunks GPU
                    for (size_t c = 0; c < num_chunks; c++) {
                        if (chunks_info[c].rama == RAMA_GPU) {
                            double *chunk_ptr = datos.data() + chunks_info[c].offset;
                            size_t size = chunks_info[c].size;
                            
                            // Transferir a GPU
                            CUDA_CHECK(cudaMemcpy(d_buffer, chunk_ptr, size * sizeof(double),
                                                 cudaMemcpyHostToDevice));
                            
                            // Ordenar en GPU
                            thrust::device_ptr<double> dev_ptr(d_buffer);
                            thrust::sort(dev_ptr, dev_ptr + size);
                            
                            // Transferir de vuelta
                            CUDA_CHECK(cudaMemcpy(chunk_ptr, d_buffer, size * sizeof(double),
                                                 cudaMemcpyDeviceToHost));
                        }
                    }
                }
            }
        }
        // ← Barrera implícita: ambas ramas terminan aquí antes de continuar
        
        // Liberar memoria GPU
        if (d_buffer) {
            CUDA_CHECK(cudaFree(d_buffer));
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double t1_ejecucion = omp_get_wtime();
        double tiempo_ejecucion = t1_ejecucion - t0_ejecucion;
        tiempo_ejecucion_total += tiempo_ejecucion;
        
        double t1_total = omp_get_wtime();
        double duracion = t1_total - t0_total;
        tiempo_total += duracion;
        
        printf("Iteracion %d completada en %.4f s\n", rep + 1, duracion);
        printf("  -> Analisis: %.4f s\n", tiempo_analisis);
        printf("  -> Ejecucion paralela (ramificacion): %.4f s\n", tiempo_ejecucion);
        printf("  -> Distribucion: CPU_Simple=%d, CPU_Paralelo=%d, GPU=%d\n",
               chunks_cpu_simple, chunks_cpu_paralelo, chunks_gpu);
        
        // Guardar resultado solo en última iteración
        if (rep == REPETICIONES - 1) {
            FILE* out = fopen("dataset_ordenado_ramificacion.csv", "w");
            for (size_t i = 0; i < N; i++) {
                fprintf(out, "%.6f\n", datos[i]);
            }
            fclose(out);
        }
    }

    // ==== ESTADISTICAS FINALES ====
    printf("\n=== Promedios ===\n");
    printf("Tiempo promedio total: %.4f s\n", tiempo_total / REPETICIONES);
    printf("Tiempo promedio analisis: %.4f s\n", tiempo_analisis_total / REPETICIONES);
    printf("Tiempo promedio ejecucion (ramificacion): %.4f s\n", tiempo_ejecucion_total / REPETICIONES);
    
    printf("\n=== Distribucion de Trabajo (Promedio) ===\n");
    printf("Chunks CPU SIMPLE:            %.1f%% (%d/%d)\n",
           100.0 * chunks_cpu_simple_total / (REPETICIONES * num_chunks),
           chunks_cpu_simple_total / REPETICIONES, (int)num_chunks);
    printf("Chunks CPU PARALELO:          %.1f%% (%d/%d)\n",
           100.0 * chunks_cpu_paralelo_total / (REPETICIONES * num_chunks),
           chunks_cpu_paralelo_total / REPETICIONES, (int)num_chunks);
    printf("Chunks GPU:                   %.1f%% (%d/%d)\n",
           100.0 * chunks_gpu_total / (REPETICIONES * num_chunks),
           chunks_gpu_total / REPETICIONES, (int)num_chunks);
    
    printf("\nDataset ordenado guardado: dataset_ordenado_ramificacion.csv\n");

    return 0;
}