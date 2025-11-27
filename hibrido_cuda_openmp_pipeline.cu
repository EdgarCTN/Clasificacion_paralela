/*
 hibrido_cuda_openmp_pipeline.cu
 Pipeline híbrido CUDA + OpenMP optimizado
 Compilar:
   nvcc -O3 -Xcompiler -fopenmp hibrido_cuda_openmp_pipeline.cu -o hibrido_cuda_openmp_pipeline
 Ejecutar:
   ./hibrido_cuda_openmp_pipeline dataset.csv 8
*/

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define REPETICIONES 3
#define NUM_BUFFERS 3

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error en %s:%d: %s\n", __FILE__, __LINE__, \
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

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Uso: %s <dataset.csv> [num_hilos] [chunk_size_MB]\n", argv[0]);
        return 1;
    }

    const char* archivo = argv[1];
    int num_hilos = (argc >= 3) ? atoi(argv[2]) : omp_get_max_threads();
    size_t chunk_mb = (argc >= 4) ? atoi(argv[3]) : 8;
    
    if (num_hilos < 1) num_hilos = 1;
    size_t chunk_size = (chunk_mb * 1024 * 1024) / sizeof(double);
    
    omp_set_num_threads(num_hilos);

    // Cargar datos originales
    auto datos_originales = leer_csv(archivo);
    size_t N = datos_originales.size();
    size_t num_chunks = (N + chunk_size - 1) / chunk_size;
    
    printf("Pipeline Hibrido CUDA+OpenMP\n");
    printf("OpenMP activo con %d hilos\n", num_hilos);
    printf("Datos cargados: %zu\n", N);
    printf("Pipeline: %zu chunks, chunk_size = %zu\n\n", num_chunks, chunk_size);

    // ========== SETUP GPU ==========
    
    cudaStream_t stream_h2d, stream_compute, stream_d2h;
    CUDA_CHECK(cudaStreamCreate(&stream_h2d));
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_d2h));

    // Eventos para dependencias (no para timing detallado)
    cudaEvent_t event_h2d_done, event_compute_done;
    CUDA_CHECK(cudaEventCreate(&event_h2d_done));
    CUDA_CHECK(cudaEventCreate(&event_compute_done));
    
    // Eventos para timing global
    cudaEvent_t event_start, event_end;
    CUDA_CHECK(cudaEventCreate(&event_start));
    CUDA_CHECK(cudaEventCreate(&event_end));

    double *h_input[NUM_BUFFERS], *h_output[NUM_BUFFERS], *d_buffers[NUM_BUFFERS];
    for (int i = 0; i < NUM_BUFFERS; i++) {
        CUDA_CHECK(cudaMallocHost(&h_input[i], chunk_size * sizeof(double)));
        CUDA_CHECK(cudaMallocHost(&h_output[i], chunk_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_buffers[i], chunk_size * sizeof(double)));
    }

    double tiempo_total = 0.0;
    double tiempo_cpu_total = 0.0;
    double tiempo_gpu_total = 0.0;
    
    // ========== BENCHMARK ==========
    
    for (int rep = 0; rep < REPETICIONES; rep++) {
        printf("Iteracion %d...\n", rep + 1);
        
        // CLAVE: Copia fresca cada iteración
        std::vector<double> datos = datos_originales;
        std::vector<double> salida(N);
        
        double t_inicio = omp_get_wtime();
        double tiempo_cpu_iter = 0.0;
        
        // Marcar inicio GPU
        CUDA_CHECK(cudaEventRecord(event_start, stream_h2d));
        
        // ========== PIPELINE LOOP ==========
        
        for (int chunk_id = 0; chunk_id < (int)num_chunks + NUM_BUFFERS - 1; chunk_id++) {
            int buf_idx = chunk_id % NUM_BUFFERS;
            
            // ETAPA 1: Preparación CPU + H→D
            if (chunk_id < (int)num_chunks) {
                size_t offset = chunk_id * chunk_size;
                size_t current_size = std::min(chunk_size, N - offset);
                
                // CPU: Copiar datos (trabajo de preparación)
                double t0_cpu = omp_get_wtime();
                
                // Usar OpenMP para copiar chunks en paralelo
                #pragma omp parallel for num_threads(num_hilos)
                for (int i = 0; i < (int)current_size; i++) {
                    h_input[buf_idx][i] = datos[offset + i];
                }
                
                double t1_cpu = omp_get_wtime();
                tiempo_cpu_iter += (t1_cpu - t0_cpu);
                
                // GPU: Transferencia H→D (asíncrona)
                CUDA_CHECK(cudaMemcpyAsync(d_buffers[buf_idx], h_input[buf_idx],
                                          current_size * sizeof(double),
                                          cudaMemcpyHostToDevice, stream_h2d));
                CUDA_CHECK(cudaEventRecord(event_h2d_done, stream_h2d));
            }
            
            // ETAPA 2: GPU Compute
            if (chunk_id >= 1 && chunk_id - 1 < (int)num_chunks) {
                int compute_buf = (chunk_id - 1) % NUM_BUFFERS;
                size_t offset = (chunk_id - 1) * chunk_size;
                size_t current_size = std::min(chunk_size, N - offset);
                
                CUDA_CHECK(cudaStreamWaitEvent(stream_compute, event_h2d_done, 0));
                
                thrust::device_ptr<double> dev_ptr(d_buffers[compute_buf]);
                thrust::sort(thrust::cuda::par.on(stream_compute), 
                           dev_ptr, dev_ptr + current_size);
                
                CUDA_CHECK(cudaEventRecord(event_compute_done, stream_compute));
            }
            
            // ETAPA 3: D→H + Postproceso CPU
            if (chunk_id >= 2 && chunk_id - 2 < (int)num_chunks) {
                int output_buf = (chunk_id - 2) % NUM_BUFFERS;
                size_t offset = (chunk_id - 2) * chunk_size;
                size_t current_size = std::min(chunk_size, N - offset);
                
                CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, event_compute_done, 0));
                
                CUDA_CHECK(cudaMemcpyAsync(h_output[output_buf], d_buffers[output_buf],
                                          current_size * sizeof(double),
                                          cudaMemcpyDeviceToHost, stream_d2h));
            }
        }
        
        // Marcar fin GPU
        CUDA_CHECK(cudaEventRecord(event_end, stream_d2h));
        
        // Sincronizar una sola vez al final
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // CPU: Copiar resultados finales con OpenMP
        double t0_cpu_final = omp_get_wtime();
        #pragma omp parallel for num_threads(num_hilos)
        for (int chunk_id = 0; chunk_id < (int)num_chunks; chunk_id++) {
            int output_buf = chunk_id % NUM_BUFFERS;
            size_t offset = chunk_id * chunk_size;
            size_t current_size = std::min(chunk_size, N - offset);
            
            for (size_t i = 0; i < current_size; i++) {
                salida[offset + i] = h_output[output_buf][i];
            }
        }
        double t1_cpu_final = omp_get_wtime();
        tiempo_cpu_iter += (t1_cpu_final - t0_cpu_final);
        
        double t_fin = omp_get_wtime();
        double duracion = t_fin - t_inicio;
        tiempo_total += duracion;
        tiempo_cpu_total += tiempo_cpu_iter;
        
        // Timing GPU
        float tiempo_gpu_ms;
        CUDA_CHECK(cudaEventElapsedTime(&tiempo_gpu_ms, event_start, event_end));
        double tiempo_gpu_iter = tiempo_gpu_ms / 1000.0;
        tiempo_gpu_total += tiempo_gpu_iter;
        
        printf("Iteracion %d completada en %.4f s\n", rep + 1, duracion);
        printf("  -> CPU (OpenMP): %.4f s, GPU (Thrust): %.4f s\n", 
               tiempo_cpu_iter, tiempo_gpu_iter);
        
        // Guardar resultado solo en última iteración
        if (rep == REPETICIONES - 1) {
            FILE* out = fopen("dataset_ordenado_pipeline_hibrido.csv", "w");
            for (size_t i = 0; i < N; i++) {
                fprintf(out, "%.6f\n", salida[i]);
            }
            fclose(out);
        }
    }

    // ========== PROMEDIOS ==========
    
    printf("\n=== Promedios ===\n");
    printf("Tiempo promedio total: %.4f s\n", tiempo_total / REPETICIONES);
    printf("Tiempo promedio CPU (OpenMP): %.4f s\n", tiempo_cpu_total / REPETICIONES);
    printf("Tiempo promedio GPU (Thrust): %.4f s\n", tiempo_gpu_total / REPETICIONES);
    
    // Calcular overlap effectiveness
    double avg_cpu = tiempo_cpu_total / REPETICIONES;
    double avg_gpu = tiempo_gpu_total / REPETICIONES;
    double avg_total = tiempo_total / REPETICIONES;
    double tiempo_secuencial = avg_cpu + avg_gpu;
    double speedup = tiempo_secuencial / avg_total;
    printf("Overlap effectiveness: %.2fx (teorico sin overlap: %.4f s)\n", 
           speedup, tiempo_secuencial);

    // ========== LIMPIEZA ==========
    
    for (int i = 0; i < NUM_BUFFERS; i++) {
        CUDA_CHECK(cudaFreeHost(h_input[i]));
        CUDA_CHECK(cudaFreeHost(h_output[i]));
        CUDA_CHECK(cudaFree(d_buffers[i]));
    }
    
    CUDA_CHECK(cudaEventDestroy(event_h2d_done));
    CUDA_CHECK(cudaEventDestroy(event_compute_done));
    CUDA_CHECK(cudaEventDestroy(event_start));
    CUDA_CHECK(cudaEventDestroy(event_end));
    
    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_d2h));

    printf("Dataset ordenado guardado: dataset_ordenado_pipeline_hibrido.csv\n");

    return 0;
}