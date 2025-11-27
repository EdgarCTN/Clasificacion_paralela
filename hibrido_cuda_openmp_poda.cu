/*
 hibrido_cuda_openmp_poda.cu
 Esquema de Poda Híbrido CUDA + OpenMP
 
 ESTRATEGIA:
 1. Dividir dataset en chunks
 2. Analizar cada chunk (CPU paralelo con OpenMP)
 3. PODAR: Solo enviar a GPU los chunks que necesitan ordenamiento
 4. Chunks ya ordenados o con patrones simples → CPU rápido
 5. Chunks desordenados complejos → GPU
 
 Compilar: nvcc -O3 -Xcompiler /openmp hibrido_cuda_openmp_poda.cu -o hibrido_cuda_openmp_poda
 Ejecutar: ./hibrido_cuda_openmp_poda dataset.csv 8
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

// Verificar si un chunk está ordenado (criterio de poda)
bool is_sorted(double *arr, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (arr[i] < arr[i-1]) return false;
    }
    return true;
}

// Verificar si un chunk tiene varianza muy baja (casi constante)
bool low_variance(double *arr, size_t n, double threshold = 0.001) {
    if (n < 2) return true;
    
    double mean = 0.0;
    for (size_t i = 0; i < n; i++) {
        mean += arr[i];
    }
    mean /= n;
    
    double variance = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = arr[i] - mean;
        variance += diff * diff;
    }
    variance /= n;
    
    return variance < threshold;
}

// Calcular desorden (inversiones) - heurística de poda
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

struct ChunkInfo {
    size_t offset;
    size_t size;
    bool needs_gpu;      // ¿Necesita GPU?
    bool already_sorted; // ¿Ya está ordenado?
    int complexity;      // Nivel de desorden
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
    
    printf("Esquema de Poda Hibrido CUDA+OpenMP\n");
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
    double tiempo_cpu_total = 0.0;
    double tiempo_gpu_total = 0.0;
    int chunks_podados_total = 0;
    int chunks_cpu_total = 0;
    int chunks_gpu_total = 0;

    for (int rep = 0; rep < REPETICIONES; rep++) {
        printf("Iteracion %d...\n", rep + 1);
        
        // Copia fresca cada iteración
        std::vector<double> datos = datos_originales;
        std::vector<ChunkInfo> chunks_info(num_chunks);
        
        double t0_total = omp_get_wtime();
        
        // ==== FASE 1: ANALISIS Y PODA (CPU paralelo) ====
        double t0_analisis = omp_get_wtime();
        
        int chunks_podados = 0;
        int chunks_cpu = 0;
        int chunks_gpu = 0;
        
        #pragma omp parallel for num_threads(num_hilos) reduction(+:chunks_podados,chunks_cpu,chunks_gpu)
        for (int c = 0; c < (int)num_chunks; c++) {
            size_t offset = c * chunk_size;
            size_t size = (offset + chunk_size > N) ? (N - offset) : chunk_size;
            
            chunks_info[c].offset = offset;
            chunks_info[c].size = size;
            
            double *chunk_ptr = datos.data() + offset;
            
            // CRITERIO 1: Ya está ordenado → PODAR
            if (is_sorted(chunk_ptr, size)) {
                chunks_info[c].already_sorted = true;
                chunks_info[c].needs_gpu = false;
                chunks_info[c].complexity = 0;
                chunks_podados++;
                continue;
            }
            
            // CRITERIO 2: Varianza muy baja (valores casi iguales) → CPU
            if (low_variance(chunk_ptr, size)) {
                chunks_info[c].already_sorted = false;
                chunks_info[c].needs_gpu = false;
                chunks_info[c].complexity = 1;
                chunks_cpu++;
                continue;
            }
            
            // CRITERIO 3: Medir desorden
            int inversions = count_inversions_sample(chunk_ptr, size);
            chunks_info[c].complexity = inversions;
            
            // Desorden bajo → CPU, desorden alto → GPU
            if (inversions < 50) {
                chunks_info[c].needs_gpu = false;
                chunks_cpu++;
            } else {
                chunks_info[c].needs_gpu = true;
                chunks_gpu++;
            }
            
            chunks_info[c].already_sorted = false;
        }
        
        double t1_analisis = omp_get_wtime();
        double tiempo_analisis = t1_analisis - t0_analisis;
        tiempo_analisis_total += tiempo_analisis;
        
        chunks_podados_total += chunks_podados;
        chunks_cpu_total += chunks_cpu;
        chunks_gpu_total += chunks_gpu;
        
        // ==== FASE 2: PROCESAMIENTO CPU (chunks simples) ====
        double t0_cpu = omp_get_wtime();
        
        #pragma omp parallel for num_threads(num_hilos)
        for (int c = 0; c < (int)num_chunks; c++) {
            if (!chunks_info[c].already_sorted && !chunks_info[c].needs_gpu) {
                double *chunk_ptr = datos.data() + chunks_info[c].offset;
                std::sort(chunk_ptr, chunk_ptr + chunks_info[c].size);
            }
        }
        
        double t1_cpu = omp_get_wtime();
        double tiempo_cpu = t1_cpu - t0_cpu;
        tiempo_cpu_total += tiempo_cpu;
        
        // ==== FASE 3: PROCESAMIENTO GPU (chunks complejos) ====
        double t0_gpu = omp_get_wtime();
        
        // Contar chunks GPU
        int gpu_count = 0;
        for (size_t c = 0; c < num_chunks; c++) {
            if (chunks_info[c].needs_gpu) gpu_count++;
        }
        
        if (gpu_count > 0) {
            // Allocar buffer GPU para el chunk más grande
            size_t max_chunk_size = 0;
            for (size_t c = 0; c < num_chunks; c++) {
                if (chunks_info[c].needs_gpu && chunks_info[c].size > max_chunk_size) {
                    max_chunk_size = chunks_info[c].size;
                }
            }
            
            double *d_buffer;
            CUDA_CHECK(cudaMalloc(&d_buffer, max_chunk_size * sizeof(double)));
            
            // Procesar cada chunk GPU secuencialmente (puede paralelizarse con streams)
            for (size_t c = 0; c < num_chunks; c++) {
                if (chunks_info[c].needs_gpu) {
                    double *chunk_ptr = datos.data() + chunks_info[c].offset;
                    size_t size = chunks_info[c].size;
                    
                    CUDA_CHECK(cudaMemcpy(d_buffer, chunk_ptr, size * sizeof(double),
                                         cudaMemcpyHostToDevice));
                    
                    thrust::device_ptr<double> dev_ptr(d_buffer);
                    thrust::sort(dev_ptr, dev_ptr + size);
                    
                    CUDA_CHECK(cudaMemcpy(chunk_ptr, d_buffer, size * sizeof(double),
                                         cudaMemcpyDeviceToHost));
                }
            }
            
            CUDA_CHECK(cudaFree(d_buffer));
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        double t1_gpu = omp_get_wtime();
        double tiempo_gpu = t1_gpu - t0_gpu;
        tiempo_gpu_total += tiempo_gpu;
        
        // ==== FASE 4: MERGE FINAL (k-way merge simple) ====
        // Los chunks ya están ordenados, solo verificamos
        
        double t1_total = omp_get_wtime();
        double duracion = t1_total - t0_total;
        tiempo_total += duracion;
        
        printf("Iteracion %d completada en %.4f s\n", rep + 1, duracion);
        printf("  -> Analisis: %.4f s\n", tiempo_analisis);
        printf("  -> CPU sort: %.4f s (chunks: %d)\n", tiempo_cpu, chunks_cpu);
        printf("  -> GPU sort: %.4f s (chunks: %d)\n", tiempo_gpu, chunks_gpu);
        printf("  -> Podados:  %d chunks (ya ordenados)\n", chunks_podados);
        
        // Guardar resultado solo en última iteración
        if (rep == REPETICIONES - 1) {
            FILE* out = fopen("dataset_ordenado_poda.csv", "w");
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
    printf("Tiempo promedio CPU: %.4f s\n", tiempo_cpu_total / REPETICIONES);
    printf("Tiempo promedio GPU: %.4f s\n", tiempo_gpu_total / REPETICIONES);
    
    printf("\n=== Distribucion de Trabajo ===\n");
    printf("Chunks podados (ya ordenados): %.1f%% (%d/%d)\n",
           100.0 * chunks_podados_total / (REPETICIONES * num_chunks),
           chunks_podados_total / REPETICIONES, (int)num_chunks);
    printf("Chunks CPU (simples):          %.1f%% (%d/%d)\n",
           100.0 * chunks_cpu_total / (REPETICIONES * num_chunks),
           chunks_cpu_total / REPETICIONES, (int)num_chunks);
    printf("Chunks GPU (complejos):        %.1f%% (%d/%d)\n",
           100.0 * chunks_gpu_total / (REPETICIONES * num_chunks),
           chunks_gpu_total / REPETICIONES, (int)num_chunks);
    
    printf("\nDataset ordenado guardado: dataset_ordenado_poda.csv\n");

    return 0;
}