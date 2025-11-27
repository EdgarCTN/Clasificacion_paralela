/*
 hybrid_cuda_openmp_stats.cu
 Híbrido CUDA + OpenMP para ordenar un CSV grande de doubles con medición CPU/GPU.
 Compilar: nvcc -O3 -Xcompiler /openmp hybrid_cuda_openmp_stats.cu -o hybrid_cuda_openmp_stats
 Ejecutar: ./hybrid_cuda_openmp_stats dataset.csv
*/

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <windows.h>

#define REPETICIONES 3
#define OMP_THREADS 4

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

void thrust_sort(double *d_data, size_t n) {
    thrust::device_ptr<double> dev_ptr(d_data);
    thrust::sort(dev_ptr, dev_ptr + n);
}

void merge(double *a, double *b, double *out, long n1, long n2) {
    long i=0, j=0, k=0;
    while (i<n1 && j<n2)
        out[k++] = (a[i] < b[j]) ? a[i++] : b[j++];
    while (i<n1) out[k++] = a[i++];
    while (j<n2) out[k++] = b[j++];
}

int main(int argc, char **argv) {
    if (argc < 2) { 
        printf("Uso: %s dataset.csv [num_hilos]\n", argv[0]); 
        return 1; 
    }

    const char* filename = argv[1];  // definir filename una sola vez

    int num_hilos = omp_get_max_threads(); // valor por defecto
    if (argc >= 3) {
        num_hilos = atoi(argv[2]);
        if (num_hilos < 1) num_hilos = 1;
    }
    omp_set_num_threads(num_hilos);
    printf("OpenMP activo con %d hilos\n", num_hilos);

    FILE *f = fopen(filename, "r");  // usar filename
    if (!f) { perror("Error al abrir dataset"); return 1; }

    std::vector<double> data;
    double val;
    while (fscanf(f, "%lf", &val) == 1) data.push_back(val);
    fclose(f);

    long N = data.size();
    long mitad = N / 2;
    printf("Datos cargados: %ld\n", N);
    printf("OpenMP activo con %d hilos\n", omp_get_max_threads());

    double *host_data = data.data();
    double *cpu_half = host_data;
    double *gpu_half = host_data + mitad;

    double tiempo_total = 0.0;
    double tiempo_cpu_total = 0.0;
    double tiempo_gpu_total = 0.0;

    for (int r = 0; r < REPETICIONES; r++) {
        printf("\nIteracion %d...\n", r+1);
        double t0_iter = tiempo_actual();

        // === CPU paralela (OpenMP) ===
        double t0_cpu = tiempo_actual();
        #pragma omp parallel num_threads(OMP_THREADS)
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            long chunk = mitad / nt;
            long ini = tid * chunk;
            long fin = (tid == nt-1) ? mitad : ini + chunk;
            std::sort(cpu_half + ini, cpu_half + fin);
        }
        double t1_cpu = tiempo_actual();
        double tiempo_cpu = t1_cpu - t0_cpu;
        tiempo_cpu_total += tiempo_cpu;

        // === GPU (Thrust) ===
	double t0_gpu_total = tiempo_actual();

	// Copia H->D
	double t0_copy = tiempo_actual();
	double *d_data;
	size_t gpu_size = (N - mitad) * sizeof(double);
	CUDA_CHECK(cudaMalloc(&d_data, gpu_size));
	CUDA_CHECK(cudaMemcpy(d_data, gpu_half, gpu_size, cudaMemcpyHostToDevice));
	double t1_copy = tiempo_actual();
	double tiempo_copy = t1_copy - t0_copy;

	// Ejecución Thrust
	double t0_sort = tiempo_actual();
	thrust_sort(d_data, N - mitad);
	CUDA_CHECK(cudaDeviceSynchronize());
	double t1_sort = tiempo_actual();
	double tiempo_sort = t1_sort - t0_sort;

	// Copia D->H
	double t0_copy_back = tiempo_actual();
	CUDA_CHECK(cudaMemcpy(gpu_half, d_data, gpu_size, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_data));
	double t1_copy_back = tiempo_actual();
	tiempo_copy += t1_copy_back - t0_copy_back;

	double t1_gpu_total = tiempo_actual();
	double tiempo_gpu_total_iter = t1_gpu_total - t0_gpu_total;
	tiempo_gpu_total += tiempo_gpu_total_iter;

        // === Merge final ===
        std::vector<double> ordenado(N);
        merge(cpu_half, gpu_half, ordenado.data(), mitad, N - mitad);

        double t1_iter = tiempo_actual();
        double dur = t1_iter - t0_iter;
        tiempo_total += dur;

	printf("Iteracion %d completada en %.4f s (CPU: %.4f s, GPU: %.4f s)\n", r+1, dur, tiempo_cpu, tiempo_gpu_total_iter);

	printf("  -> GPU detalle: copia H<->D %.4f s, sort %.4f s\n", tiempo_copy, tiempo_sort);
    }

    printf("\n=== Promedios ===\n");
    printf("Tiempo promedio total: %.4f s\n", tiempo_total / REPETICIONES);
    printf("Tiempo promedio CPU (OpenMP): %.4f s\n", tiempo_cpu_total / REPETICIONES);
    printf("Tiempo promedio GPU (Thrust): %.4f s\n", tiempo_gpu_total / REPETICIONES);

    // Guardar resultado final
    FILE *out = fopen("dataset_ordenado_hibrido.csv", "w");
    for (auto &x : data) fprintf(out, "%.0f\n", x);
    fclose(out);
    printf("Dataset ordenado guardado: dataset_ordenado_hibrido.csv\n");

    return 0;
}
