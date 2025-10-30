# Generar un dataset grande
## Linux:
```bash
seq 1 1000000 | shuf > dataset.csv
```

## Windows (PowerShell):
```powershell
1..1000000 | Get-Random -Count 1000000 | Out-File -Encoding ASCII dataset.csv
```

# Clasificacion secuencial

## 1. Compilar
gcc secuencial.c -o secuencial -O2

## 2. Ejecutar
./secuencial

# Clasificacion paralela - MPI (Maestro y esclavo)
## 1. Compilar
mpicc mpi.c -o mpi

mpirun -H master,client1,client2 -np 3 ./mpi
## 2. Ejecutar
mpicc mpi.c -o mpi -O2

# Clasificacion paralela - OpenMP (Divide y venceras)
## 1. Compilar
gcc openmp.c -o openmp -fopenmp -O2 -lm

## 2. Ejecutar
OMP_NUM_THREADS=4 ./openmp

# Clasificacion paralela - CUDA (GPU)
## 1. Requisitos previos
- GPU NVIDIA compatible (GTX/RTX series)
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Drivers NVIDIA actualizados

## 2. Compilar
**Linux:**
```bash
nvcc cuda_sort.cu -o cuda_sort -O3
```
**Windows:**
```bash
nvcc cuda_sort.cu -o cuda_sort.exe -O3 -std=c++17
```

## 3. Ejecutar
**Linux:** `./cuda_sort`  
**Windows:** `cuda_sort.exe`

## 4. Salida
- `dataset_ordenado.csv` - Dataset ordenado completo
- `resultados_cuda.csv` - Estadísticas de rendimiento
