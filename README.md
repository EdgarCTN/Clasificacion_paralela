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

## 2. Ejecutar
mpirun -H master,client1,client2 -np 3 ./mpi

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

# Clasificacion paralela  Hibrida- OpenMP y MPI
## 1. Compilar
mpicc -fopenmp -O2 mpi_openmp_hibrido.c -o mpi_openmp_hibrido -lm

## 2. Definir hilos
OMP_NUM_THREADS=4

## 3. Ejecutar
mpirun -H master,client1,client2 -np 3 ./mpi_openmp_hibrido


# Interfaz

1 ) sudo apt install python3-tk python3-matplotlib -y
### Opciones (Este programa se hizo considerando que uses WSL y que solo tengas un nucleo)
A. Crear el archivo:
nano hostfile   

B. Dentro del archivo, agrega :
localhost slots=3   




2) Ejecutar el programa MPI
mpirun -np 3 --hostfile hostfile ./mpi

3) Ejecutar la interfaz gráfica
nano gui.py
python3 gui.py

