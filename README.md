# Generar un dataset grande
seq 1 1000000 | shuf > dataset.csv


# Clasificacion secuencial

## 1. Compilar
gcc secuencial.c -o secuencial -O2

## 2. Ejecutar
./secuencial

# Clasificacion paralela - MPI
## 1. Instalar MPI
sudo apt update  
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev

## 2. Compilar
mpicc mpi.c -o mpi -O2

## 3. Ejecutar
mpirun -np 2 ./mpi

# Clasificacion paralela - OpenMP (Divide y venceras)
## 1. Compilar
gcc openmp.c -o openmp -fopenmp -O2 -lm

## 2. Ejecutar
OMP_NUM_THREADS=4 ./openmp
