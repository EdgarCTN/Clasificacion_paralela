# Clasificacion secuencial

## 1. Generar un dataset grande
seq 1 1000000 | shuf > dataset.csv

## 2. Compilar
gcc secuencial.c -o secuencial -O2

## 3. Ejecutar
./secuencial

# Clasificacion paralela - MPI
## 1. Generar un dataset grande
seq 1 1000000 | shuf > dataset.csv

## 2. Instalar MPI
sudo apt update  
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev

## 3. Compilar
mpicc mpi.c -o mpi -O2

## 4. Ejecutar
mpirun -np 2 ./mpi
