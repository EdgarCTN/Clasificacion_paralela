# Clasificacion secuencial

## 1. Generar un dataset grande
seq 1 1000000 | shuf > dataset.csv

## 2. Compilar
gcc secuencial.c -o secuencial -O2

## 3. Ejecutar
./secuencial
