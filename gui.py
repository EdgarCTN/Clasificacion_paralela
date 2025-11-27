import tkinter as tk
from tkinter import scrolledtext
import subprocess
import os
import re
import matplotlib.pyplot as plt


# ======================================================
#            VARIABLES PARA GUARDAR RESULTADOS
# ======================================================
tiempo_secuencial = None
tiempo_openmp = None
tiempo_mpi = None


# ======================================================
#   EXTRACCIÓN DEL TIEMPO PROMEDIO DESDE LA SALIDA
# ======================================================
def extraer_tiempo(texto):
    """
    Extrae un número decimal desde líneas como:
    'Tiempo promedio (QuickSort ...): 0.12345 segundos'
    """
    patron = r"Tiempo promedio.*?:\s*([0-9]+\.[0-9]+)"
    match = re.search(patron, texto)
    if match:
        return float(match.group(1))
    return None


# ======================================================
#      FUNCIÓN GENÉRICA PARA LA EJECUCIÓN DE PROGRAMAS
# ======================================================
def ejecutar_programa(comando, variables_entorno=None):
    salida_text.delete(1.0, tk.END)

    try:
        entorno = os.environ.copy()
        if variables_entorno:
            entorno.update(variables_entorno)

        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            shell=False,
            env=entorno
        )

        salida_text.insert(tk.END, resultado.stdout)

        if resultado.stderr:
            salida_text.insert(tk.END, "\n[ERRORES DETECTADOS]\n" + resultado.stderr)

        return resultado.stdout

    except Exception as ex:
        salida_text.insert(tk.END,
            f"[ERROR] No fue posible ejecutar el programa.\n{ex}\n"
        )
        return ""


# ======================================================
#                 GENERACIÓN DE DATASET
# ======================================================
def generar_dataset():
    global tiempo_secuencial, tiempo_openmp, tiempo_mpi

    cantidad = entry_dataset.get().strip()

    if not cantidad.isdigit() or int(cantidad) < 1:
        salida_text.insert(tk.END, "❌ Error: Debe ingresar un número entero positivo.\n")
        return

    comando = f"seq 1 {cantidad} | shuf > dataset.csv"

    try:
        subprocess.run(comando, shell=True, executable="/bin/bash")
        salida_text.insert(tk.END,
            f"✔ Dataset generado correctamente ({cantidad} registros).\n"
        )

        # Reiniciar tiempos para obligar a reejecutar todo
        tiempo_secuencial = None
        tiempo_openmp = None
        tiempo_mpi = None

        salida_text.insert(
            tk.END,
            "⚠ Se ha generado un nuevo dataset.\n"
            "   Debe ejecutar nuevamente Secuencial, OpenMP y MPI antes de generar el gráfico.\n"
        )

    except Exception as ex:
        salida_text.insert(tk.END,
            f"❌ Error durante la generación del dataset.\n{ex}\n"
        )


# ======================================================
#                 ACCIONES DE EJECUCIÓN
# ======================================================
def ejecutar_secuencial():
    global tiempo_secuencial
    salida = ejecutar_programa(["./secuencial"])
    tiempo_secuencial = extraer_tiempo(salida)
    salida_text.insert(tk.END, f"\n✔ Tiempo capturado: {tiempo_secuencial}s\n")


def ejecutar_openmp():
    global tiempo_openmp

    hilos = entry_hilos.get().strip()
    if not hilos.isdigit():
        salida_text.insert(tk.END, "❌ Error: Ingrese un valor numérico válido.\n")
        return

    salida = ejecutar_programa(
        ["./openmp"],
        variables_entorno={"OMP_NUM_THREADS": hilos}
    )
    tiempo_openmp = extraer_tiempo(salida)
    salida_text.insert(tk.END, f"\n✔ Tiempo capturado: {tiempo_openmp}s\n")


def ejecutar_mpi():
    global tiempo_mpi

    procesos = entry_procesos.get().strip()
    if not procesos.isdigit():
        salida_text.insert(tk.END, "❌ Error: Número de procesos inválido.\n")
        return

    salida = ejecutar_programa([
        "mpirun", "-np", procesos, "--hostfile", "hostfile", "./mpi"
    ])
    tiempo_mpi = extraer_tiempo(salida)
    salida_text.insert(tk.END, f"\n✔ Tiempo capturado: {tiempo_mpi}s\n")


# ======================================================
#                 GENERAR GRÁFICO
# ======================================================
def generar_grafico():
    global tiempo_secuencial, tiempo_openmp, tiempo_mpi

    resultados = {
        "Secuencial": tiempo_secuencial,
        "OpenMP": tiempo_openmp,
        "MPI": tiempo_mpi
    }

    faltantes = [nombre for nombre, t in resultados.items() if t is None]
    if faltantes:
        salida_text.insert(
            tk.END,
            f"\n⚠ Debes ejecutar antes: {', '.join(faltantes)}\n"
        )
        return

    dataset_tamano = entry_dataset.get().strip()

    nombres = list(resultados.keys())
    tiempos = list(resultados.values())

    plt.figure(figsize=(7, 5))
    plt.bar(nombres, tiempos)
    plt.ylabel("Tiempo (segundos)")
    plt.title(f"Comparación de Rendimiento\nDataset: {dataset_tamano} elementos")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


# ======================================================
#              CONFIGURACIÓN DE LA INTERFAZ GRÁFICA
# ======================================================
ventana = tk.Tk()
ventana.title("📊 Comparador de Rendimiento — Secuencial, OpenMP y MPI")
ventana.geometry("950x750")
ventana.configure(bg="#F2F2F2")

# -------- TÍTULO --------
titulo = tk.Label(
    ventana,
    text="Comparador de Rendimiento\nSecuencial • OpenMP • MPI",
    font=("Arial", 20, "bold"),
    bg="#F2F2F2"
)
titulo.pack(pady=12)

# -------- PANEL SUPERIOR --------
panel_config = tk.Frame(ventana, bg="#F2F2F2")
panel_config.pack(pady=10)

# Dataset
tk.Label(panel_config, text="Tamaño Dataset:", font=("Arial", 12), bg="#F2F2F2").grid(row=0, column=0)
entry_dataset = tk.Entry(panel_config, width=10, font=("Arial", 12))
entry_dataset.insert(0, "1000000")
entry_dataset.grid(row=0, column=1, padx=10)

btn_dataset = tk.Button(
    panel_config, text="Generar Dataset",
    font=("Arial", 11), width=18,
    command=generar_dataset,
    bg="#DDEBF7"
)
btn_dataset.grid(row=0, column=2, padx=15)

# OpenMP
tk.Label(panel_config, text="Hilos OpenMP:", bg="#F2F2F2", font=("Arial", 12)).grid(row=1, column=0, pady=10)
entry_hilos = tk.Entry(panel_config, width=6, font=("Arial", 12))
entry_hilos.insert(0, "4")
entry_hilos.grid(row=1, column=1)

# MPI
tk.Label(panel_config, text="Procesos MPI:", bg="#F2F2F2", font=("Arial", 12)).grid(row=1, column=2)
entry_procesos = tk.Entry(panel_config, width=6, font=("Arial", 12))
entry_procesos.insert(0, "3")
entry_procesos.grid(row=1, column=3)

# -------- BOTONES DE EJECUCIÓN --------
panel_botones = tk.Frame(ventana, bg="#F2F2F2")
panel_botones.pack(pady=10)

btn_secuencial = tk.Button(
    panel_botones, text="Ejecutar Secuencial",
    width=22, font=("Arial", 12, "bold"),
    bg="#CFE2F3", command=ejecutar_secuencial
)
btn_secuencial.grid(row=0, column=0, padx=10)

btn_openmp = tk.Button(
    panel_botones, text="Ejecutar OpenMP",
    width=22, font=("Arial", 12, "bold"),
    bg="#D9EAD3", command=ejecutar_openmp
)
btn_openmp.grid(row=0, column=1, padx=10)

btn_mpi = tk.Button(
    panel_botones, text="Ejecutar MPI",
    width=22, font=("Arial", 12, "bold"),
    bg="#F4CCCC", command=ejecutar_mpi
)
btn_mpi.grid(row=0, column=2, padx=10)

# -------- BOTÓN PARA GRAFICAR --------
btn_grafico = tk.Button(
    ventana, text=" Generar Gráfico Comparativo",
    font=("Arial", 14, "bold"),
    bg="#FFF2CC", width=30,
    command=generar_grafico
)
btn_grafico.pack(pady=10)

# -------- SALIDA --------
salida_text = scrolledtext.ScrolledText(
    ventana, width=115, height=25,
    font=("Consolas", 10)
)
salida_text.pack(pady=15)

ventana.mainloop()
