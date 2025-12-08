import os
import pandas as pd
import numpy as np

from src import (
    amplitud,
    recocido,
    genetico,
    experiment_runner,
    generar_tabla_resumen,
    plot_convergencia,
    plot_solucion,
    plot_boxplot,
)

print("=== PROYECTO DE OPTIMIZACIÓN: 30 EJECUCIONES ===")

# --- 1. Carga de Datos ---
base_dir = os.getcwd()
csv_path = os.path.join(base_dir, 'data', 'processed', 'clean_fish_data.csv')
results_dir = os.path.join(base_dir, 'reports', 'results')
figures_dir = os.path.join(base_dir, 'reports', 'figures')

df = pd.read_csv(csv_path)
X = df['Length1_norm'].values
Y = df['Weight_norm'].values

print(f"Datos cargados correctamente: {len(X)} registros normalizados.")

# --- 2. Configuración de Parámetros ---
params_bfs = {
    'inicio_b0': 0.0,
    'inicio_b1': 0.0,
    'paso': 0.05,
    'max_iter': 2000
}

params_sa = {
    'inicio_b0': 0.0,
    'inicio_b1': 0.0,
    't_inicial': 100.0,
    't_final': 0.001,
    'alpha': 0.95,
    'paso': 0.2
}

params_ga = {
    'tam_poblacion': 50,
    'generaciones': 100,
    'prob_mutacion': 0.1,
    'rango_mutacion': 0.2,
    'k_torneo': 3,
    'rango_inicio': (-1, 1)
}

# --- 3. Ejecución de Experimentos (Decorados) ---
# El decorador ejecuta cada algoritmo 30 veces y registra desempeño

print("\nEjecutando Búsqueda en Amplitud (30 corridas)...")
df_bfs = experiment_runner(30)(amplitud)(X, Y, params_bfs)

print("Ejecutando Recocido Simulado (30 corridas)...")
df_sa = experiment_runner(30)(recocido)(X, Y, params_sa)

print("Ejecutando Algoritmo Genético (30 corridas)...")
df_ga = experiment_runner(30)(genetico)(X, Y, params_ga)

# --- 4. Consolidación y Reporte ---
df_total = pd.concat([df_bfs, df_sa, df_ga], ignore_index=True)

# Conversión de datos numpy a float nativo para evitar formato sucio en CSV
df_total['Historial'] = df_total['Historial'].apply(lambda lista: [float(x) for x in lista])

# Guardar resultados
results_path = os.path.join(results_dir, 'resultados.csv')
df_total.to_csv(results_path, index=False)
print(f"\nResultados guardados en: {results_path}")

# Generar tabla resumen
tabla_resumen = generar_tabla_resumen(df_total)

print("\n" + "=" * 190)
print("TABLA RESUMEN DE DESEMPEÑO (30 ejecuciones por algoritmo)")
print("=" * 190)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.6f}'.format)

print(tabla_resumen)
print("=" * 190)

# --- 5. Generación de Gráficas ---
print("\nGenerando gráficas comparativas...")

try:
    plot_convergencia(df_total, figures_dir)
    plot_solucion(df, df_total, figures_dir)
    plot_boxplot(df_total, figures_dir)
    print(f"Gráficas generadas exitosamente en: {figures_dir}")
except Exception as e:
    print(f"Error durante la generación de gráficas: {e}")