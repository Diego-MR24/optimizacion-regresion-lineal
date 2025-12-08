import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import ast

def parse_historial(hist_str):
    """
    Convierte una cadena de lista limpia (ej: "[0.1, 0.2]") a una lista de Python.
    """
    try:
        if isinstance(hist_str, str):
            return ast.literal_eval(hist_str)
        return hist_str
    except (ValueError, SyntaxError):
        return []

def plot_convergencia(df_resultados, output_dir):
    """
    Gráfica 1: Evolución del Error (ECM) a través de las iteraciones.
    Muestra la curva de convergencia de la MEJOR ejecución de cada algoritmo.
    """
    plt.figure(figsize=(10, 6))
    
    algoritmos = df_resultados['Algoritmo'].unique()
    # Diccionario de colores para consistencia en todas las gráficas
    colores = {'amplitud': '#1f77b4', 'recocido': '#ff7f0e', 'genetico': '#2ca02c'}
    
    for algo in algoritmos:
        subset = df_resultados[df_resultados['Algoritmo'] == algo]
        if subset.empty: continue

        # Seleccionar la mejor ejecución (menor error final)
        idx_mejor = subset['ECM_Final'].idxmin()
        mejor_run = subset.loc[idx_mejor]
        
        historia = parse_historial(mejor_run['Historial'])
        
        if historia:
            color = colores.get(algo, 'black')
            plt.plot(historia, label=f"{algo} (Mejor Run)", linewidth=2, color=color, alpha=0.8)

    plt.title('Evolución del Error por Iteración')
    plt.xlabel('Iteraciones')
    plt.ylabel('Error Cuadrático Medio (ECM)')
    plt.yscale('log') # Escala logarítmica para visualizar mejor la convergencia fina
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    output_path = os.path.join(output_dir, 'grafica_convergencia.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_solucion(df_datos, df_resultados, output_dir):
    """
    Gráfica 2: Visualización de la Solución Final.
    Superpone la recta de regresión promedio sobre los datos originales (peces).
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Puntos de Datos Reales
    plt.scatter(df_datos['Length1_norm'], df_datos['Weight_norm'], 
                color='gray', alpha=0.5, label='Datos Reales')
    
    # 2. Rectas de Regresión (Promedio)
    x_vals = np.linspace(0, 1, 100)
    algoritmos = df_resultados['Algoritmo'].unique()
    colores = {'amplitud': '#1f77b4', 'recocido': '#ff7f0e', 'genetico': '#2ca02c'}
    
    for algo in algoritmos:
        subset = df_resultados[df_resultados['Algoritmo'] == algo]
        if subset.empty: continue

        # Calcular promedios de beta_0 y beta_1
        b0_mean = subset['Beta_0'].mean()
        b1_mean = subset['Beta_1'].mean()
        
        # Ecuación de la recta: y = b1*x + b0
        y_vals = b1_mean * x_vals + b0_mean
        
        color = colores.get(algo, 'red')
        plt.plot(x_vals, y_vals, label=f"{algo} (Promedio)", linewidth=2.5, color=color)

    plt.title('Solución Final: Regresión Lineal Ajustada')
    plt.xlabel('Largo Normalizado')
    plt.ylabel('Peso Normalizado')
    plt.legend()
    plt.grid(True, ls="--", alpha=0.3)
    
    output_path = os.path.join(output_dir, 'grafica_solucion.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_boxplot(df_resultados, output_dir):
    """
    Gráfica 3: Boxplot Comparativo.
    Muestra la distribución y estabilidad del ECM final en las 30 ejecuciones.
    """
    plt.figure(figsize=(10, 6))
    
    algoritmos = df_resultados['Algoritmo'].unique()
    datos_plot = []
    labels = []
    
    for algo in algoritmos:
        subset = df_resultados[df_resultados['Algoritmo'] == algo]
        if not subset.empty:
            datos_plot.append(subset['ECM_Final'].values)
            labels.append(algo)
        
    # Crear Boxplot
    bplot = plt.boxplot(datos_plot, labels=labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5))
    
    # Colorear las cajas
    colores = ['#aec7e8', '#ffbb78', '#98df8a'] # Colores pastel para las cajas
    for patch, color in zip(bplot['boxes'], colores):
        patch.set_facecolor(color)

    plt.title('Comparación Estadística del Error (30 Ejecuciones)')
    plt.ylabel('ECM Final')
    plt.grid(True, axis='y', ls="--", alpha=0.3)
    
    output_path = os.path.join(output_dir, 'grafica_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()