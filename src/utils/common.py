import numpy as np
import pandas as pd

def calcular_ecm(beta_0, beta_1, x, y):
    """
    Calcula el Error Cuadrático Medio (ECM) para un modelo de regresión lineal simple.

    El Error Cuadrático Medio (ECM) mide la magnitud promedio de los errores en las
    predicciones de un modelo. Se define como el promedio de los cuadrados de las
    diferencias entre los valores reales y las predicciones realizadas por el modelo.

    Parámetros
    ----------
    beta_0 : float
        Término independiente del modelo (intercepto).
    beta_1 : float
        Coeficiente asociado a la variable independiente x (pendiente).
    x : float o array_like
        Valores de entrada para el modelo. Puede ser un escalar, lista o arreglo numpy.
    y : float o array_like
        Valores reales observados. Debe tener la misma forma que x.

    Retorna
    -------
    float
        Valor del Error Cuadrático Medio calculado entre las predicciones del modelo
        y los valores reales proporcionados.

    Fórmula
    -------
    Para un modelo lineal simple:
        y_pred = beta_1 * x + beta_0

    El ECM se define como:
        ECM = (1/n) * Σ (y_real - y_pred)^2

    Ejemplos
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([5, 7, 10, 15])
    >>> calcular_ecm(2, 3, x, y)
    0.75
    """

    predicciones = beta_1 * x + beta_0
    return np.mean((y - predicciones) ** 2)

def generar_tabla_resumen(df_total):
    """
    Genera una tabla resumen con estadísticas descriptivas básicas para
    evaluar el desempeño de los algoritmos de optimización.

    La tabla contiene, para cada algoritmo:
        - Promedio, mediana y desviación estándar del ECM final.
        - Promedio, mediana y desviación estándar del número de iteraciones.
        - Promedio, mediana y desviación estándar del tiempo de ejecución.

    Parámetros
    ----------
    df_total : pandas.DataFrame
        DataFrame consolidado con los resultados de las ejecuciones.
        Debe contener las columnas:
        ['Algoritmo', 'ECM_Final', 'Iteraciones', 'Tiempo_seg']

    Retorna
    -------
    pandas.DataFrame
        Tabla resumen con estadísticas por algoritmo.
        Si el DataFrame está vacío, retorna un DataFrame vacío.
    """
    
    if df_total.empty:
        return pd.DataFrame()

    # Columnas esperadas
    columnas_metricas = ['ECM_Final', 'Iteraciones', 'Tiempo_seg']

    # Filtrar solo columnas válidas (por seguridad)
    columnas_presentes = [c for c in columnas_metricas if c in df_total.columns]
    if not columnas_presentes:
        raise ValueError("El DataFrame no contiene columnas válidas para generar el resumen.")

    # Calcular estadísticas agrupadas por algoritmo
    resumen = (
        df_total
        .groupby('Algoritmo')[columnas_presentes]
        .agg(['mean', 'median', 'std'])
    )

    # Crear nombres limpios para las columnas finales
    nombres_finales = []
    for col in columnas_presentes:
        nombres_finales.extend([
            f"{col} Promedio",
            f"{col} Mediana",
            f"{col} Desv."
        ])

    resumen.columns = nombres_finales

    return resumen

# Código de prueba
def main():
    beta_0 = 2
    beta_1 = 3

    x = np.array([1, 2, 3, 4])
    y = np.array([5, 7, 10, 15])

    resultado = calcular_ecm(beta_0, beta_1, x, y)

    print("ECM calculado:", resultado)

if __name__ == "__main__":
    main()

