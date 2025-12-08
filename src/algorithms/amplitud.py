import sys
import os
import pandas as pd
import numpy as np
from collections import deque

# Agregar la raíz del proyecto al path (solo para ejecución directa de este archivo)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.utils.common import calcular_ecm


def generar_vecinos_grid(b0, b1, paso):
    """
    Genera los cuatro vecinos inmediatos (arriba, abajo, izquierda y derecha)
    en una cuadrícula definida por los parámetros b0, b1 y un tamaño de paso.
    """
    return [
        (b0 + paso, b1),
        (b0 - paso, b1),
        (b0, b1 + paso),
        (b0, b1 - paso)
    ]


def amplitud(x, y, params):
    """
    Algoritmo de búsqueda en amplitud (BFS) aplicado a la estimación de parámetros
    de un modelo de regresión lineal simple mediante exploración de una cuadrícula.

    Parámetros
    ----------
    x : array_like
        Valores de la variable independiente.
    y : array_like
        Valores de la variable objetivo.
    params : dict
        Diccionario de parámetros del algoritmo:
        - inicio_b0 : valor inicial del intercepto.
        - inicio_b1 : valor inicial de la pendiente.
        - paso : tamaño del desplazamiento en cada expansión.
        - max_iter : número máximo de iteraciones permitidas.

    Retorna
    -------
    list
        Lista con los valores óptimos encontrados para b0 y b1.
    float
        Valor del ECM en la mejor solución.
    int
        Número total de iteraciones realizadas.
    list
        Historial de valores de ECM durante la búsqueda.
    """

    inicio_b0 = params.get('inicio_b0', 0.0)
    inicio_b1 = params.get('inicio_b1', 0.0)
    paso = params.get('paso', 0.05)
    max_iter = params.get('max_iter', 1000)

    cola = deque([(inicio_b0, inicio_b1)])
    visitados = {(round(inicio_b0, 4), round(inicio_b1, 4))}

    mejor_b0, mejor_b1 = inicio_b0, inicio_b1
    mejor_ecm = calcular_ecm(mejor_b0, mejor_b1, x, y)

    historial = [mejor_ecm]
    iteracion = 0

    while cola and iteracion < max_iter:
        actual_b0, actual_b1 = cola.popleft()
        ecm_actual = calcular_ecm(actual_b0, actual_b1, x, y)
        historial.append(ecm_actual)

        if ecm_actual < mejor_ecm:
            mejor_ecm = ecm_actual
            mejor_b0, mejor_b1 = actual_b0, actual_b1

            if mejor_ecm == 0:
                break

        vecinos = generar_vecinos_grid(actual_b0, actual_b1, paso)

        for v_b0, v_b1 in vecinos:
            estado_round = (round(v_b0, 4), round(v_b1, 4))
            if estado_round not in visitados:
                visitados.add(estado_round)
                cola.append((v_b0, v_b1))

        iteracion += 1

    return [mejor_b0, mejor_b1], mejor_ecm, iteracion, historial


def main():
    print("--- Ejecución del algoritmo BFS de regresión lineal (ejemplo simple) ---")

    # Dataset artificial sencillo y autocontenido
    X = np.array([1, 2, 3, 4])
    Y = np.array([2, 4, 6, 8])

    print("Datos utilizados:")
    print("X =", X)
    print("Y =", Y)

    # Parámetros del algoritmo BFS
    params = {
        'inicio_b0': 0.0,
        'inicio_b1': 0.0,
        'paso': 0.1,
        'max_iter': 200
    }

    # Ejecutar BFS
    betas, error, iters, hist = amplitud(X, Y, params)

    print("\n--- Resultados ---")
    print(f"Intercepto (b0): {betas[0]:.4f}")
    print(f"Pendiente (b1):  {betas[1]:.4f}")
    print(f"ECM final:       {error:.6f}")
    print(f"Iteraciones:     {iters}")

    # Validación conceptual
    print("\nComparación esperada:")
    print("La relación real es y = 2x, así que el algoritmo debería aproximar b1 ≈ 2 y b0 ≈ 0.")


if __name__ == "__main__":
    main()