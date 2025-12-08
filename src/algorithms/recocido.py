import sys
import os
import math
import random
import numpy as np

# Ajuste de path (solo si se usa dentro de un proyecto estructurado)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.utils.common import calcular_ecm


def generar_vecino(b0, b1, paso):
    """
    Genera un vecino perturbando aleatoriamente los parámetros.

    La perturbación es uniforme en el rango [-paso, paso].

    Parámetros
    ----------
    b0, b1 : float
        Parámetros actuales de la regresión.
    paso : float
        Magnitud máxima de la perturbación.

    Retorna
    -------
    (float, float)
        Nuevos valores (b0, b1) vecinos.
    """
    return (
        b0 + random.uniform(-paso, paso),
        b1 + random.uniform(-paso, paso)
    )


def recocido(x, y, params):
    """
    Implementación del algoritmo de Recocido Simulado (Simulated Annealing)
    aplicado a la minimización del Error Cuadrático Medio (ECM) en un modelo
    de regresión lineal simple.

    Parámetros
    ----------
    x, y : array_like
        Datos de entrenamiento.
    params : dict
        Parámetros del algoritmo:
            - inicio_b0, inicio_b1 : valores iniciales.
            - t_inicial : temperatura inicial.
            - t_final : temperatura mínima para detener el proceso.
            - alpha : tasa de enfriamiento (0 < alpha < 1).
            - paso : magnitud de perturbación en vecinos.

    Retorna
    -------
    list
        Mejor par [b0, b1] encontrado.
    float
        Valor mínimo de ECM alcanzado.
    int
        Número total de iteraciones realizadas.
    list
        Historial de ECM en cada iteración.
    """

    b0_actual = params.get('inicio_b0', 0.0)
    b1_actual = params.get('inicio_b1', 0.0)
    temp_actual = params.get('t_inicial', 100.0)
    temp_final = params.get('t_final', 0.01)
    alpha = params.get('alpha', 0.95)
    paso = params.get('paso', 0.1)

    ecm_actual = calcular_ecm(b0_actual, b1_actual, x, y)

    mejor_b0, mejor_b1 = b0_actual, b1_actual
    mejor_ecm = ecm_actual

    historial = [ecm_actual]                        
    iteracion = 0

    while temp_actual > temp_final:

        # Generación de vecino
        b0_vecino, b1_vecino = generar_vecino(b0_actual, b1_actual, paso)
        ecm_vecino = calcular_ecm(b0_vecino, b1_vecino, x, y)

        delta = ecm_vecino - ecm_actual

        # Regla de aceptación
        if delta < 0:
            aceptar = True
        else:
            prob = math.exp(-delta / temp_actual)
            aceptar = random.random() < prob

        # Actualización si se acepta la transición
        if aceptar:
            b0_actual = b0_vecino
            b1_actual = b1_vecino
            ecm_actual = ecm_vecino

            if ecm_actual < mejor_ecm:
                mejor_ecm = ecm_actual
                mejor_b0 = b0_actual
                mejor_b1 = b1_actual

        historial.append(ecm_actual)

        # Enfriamiento
        temp_actual *= alpha
        iteracion += 1

    return [mejor_b0, mejor_b1], mejor_ecm, iteracion, historial


def main():
    print("--- Prueba del algoritmo de Recocido Simulado ---")

    # Datos sintéticos exactos: y = 4x + 2
    X = np.array([0, 1, 2, 3, 4, 5])
    Y = 4 * X + 2

    print("Modelo objetivo: b0 = 2, b1 = 4")

    params = {
        'inicio_b0': 0.0,
        'inicio_b1': 0.0,
        't_inicial': 500,
        't_final': 0.001,
        'alpha': 0.97,
        'paso': 0.3
    }

    betas, error, iters, hist = recocido(X, Y, params)

    print("\nResultados:")
    print(f"b0 estimado: {betas[0]:.4f}")
    print(f"b1 estimado: {betas[1]:.4f}")
    print(f"ECM final:   {error:.6f}")
    print(f"Iteraciones: {iters}")

if __name__ == "__main__":
    main()
