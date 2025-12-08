import sys
import os
import random
import numpy as np

# Ajuste de ruta para importar calcular_ecm
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.utils.common import calcular_ecm


def seleccion_torneo(poblacion, k=3):
    """
    Selección mediante torneo.

    Se eligen aleatoriamente 'k' individuos de la población y 
    se retorna aquel con menor valor de error (ECM).

    Parámetros
    ----------
    poblacion : list
        Lista de individuos, cada uno con la forma [b0, b1, ecm].
    k : int
        Cantidad de participantes del torneo.

    Retorna
    -------
    list
        Individuo ganador del torneo.
    """
    torneo = random.sample(poblacion, k)
    ganador = min(torneo, key=lambda ind: ind[2])
    return ganador


def cruza_aritmetica_completa(padre1, padre2):
    """
    Cruza aritmética completa entre dos individuos.

    La combinación se basa en un factor aleatorio alpha:
        hijo = alpha * padre1 + (1 - alpha) * padre2

    Parámetros
    ----------
    padre1 : list
        Individuo con formato [b0, b1, ecm].
    padre2 : list
        Individuo con formato [b0, b1, ecm].

    Retorna
    -------
    tuple
        (hijo1, hijo2) donde cada hijo es [b0, b1].
    """
    alpha = random.random()

    p1_b0, p1_b1 = padre1[0], padre1[1]
    p2_b0, p2_b1 = padre2[0], padre2[1]

    h1_b0 = alpha * p1_b0 + (1 - alpha) * p2_b0
    h1_b1 = alpha * p1_b1 + (1 - alpha) * p2_b1

    h2_b0 = alpha * p2_b0 + (1 - alpha) * p1_b0
    h2_b1 = alpha * p2_b1 + (1 - alpha) * p1_b1

    return [h1_b0, h1_b1], [h2_b0, h2_b1]


def mutacion_uniforme(individuo, prob_mutacion, rango_mutacion):
    """
    Mutación uniforme sobre un individuo.

    Cada gen (b0 o b1) tiene prob_mutacion de sufrir 
    un cambio aleatorio dentro de [-rango_mutacion, +rango_mutacion].

    Parámetros
    ----------
    individuo : list
        Genes del individuo en forma [b0, b1].
    prob_mutacion : float
        Probabilidad de que cada gen mute.
    rango_mutacion : float
        Magnitud de la perturbación uniforme.

    Retorna
    -------
    list
        Individuo mutado.
    """
    b0, b1 = individuo

    if random.random() < prob_mutacion:
        b0 += random.uniform(-rango_mutacion, rango_mutacion)

    if random.random() < prob_mutacion:
        b1 += random.uniform(-rango_mutacion, rango_mutacion)

    return [b0, b1]


def genetico(x, y, params):
    """
    Algoritmo Genético para aproximar los parámetros de una 
    regresión lineal simple minimizando el ECM.

    Cada individuo representa un par (b0, b1).

    Parámetros
    ----------
    x, y : array_like
        Datos de entrada y salida.
    params : dict
        - tam_poblacion : tamaño de la población
        - generaciones : número de generaciones
        - prob_mutacion : probabilidad de mutar cada gen
        - rango_mutacion : magnitud de la mutación
        - k_torneo : tamaño del torneo
        - rango_inicio : (min, max) para inicializar poblaciones

    Retorna
    -------
    tuple
        ([b0, b1], mejor_ecm, generaciones_usadas, historial_ecm)
    """

    tam_poblacion = params.get('tam_poblacion', 50)
    num_generaciones = params.get('generaciones', 100)
    prob_mutacion = params.get('prob_mutacion', 0.1)
    rango_mutacion = params.get('rango_mutacion', 0.1)
    k_torneo = params.get('k_torneo', 3)
    rango_ini = params.get('rango_inicio', (-10, 10))

    # Inicialización de población
    poblacion = []
    for _ in range(tam_poblacion):
        b0 = random.uniform(*rango_ini)
        b1 = random.uniform(*rango_ini)
        ecm = calcular_ecm(b0, b1, x, y)
        poblacion.append([b0, b1, ecm])

    poblacion.sort(key=lambda ind: ind[2])
    mejor_global = list(poblacion[0])
    historial = []

    # Bucle generacional
    for gen in range(num_generaciones):

        mejor_actual = poblacion[0]
        historial.append(mejor_actual[2])

        if mejor_actual[2] < mejor_global[2]:
            mejor_global = list(mejor_actual)

        if mejor_global[2] < 1e-10:
            break

        nueva = [list(poblacion[0])]  # Elitismo

        while len(nueva) < tam_poblacion:
            padre1 = seleccion_torneo(poblacion, k_torneo)
            padre2 = seleccion_torneo(poblacion, k_torneo)

            hijo1, hijo2 = cruza_aritmetica_completa(padre1, padre2)

            hijo1 = mutacion_uniforme(hijo1, prob_mutacion, rango_mutacion)
            hijo2 = mutacion_uniforme(hijo2, prob_mutacion, rango_mutacion)

            ecm1 = calcular_ecm(hijo1[0], hijo1[1], x, y)
            ecm2 = calcular_ecm(hijo2[0], hijo2[1], x, y)

            nueva.append([hijo1[0], hijo1[1], ecm1])
            if len(nueva) < tam_poblacion:
                nueva.append([hijo2[0], hijo2[1], ecm2])

        nueva.sort(key=lambda ind: ind[2])
        poblacion = nueva

    return [mejor_global[0], mejor_global[1]], mejor_global[2], gen, historial


def main():
    """
    Prueba unitaria simple.

    Se utiliza el modelo Y = 4X + 2 como objetivo.
    El algoritmo debe aproximar b0 ≈ 2 y b1 ≈ 4.
    """
    print("--- Prueba Algoritmo Genético ---")

    X = np.array([1, 2, 3, 4, 5])
    Y = 4 * X + 2

    print("Modelo objetivo: b0 = 2, b1 = 4")

    params = {
        'tam_poblacion': 40,
        'generaciones': 120,
        'prob_mutacion': 0.15,
        'rango_mutacion': 0.5,
        'k_torneo': 3,
        'rango_inicio': (0, 10)
    }

    betas, error, gens, hist = genetico(X, Y, params)

    print("\nResultados:")
    print(f"b0 estimado: {betas[0]:.4f}")
    print(f"b1 estimado: {betas[1]:.4f}")
    print(f"ECM final:   {error:.6f}")
    print(f"Generaciones: {gens}")


if __name__ == "__main__":
    main()
