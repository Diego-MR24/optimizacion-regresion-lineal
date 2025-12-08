import time
import pandas as pd
from functools import wraps

def experiment_runner(n_runs=30):
    """
    Ejecuta un algoritmo de optimización 'n_runs' veces y registra:
    - ECM final
    - Iteraciones
    - Tiempo de ejecución
    - Parámetros óptimos (beta_0, beta_1)
    - Historial de error

    Retorna un DataFrame con los resultados.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nombre = func.__name__
            resultados = []

            for i in range(n_runs):
                inicio = time.time()
                betas, ecm, iters, hist = func(*args, **kwargs)
                fin = time.time()

                resultados.append({
                    "Algoritmo": nombre,
                    "Ejecucion": i + 1,
                    "Beta_0": betas[0],
                    "Beta_1": betas[1],
                    "ECM_Final": ecm,
                    "Iteraciones": iters,
                    "Tiempo_seg": fin - inicio,
                    "Historial": hist
                })

            return pd.DataFrame(resultados)
        return wrapper
    return decorator
