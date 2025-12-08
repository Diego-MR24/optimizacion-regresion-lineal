from .utils.common import calcular_ecm, generar_tabla_resumen
from .utils.decorators import experiment_runner

from .algorithms.recocido import recocido
from .algorithms.amplitud import amplitud
from .algorithms.genetico import genetico

from .visualization.plots import plot_convergencia
from .visualization.plots import plot_solucion
from .visualization.plots import plot_boxplot
