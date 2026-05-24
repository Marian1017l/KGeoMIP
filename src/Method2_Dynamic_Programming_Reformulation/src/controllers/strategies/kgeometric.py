import numpy as np
import time
from typing import List, Tuple

from src.constants.base import NET_LABEL
from src.funcs.base import ABECEDARY, emd_efecto
from src.middlewares.slogger import SafeLogger
from src.models.base.sia import SIA
from src.constants.base import (
    ACTUAL,
    EFECTO,
    TYPE_TAG,
)
from src.constants.models import (
    GEOMETRIC_ANALYSIS_TAG,
    GEOMETRIC_LABEL,
    GEOMETRIC_STRATEGY_TAG,
)
from src.controllers.manager import Manager
from src.funcs.format import fmt_biparte_q
from src.middlewares.profile import profiler_manager, profile
from src.models.core.solution import Solution


class KGeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.logger = SafeLogger(GEOMETRIC_STRATEGY_TAG)
        self.n: int = 0          # variables futuras  (|indices_ncubos|)
        self.m: int = 0          # variables presentes (|dims_ncubos|)
        self.tensors: List[np.ndarray] = []
        self._j_actual: int = 0
        self._popcount: np.ndarray = np.empty(0, dtype=np.int8)

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
    ) -> Solution:
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        self._representacion_inicial()
        tabla      = self._construir_tabla_costos()
        candidatos = self._identificar_candidatos(tabla)
        phi, dist, particion = self._evaluar_candidatos(candidatos)

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=particion,
        )

    # ------------------------------------------------------------------
    # Fase 1 – representación tensorial
    # ------------------------------------------------------------------

    def _representacion_inicial(self) -> None:
        self.n = self.sia_subsistema.indices_ncubos.size
        self.m = self.sia_subsistema.dims_ncubos.size
        self.tensors = [
            self.sia_subsistema.ncubos[i].data.flatten().astype(np.float64)
            for i in range(self.n)
        ]
        S = 1 << self.m
        self._popcount = np.array([bin(x).count("1") for x in range(S)], dtype=np.int8)

    # ------------------------------------------------------------------
    # Fase 2 – tabla de costos vectorizada  O(n · m · 2^m)
    #
    # Fórmula: t(i, j) = γ · (|X[i] − X[j]| + Σ t(k, j))
    #          con γ = 2^(−d),  d = hamming(i, j)
    # Se fija j = estado actual del mecanismo y se barre todo i ∈ {0…S−1}.
    # El loop externo en d garantiza que los vecinos de distancia d−1
    # ya están calculados cuando se necesitan.
    # ------------------------------------------------------------------

    def _construir_tabla_costos(self) -> List[np.ndarray]:
        S        = 1 << self.m
        estados  = np.arange(S, dtype=np.int32)
        popcount = self._popcount

        dims   = self.sia_subsistema.dims_ncubos
        estado = self.sia_subsistema.estado_inicial
        # Índice entero little-endian del estado actual del mecanismo
        self._j_actual = int(
            sum(int(estado[d]) * (1 << local) for local, d in enumerate(dims))
        )

        # Mapa dimensión-global → posición local dentro de dims
        pos_global = {int(d): i for i, d in enumerate(dims)}

        tabla: List[np.ndarray] = []

        for x in range(self.n):
            costos_j = np.zeros(S, dtype=np.float64)
            tensor   = self.tensors[x]
            j        = self._j_actual

            ncubo      = self.sia_subsistema.ncubos[x]
            dims_local = ncubo.dims   # dimensiones de las que depende este n-cubo

            # Índice local de j en el espacio reducido del n-cubo x
            j_local = 0
            for pos_local, d in enumerate(dims_local):
                bit = (j >> pos_global[int(d)]) & 1
                j_local |= bit << pos_local

            # Proyección vectorizada de cada estado global al espacio local del n-cubo x.
            # Imprescindible cuando dims_local ⊂ dims: el tensor vive en 2^|dims_local|
            # pero costos_j opera en 2^m.
            estados_local = np.zeros(S, dtype=np.int32)
            for pos_local, d in enumerate(dims_local):
                bit_col        = (estados >> pos_global[int(d)]) & 1
                estados_local |= bit_col << pos_local

            dist          = popcount[estados ^ j]
            costo_directo = np.abs(tensor[estados_local] - tensor[j_local])

            for d in range(1, self.m + 1):
                gamma    = 2.0 ** (-d)
                states_d = np.where(dist == d)[0]
                if states_d.size == 0:
                    continue
                # vecinos a distancia d-1 ya tienen costos_j calculado porque el loop externo en d es creciente
                vecinos_mat   = states_d[:, None] ^ (1 << np.arange(self.m, dtype=np.int32))
                mask          = popcount[vecinos_mat ^ j] == d - 1
                costo_vecinos = (mask * costos_j[vecinos_mat]).sum(axis=1)
                costos_j[states_d] = gamma * (costo_directo[states_d] + costo_vecinos)

            tabla.append(costos_j)

        return tabla

    # ------------------------------------------------------------------
    # Fase 3 – identificación de candidatos a bipartición
    # ------------------------------------------------------------------

    def _identificar_candidatos(self, tabla: List[np.ndarray]) -> list:
        indices  = self.sia_subsistema.indices_ncubos
        dims     = self.sia_subsistema.dims_ncubos
        j        = self._j_actual
        popcount = self._popcount
        candidatos: set = set()

        for x in range(self.n):
            costos    = tabla[x].copy()
            costos[j] = np.inf

            costo_min   = costos.min()
            estados_min = np.where(
                np.isclose(costos, costo_min, rtol=1e-9, atol=1e-15)
            )[0]

            # Demasiados empates: conservar los más cercanos en Hamming
            if estados_min.size > self.m + 1:
                hamming     = popcount[estados_min ^ j]
                estados_min = estados_min[hamming == hamming.min()]

            # Traducir cada estado mínimo a una bipartición (alcance, mecanismo)
            for i_cand in estados_min:
                mascara = int(i_cand) ^ j
                if mascara == 0:
                    continue
                sub_alcance   = (int(indices[x]),)
                sub_mecanismo = tuple(
                    int(dims[b]) for b in range(self.m) if (mascara >> b) & 1
                )
                if sub_alcance and sub_mecanismo:
                    candidatos.add((sub_alcance, sub_mecanismo))

            # Corte total: nodo futuro x desconectado de todo presente
            candidatos.add(((int(indices[x]),), ()))

            # Pares simples: nodo futuro x vs cada variable del mecanismo
            for b in range(self.m):
                candidatos.add(((int(indices[x]),), (int(dims[b]),)))

        return list(candidatos)

    # ------------------------------------------------------------------
    # Fase 4 – evaluación y selección del MIP
    # ------------------------------------------------------------------

    def _evaluar_candidatos(
        self, candidatos: list
    ) -> Tuple[float, np.ndarray, str]:
        futuros  = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos

        # Vértices completos del grafo bipartito en formato (tipo, nodo)
        vertices: List[Tuple[int, int]] = (
            [(ACTUAL, int(d)) for d in presentes]
            + [(EFECTO, int(i)) for i in futuros]
        )

        mejor_phi  = np.inf
        mejor_dist = None
        mejor_fmt  = None

        for sub_alcance, sub_mecanismo in candidatos:
            arr_alcance   = np.array(sub_alcance,   dtype=np.int8)
            arr_mecanismo = np.array(sub_mecanismo, dtype=np.int8)

            particion      = self.sia_subsistema.bipartir(arr_alcance, arr_mecanismo)
            dist_particion = particion.distribucion_marginal()
            phi            = emd_efecto(dist_particion, self.sia_dists_marginales)

            if phi < mejor_phi:
                mejor_phi  = phi
                mejor_dist = dist_particion

                parte_a = (
                    [(ACTUAL, n) for n in sub_mecanismo]
                    + [(EFECTO, n) for n in sub_alcance]
                )
                parte_a_set = set(parte_a)
                parte_b = [v for v in vertices if v not in parte_a_set]
                mejor_fmt = fmt_biparte_q(parte_a, parte_b)

        return mejor_phi, mejor_dist, mejor_fmt
