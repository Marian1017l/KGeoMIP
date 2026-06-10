import numpy as np
import time
from typing import List, Tuple

from src.constants.base import NET_LABEL
from src.funcs.base import emd_efecto
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
    GEOMETRIC_STRAREGY_TAG,
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
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)
        self.n: int = 0          # variables futuras  (|indices_ncubos|)
        self.m: int = 0          # variables presentes (|dims_ncubos|)
        self.tensors: List[np.ndarray] = []
        self._j_actual: int = 0
        self._popcount: np.ndarray = np.empty(0, dtype=np.int8)
        self._cache_dists: dict[tuple, np.ndarray] = {}

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: int = 2,
    ) -> Solution:
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)
        self._cache_dists = {}

        futuros = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        N_v = len(futuros) + len(presentes)
        
        if k < 2:
            raise ValueError(f"k={k} no es válido. Debe ser k >= 2.")
        if k > N_v:
            raise ValueError(f"k={k} > N_v={N_v}. No es posible crear {k} grupos no vacíos.")
        
        self._representacion_inicial()
        tabla      = self._construir_tabla_costos()
        candidatos = self._identificar_candidatos(tabla)
        phi, dist, particion = self._evaluar_k_particiones(candidatos, k)

        return Solution(
            estrategia=GEOMETRIC_LABEL,
            perdida=phi,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=particion,
        )

    def _representacion_inicial(self) -> None:
        self.n = self.sia_subsistema.indices_ncubos.size
        self.m = self.sia_subsistema.dims_ncubos.size
        self.tensors = [
            self.sia_subsistema.ncubos[i].data.flatten().astype(np.float64)
            for i in range(self.n)
        ]
        S = 1 << self.m
        self._popcount = np.fromiter(
            (x.bit_count() for x in range(S)), dtype=np.int8, count=S
        )

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

        gammas = np.fromiter(
            (2.0 ** (-d) for d in range(self.m + 1)), dtype=np.float64, count=self.m + 1
        )

        j = self._j_actual
        dist_shared        = popcount[estados ^ j]
        states_by_distance = [np.where(dist_shared == d)[0] for d in range(self.m + 1)]

        cache_proj: dict[tuple, np.ndarray] = {}

        tabla: List[np.ndarray] = []
        costos_j = np.zeros(S, dtype=np.float64)

        for x in range(self.n):
            costos_j[:] = 0.0
            tensor   = self.tensors[x]

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
            key_proj = tuple(int(d) for d in dims_local)
            if key_proj not in cache_proj:
                proj = np.zeros(S, dtype=np.int32)
                for pos_local, d in enumerate(dims_local):
                    bit_col  = (estados >> pos_global[int(d)]) & 1
                    proj    |= bit_col << pos_local
                cache_proj[key_proj] = proj
            estados_local = cache_proj[key_proj]

            costo_directo = np.abs(tensor[estados_local] - tensor[j_local])

            for d in range(1, self.m + 1):
                gamma    = gammas[d]
                states_d = states_by_distance[d]
                if states_d.size == 0:
                    continue
                # vecinos a distancia d-1 ya tienen costos_j calculado porque el loop externo en d es creciente
                vecinos_mat   = states_d[:, None] ^ (1 << np.arange(self.m, dtype=np.int32))
                mask          = dist_shared[vecinos_mat] == d - 1
                costo_vecinos = (mask * costos_j[vecinos_mat]).sum(axis=1)
                costos_j[states_d] = gamma * (costo_directo[states_d] + costo_vecinos)

            tabla.append(costos_j.copy())

        return tabla

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

        # Corte total presente: nodo presente b desconectado de todo futuro
        # Simétrico a ((indices[x],), ()) — cubre candidatos de la forma (∅, {pres_b})
        for b in range(self.m):
            candidatos.add(((), (int(dims[b]),)))

        # Instrumentación temporal
        candidatos_futuro_singleton = sum(
            1 for a, m in candidatos if len(a) == 1 and len(m) == 0
        )
        candidatos_presente_singleton = sum(
            1 for a, m in candidatos if len(a) == 0 and len(m) == 1
        )
        self.logger.critic(
            f"[candidatos] futuro_singleton={candidatos_futuro_singleton}"
            f"  presente_singleton={candidatos_presente_singleton}"
            f"  total={len(candidatos)}"
        )

        return list(candidatos)

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

            cache_key = (tuple(sorted(sub_alcance)), tuple(sorted(sub_mecanismo)))
            if cache_key not in self._cache_dists:
                particion = self.sia_subsistema.bipartir(arr_alcance, arr_mecanismo)
                self._cache_dists[cache_key] = particion.distribucion_marginal()
            dist_particion = self._cache_dists[cache_key]
            phi = emd_efecto(dist_particion, self.sia_dists_marginales)

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
    
    def _evaluar_phi_k(self, grupos: List[Tuple[List[int], List[int]]]) -> Tuple[float, np.ndarray, str]:
        """
        Calcula la pérdida REAL (φ) de una k-partición usando `participar_k`.
        Retorna (phi, dist_total, formato_legible)
        """
        # Convertir a arrays para particionar_k
        grupos_arr = [
            (np.array(futuros_g, dtype=np.int8), np.array(presentes_g, dtype=np.int8))
            for futuros_g, presentes_g in grupos
        ]
        
        # Cálculo correcto: una sola distribución conjunta
        dist_total = self.sia_subsistema.particionar_k(grupos_arr).distribucion_marginal()
        phi = emd_efecto(dist_total, self.sia_dists_marginales)
        
        # Formato para mostrar la partición
        vertices = (
            [(ACTUAL, int(d)) for d in self.sia_subsistema.dims_ncubos] +
            [(EFECTO, int(i)) for i in self.sia_subsistema.indices_ncubos]
        )
        
        partes_fmt = []
        for alcance_g, mec_g in grupos:
            parte_a = [(ACTUAL, n) for n in mec_g] + [(EFECTO, n) for n in alcance_g]
            parte_b = [v for v in vertices if v not in set(parte_a)]
            partes_fmt.append(fmt_biparte_q(parte_a, parte_b))
        
        return phi, dist_total, " ‖ ".join(partes_fmt)
    
    def _evaluar_k_exacto(self, k: int) -> Tuple[float, np.ndarray, str]:
        futuros   = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        vertices: List[Tuple[int, int]] = (
            [(ACTUAL, int(d)) for d in presentes] +
            [(EFECTO, int(i)) for i in futuros]
        )
        N_v = len(vertices)

        mejor_phi = np.inf
        mejor_dist = None
        mejor_grupos = None

        for asignacion in self._gen_particiones_rgs(N_v, k):
            grupos_actuales: List[Tuple[List[int], List[int]]] = []
            
            for g in range(1, k + 1):
                futuros_g = [vertices[i][1] for i in range(N_v) 
                            if asignacion[i] == g and vertices[i][0] == EFECTO]
                presentes_g = [vertices[i][1] for i in range(N_v) 
                              if asignacion[i] == g and vertices[i][0] == ACTUAL]
                grupos_actuales.append((futuros_g, presentes_g))
            
            phi, dist, _ = self._evaluar_phi_k(grupos_actuales)
            
            if phi < mejor_phi:
                mejor_phi = phi
                mejor_dist = dist
                mejor_grupos = grupos_actuales[:]

        # Formato final
        _, _, mejor_fmt = self._evaluar_phi_k(mejor_grupos)
        return mejor_phi, mejor_dist, mejor_fmt

    def _gen_particiones_rgs(self, N_v: int, k: int):
        """Genera cada k-partición de {0,...,N_v-1} EXACTAMENTE UNA VEZ, como
        tupla de etiquetas en {1,...,k} (Restricted Growth String desplazada +1
        para conservar la convención de etiquetas 1..k ya usada en el resto
        del método). Evita las k! reetiquetaciones equivalentes que recorre
        itertools.product + filtro."""
        a = [1] * N_v
        def _rec(i: int, max_so_far: int):
            if i == N_v:
                if max_so_far == k:
                    yield tuple(a)
                return
            for v in range(1, min(max_so_far + 1, k) + 1):
                a[i] = v
                yield from _rec(i + 1, max(max_so_far, v))
        yield from _rec(1, 1)

    def _evaluar_k_heuristico(
        self, candidatos: list, k: int
    ) -> Tuple[float, np.ndarray, str]:
        """Heurística mejorada (todo inline) - Mejor para k=4 y k=5"""
        futuros   = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        N_v = len(presentes) + len(futuros)

        vertices: List[Tuple[int, int]] = (
            [(ACTUAL, int(d)) for d in presentes] +
            [(EFECTO, int(i)) for i in futuros]
        )
        v_idx = {v: i for i, v in enumerate(vertices)}

        # Ordenar candidatos por pérdida (mejores primero) - inline
        scored = []
        for sub_alcance, sub_mecanismo in candidatos:
            cache_key = (tuple(sorted(sub_alcance)), tuple(sorted(sub_mecanismo)))
            if cache_key not in self._cache_dists:
                dist_part = self.sia_subsistema.bipartir(
                    np.array(sub_alcance, dtype=np.int8),
                    np.array(sub_mecanismo, dtype=np.int8)
                ).distribucion_marginal()
                self._cache_dists[cache_key] = dist_part
            phi = emd_efecto(self._cache_dists[cache_key], self.sia_dists_marginales)
            scored.append((phi, (sub_alcance, sub_mecanismo)))
        
        scored.sort()  # menor pérdida primero
        candidatos_ordenados = [cand for _, cand in scored]

        restantes = set(range(N_v))
        grupos_info: List[Tuple[List[int], List[int]]] = []

        # Construir k-1 grupos
        for step in range(k - 1):
            if not restantes:
                break

            mejor_cand = None
            mejor_phi_nivel = np.inf

            for sub_alcance, sub_mecanismo in candidatos_ordenados:
                efecto_idxs = {v_idx[(EFECTO, n)] for n in sub_alcance 
                            if (EFECTO, n) in v_idx}
                actual_idxs = {v_idx[(ACTUAL, n)] for n in sub_mecanismo 
                            if (ACTUAL, n) in v_idx}
                grupo_idx = efecto_idxs | actual_idxs

                if not grupo_idx or not (grupo_idx & restantes):
                    continue

                temp_grupos = grupos_info + [(list(sub_alcance), list(sub_mecanismo))]
                temp_grupos.append(([], []))  # placeholder residual

                phi_c, _, _ = self._evaluar_phi_k(temp_grupos)
                
                if phi_c < mejor_phi_nivel:
                    mejor_phi_nivel = phi_c
                    mejor_cand = (list(sub_alcance), list(sub_mecanismo))

            # Si no encontró buen candidato, tomar singleton
            if mejor_cand is None:
                for idx in list(restantes):
                    v_type, v_id = vertices[idx]
                    if v_type == EFECTO:
                        mejor_cand = ([v_id], [])
                    else:
                        mejor_cand = ([], [v_id])
                    break

            if mejor_cand:
                grupos_info.append(mejor_cand)
                alcance_g, mec_g = mejor_cand
                grupo_idx = (
                    {v_idx[(EFECTO, n)] for n in alcance_g if (EFECTO, n) in v_idx} |
                    {v_idx[(ACTUAL, n)] for n in mec_g if (ACTUAL, n) in v_idx}
                )
                restantes -= grupo_idx

        # Grupo residual
        alcance_res = [vertices[i][1] for i in restantes if vertices[i][0] == EFECTO]
        mec_res     = [vertices[i][1] for i in restantes if vertices[i][0] == ACTUAL]
        grupos_info.append((alcance_res, mec_res))

        # === EVITAR GRUPOS VACÍOS (inline) ===
        grupos_info = [g for g in grupos_info if g[0] or g[1]]  # eliminar vacíos
        
        while len(grupos_info) < k and grupos_info:
            # Tomar del grupo más grande y mover un elemento
            grupos_info.sort(key=lambda g: -(len(g[0]) + len(g[1])))
            grande = grupos_info[0]
            if grande[0]:
                nodo = grande[0].pop()
                grupos_info.append(([nodo], []))
            elif grande[1]:
                nodo = grande[1].pop()
                grupos_info.append(([], [nodo]))

        # Evaluación final
        phi_total, mejor_dist, mejor_fmt = self._evaluar_phi_k(grupos_info)

        return phi_total, mejor_dist, mejor_fmt

    def _evaluar_k_particiones(
        self, candidatos: list, k: int = 2
    ) -> Tuple[float, np.ndarray, str]:
        futuros   = self.sia_subsistema.indices_ncubos
        presentes = self.sia_subsistema.dims_ncubos
        N_v = len(presentes) + len(futuros)

        if k == 2:
            return self._evaluar_candidatos(candidatos)
        if k <= 5 and N_v <= 10:
            return self._evaluar_k_exacto(k)

        self.logger.critic("Resultado heurístico: no garantiza MIP global para este tamaño.")
        return self._evaluar_k_heuristico(candidatos, k)