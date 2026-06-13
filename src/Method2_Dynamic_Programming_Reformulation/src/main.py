# from src.controllers.manager import Manager

# from src.controllers.strategies.force import BruteForce
# from src.controllers.strategies.q_nodes import QNodes
# from src.controllers.strategies.geometric import GeometricSIA


# def iniciar():
#     """Punto de entrada principal"""
#                     # ABCD #
#     # estado_inicial = "100"
#     # condiciones =    "111"
#     # alcance =        "111"
#     # mecanismo =      "111"
#     # estado_inicial = "0000"
#     # condiciones =    "1111"
#     # alcance =        "1111"
#     # mecanismo =      "1111"
#     # estado_inicial = "1000"
#     # condiciones =    "1111"
#     # alcance =        "0111"
#     # mecanismo =      "1111"
#     # estado_inicial = "100000"
#     # condiciones =    "111111"
#     # alcance =        "101011"
#     # mecanismo =      "111111"
#     # estado_inicial = "100000"
#     # condiciones =    "111111"
#     # alcance =        "111111"
#     # mecanismo =      "111111"
#     # estado_inicial = "100000"
#     # condiciones =    "111111"
#     # alcance =        "111111"
#     # mecanismo =      "011111"
#     # estado_inicial = "1000000000"
#     # condiciones =    "1111111111"
#     # alcance =        "1111111111"
#     # mecanismo =      "1111111111"
#     estado_inicial = "1000000000"
#     condiciones =    "1111111111"
#     alcance =        "0101010101"
#     mecanismo =      "1111111111"
#     # estado_inicial = "1000000000"
#     # condiciones =    "1111111111"
#     # alcance =        "1111111110"
#     # mecanismo =      "1111111111"
#     # estado_inicial = "10000000000000000000"
#     # condiciones =    "11111111111111111111"
#     # alcance =        "11111111111111111111"
#     # mecanismo =      "11111111111111111111"
#     # estado_inicial = "10000000000000000000"
#     # condiciones =    "11111111111111111111"
#     # alcance =        "11011011011011011011"
#     # mecanismo =      "10101010101010101010"

#     gestor_sistema = Manager(estado_inicial)

#     ### Ejemplo de solución mediante módulo de fuerza bruta ###
#     analizador_fb = GeometricSIA(gestor_sistema)
#     # analizador_fb = BruteForce(gestor_sistema)
#     sia_uno = analizador_fb.aplicar_estrategia(
#         condiciones,
#         alcance,
#         mecanismo,
#     )
#     print(sia_uno)
from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.kgeometric import KGeometricSIA
from src.controllers.strategies.q_nodes import QNodes
# Optional import: this project often runs only geometric strategy.
try:
    from src.controllers.strategies.phi import Phi
except Exception:
    Phi = None
import multiprocessing
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path


METHOD2_ROOT = Path(__file__).resolve().parents[1]
GEOMIP_ROOT = Path(__file__).resolve().parents[3]

def convertir_a_binario(texto, n_bits=20):
    posiciones = "ABCDEFGHIJKLMNOPQRST"[:n_bits]
    binario = ["0"] * n_bits
    for letra in texto:
        if letra in posiciones:
            binario[posiciones.index(letra)] = "1"
    return "".join(binario)

def ejecutar_con_tiempo(config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm):
    try:
        analizador_fi = GeometricSIA(config_sistema)
        sia_dos = analizador_fi.aplicar_estrategia(condiciones, alcance, mecanismo, tpm)
        resultado_queue.put({
            "particion": sia_dos.particion,
            "perdida": str(sia_dos.perdida).replace('.', ','),
            "tiempo": str(sia_dos.tiempo_ejecucion).replace('.', ','),
        })

    except Exception as e:
        resultado_queue.put({
            "particion": None,
            "perdida": None,
            "tiempo": None,
        })

def resolver_tpm_path(estado_inicio: str) -> Path:
    """Find TPM file in common project locations based on state size."""
    sample_name = f"N{len(estado_inicio)}A.csv"
    candidates = (
        METHOD2_ROOT / "src" / ".samples" / sample_name,
        METHOD2_ROOT / ".samples" / sample_name,
        GEOMIP_ROOT / "data" / "samples" / sample_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No se encontró la TPM '{sample_name}'. Busqué en: {', '.join(str(c) for c in candidates)}"
    )


def inferir_estado_inicial() -> str:
    """Infer an initial state from available datasets (prefers largest NxA.csv)."""
    sample_dirs = (
        METHOD2_ROOT / "src" / ".samples",
        METHOD2_ROOT / ".samples",
        GEOMIP_ROOT / "data" / "samples",
    )
    pattern = re.compile(r"N(\d+)[A-Z]\.csv$")
    available_sizes = []

    for sample_dir in sample_dirs:
        if not sample_dir.exists():
            continue
        for sample_file in sample_dir.glob("N*.csv"):
            match = pattern.match(sample_file.name)
            if match:
                available_sizes.append(int(match.group(1)))

    if not available_sizes:
        raise FileNotFoundError("No hay archivos de muestras TPM disponibles en data/samples ni .samples.")

    n_bits = max(available_sizes)
    return "1" + ("0" * (n_bits - 1))


def ejecutar_desde_excel(
    ruta_excel: Path,
    ruta_salida: Path,
    inicio=0,
    cantidad=50,
    estado_inicio: str | None = None,
    condiciones: str | None = None,
):
    df = pd.read_excel(ruta_excel, sheet_name=8, usecols="B", skiprows=3, names=["Subsistema"]) #! here
    filas = df["Subsistema"].dropna().tolist()
    filas = filas[inicio:inicio + cantidad]
    resultados = []

    estado_inicio = estado_inicio or inferir_estado_inicial()
    condiciones = condiciones or ("1" * len(estado_inicio))
    tpm_path = resolver_tpm_path(estado_inicio)
    tpm = np.genfromtxt(tpm_path, delimiter=",")

    for i, fila in enumerate(filas, start=inicio + 1):
        partes = fila.split("|")
        if len(partes) != 2:
            continue

        alcance = convertir_a_binario(partes[0][:len(partes[0]) - 3], n_bits=len(estado_inicio))
        mecanismo = convertir_a_binario(partes[1][:len(partes[1]) - 1], n_bits=len(estado_inicio))
        print(f"Iteración {i} - Alcance: {alcance}, Mecanismo: {mecanismo}")

        config_sistema = Manager(estado_inicial=estado_inicio)

        resultado_queue = multiprocessing.Queue()
        proceso = multiprocessing.Process(target=ejecutar_con_tiempo, args=(config_sistema, condiciones, alcance, mecanismo, resultado_queue, tpm))
        
        proceso.start()
        proceso.join(timeout=3600)  

        if proceso.is_alive():
            print(f"Iteración {i} - Tiempo límite alcanzado, terminando proceso...")
            proceso.terminate()
            proceso.join()
            resultado = {"perdida": None, "tiempo": None, "particion": None}
        else:
            resultado = (
                resultado_queue.get()
                if not resultado_queue.empty()
                else {"perdida": None, "tiempo": None, "particion": None}
            )

        resultados.append({
            "Iteración": i,
            "Alcance": alcance,
            "Mecanismo": mecanismo,
            "Partición": resultado["particion"],
            "Pérdida": resultado["perdida"],
            "Tiempo de ejecución (s)": resultado["tiempo"],
        })
    df_resultados = pd.DataFrame(resultados)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_excel(ruta_salida, index=False)
    print(f"Resultados guardados en {ruta_salida}")


def ejecutar_pruebas_kgeometric_tpm_real():
    """Corre KGeometricSIA con las TPMs reales de data/samples (N4A..N10A),
    cubriendo las tres rutas de _evaluar_k_particiones:
      k=2              -> _evaluar_candidatos
      k=3/4, N_v <= 10 -> _evaluar_k_exacto (RGS)
      resto            -> _evaluar_k_heuristico
    """
    casos = [
        # (estado_inicial, alcance, mecanismo, k)
        ("1000",       "1111",       "1111",       2),  # N_v=8  -> exacto bipartición
        ("1000",       "1111",       "1111",       3),  # N_v=8  -> exacto RGS
        ("10000",      "11111",      "11111",      2),  # N_v=10 -> exacto bipartición
        ("10000",      "11111",      "11111",      4),  # N_v=10 -> exacto RGS (peor caso)
        ("100000",     "111111",     "111111",     2),  # N_v=12 -> exacto bipartición
        ("100000",     "111111",     "111111",     3),  # N_v=12 -> heurístico greedy DP
        ("10000000",   "11111111",   "11111111",   3),  # N_v=16 -> heurístico greedy DP
        ("1000000000", "1111111111", "1111111111", 4),  # N_v=20 -> heurístico greedy DP
    ]

    sep = "=" * 60
    for estado_inicial, alcance, mecanismo, k in casos:
        condiciones = "1" * len(estado_inicial)
        tpm_path = resolver_tpm_path(estado_inicial)
        tpm = np.genfromtxt(tpm_path, delimiter=",")

        resultado = KGeometricSIA(Manager(estado_inicial)).aplicar_estrategia(
            condiciones, alcance, mecanismo, tpm, k=k
        )

        print(f"\n{sep}")
        print(f"TPM={tpm_path.name}  nodos={len(estado_inicial)}  k={k}")
        print(sep)
        print(f"phi        : {resultado.perdida:.8f}")
        print(f"tiempo     : {resultado.tiempo_ejecucion:.4f}s")
        print(f"partición  : {resultado.particion}")


def comparar_estrategias(n_nodos: int = 6, k: int = 2, seed: int = 42) -> None:
    """Corre GeometricSIA y KGeometricSIA sobre el mismo sistema y compara phi."""
    estado_inicial = "1" + "0" * (n_nodos - 1)
    condiciones    = "1" * n_nodos
    alcance        = "1" * n_nodos
    mecanismo      = "1" * n_nodos

    S = 1 << n_nodos
    np.random.seed(seed)
    tpm = np.random.rand(S, n_nodos)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"COMPARACIÓN  GeometricSIA  vs  KGeometricSIA (k={k})")
    print(f"Nodos: {n_nodos}  |  Estado: {estado_inicial}  |  seed={seed}")
    print(sep)

    res_geo = GeometricSIA(Manager(estado_inicial)).aplicar_estrategia(
        condiciones, alcance, mecanismo, tpm
    )
    print(f"\n[GeometricSIA]       phi={res_geo.perdida:.8f}  t={res_geo.tiempo_ejecucion:.4f}s")

    res_k = KGeometricSIA(Manager(estado_inicial)).aplicar_estrategia(
        condiciones, alcance, mecanismo, tpm, k=k
    )
    print(f"[KGeometricSIA k={k}]  phi={res_k.perdida:.8f}  t={res_k.tiempo_ejecucion:.4f}s")

    diff = abs(res_geo.perdida - res_k.perdida)
    if diff < 1e-9:
        print(f"\n  RESULTADO: phi IDENTICO  (diff={diff:.2e})  ✓")
    else:
        print(f"\n  RESULTADO: phi DIFERENTE (diff={diff:.8f})  ✗")
    print(sep)


def iniciar():
    ruta_entrada = Path(
        os.getenv(
            "GEOMIP_INPUT_XLSX",
            str(GEOMIP_ROOT / "results" / "Pruebas_Metodo2.xlsx"),
        )
    )
    ruta_salida = Path(
        os.getenv(
            "GEOMIP_OUTPUT_XLSX",
            str(GEOMIP_ROOT / "results" / "resultados_Geometric.xlsx"),
        )
    )
    ejecutar_desde_excel(ruta_entrada, ruta_salida)

def probar_mismo_sistema_15_nodos():
    estado_inicial = "1" + "0" * 14   # 15 nodos
    condiciones    = "1" * 15
    alcance        = "1" * 15
    mecanismo      = "1" * 15

    tpm_path = resolver_tpm_path(estado_inicial)
    tpm = np.genfromtxt(tpm_path, delimiter=",")

    ks = [2, 3, 4, 5, 6]

    sep = "=" * 70

    print(f"\nSistema fijo de {len(estado_inicial)} nodos")
    print(f"TPM: {tpm_path.name}")

    for k in ks:
        print(f"\n{sep}")
        print(f"PRUEBA k={k}")
        print(sep)

        resultado = KGeometricSIA(
            Manager(estado_inicial)
        ).aplicar_estrategia(
            condiciones,
            alcance,
            mecanismo,
            tpm,
            k=k,
        )

        print(f"phi       : {resultado.perdida:.8f}")
        print(f"tiempo    : {resultado.tiempo_ejecucion:.4f}s")
        print(f"particion : {resultado.particion}")


if __name__ == "__main__":
    # Pruebas con TPMs reales de data/samples (N4A..N10A): ejercitan las tres
    # rutas de _evaluar_k_particiones sobre datos del proyecto, no aleatorios.
    # ejecutar_pruebas_kgeometric_tpm_real()
    probar_mismo_sistema_15_nodos()
