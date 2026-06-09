"""
Runner CLI para KGeometricSIA.  Llamado como subproceso por comparacion_benchmark.py.

Uso:
    python kgeo_runner.py --estado 1000 --condicion 1111 --alcance 1111 \
                          --mecanismo 1111 --k 2 --samples /abs/path/samples

Salida: JSON en stdout
    { "phi": float, "tiempo": float, "particion": str,
      "modo": "exacto" | "heuristico_candidatos" | "heuristico_greedy",
      "n_vertices": int, "error": null | str }
"""

import argparse
import json
import os
import sys
import time

sys.setrecursionlimit(10000)

import numpy as np

# ── Argumentos ───────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--estado",    required=True)
ap.add_argument("--condicion", required=True)
ap.add_argument("--alcance",   required=True)
ap.add_argument("--mecanismo", required=True)
ap.add_argument("--k",         type=int, default=2)
ap.add_argument("--samples",   required=True, help="Path absoluto al directorio de muestras CSV")
ap.add_argument("--pagina",    default="A")
args = ap.parse_args()

# ── Apuntar a los samples y deshabilitar profiling ───────────────────────────
os.environ["GEOMIP_SAMPLES_DIR"] = args.samples

from src.models.base.application import aplicacion
aplicacion.pagina_sample_network = args.pagina
aplicacion.profiler_habilitado   = False

from src.controllers.manager import Manager
from src.controllers.strategies.kgeometric import KGeometricSIA

# ── Determinar modo antes de ejecutar ────────────────────────────────────────
n = sum(1 for b in args.alcance   if b == "1")
m = sum(1 for b in args.mecanismo if b == "1")
n_vertices = n + m

if args.k == 2:
    modo = "heuristico_candidatos"
elif args.k <= 4 and n_vertices <= 10:
    modo = "exacto"
else:
    modo = "heuristico_greedy"

# ── Ejecutar ─────────────────────────────────────────────────────────────────
resultado = {
    "phi":        None,
    "tiempo":     None,
    "particion":  None,
    "modo":       modo,
    "n_vertices": n_vertices,
    "error":      None,
}

try:
    gestor = Manager(estado_inicial=args.estado)
    tpm    = np.genfromtxt(gestor.tpm_filename, delimiter=",")

    analizador = KGeometricSIA(gestor)
    sol = analizador.aplicar_estrategia(
        condicion=args.condicion,
        alcance=args.alcance,
        mecanismo=args.mecanismo,
        tpm=tpm,
        k=args.k,
    )
    resultado["phi"]       = float(sol.perdida)
    resultado["tiempo"]    = float(sol.tiempo_ejecucion)
    resultado["particion"] = sol.particion

except Exception as exc:
    resultado["error"] = str(exc)

print(json.dumps(resultado))
