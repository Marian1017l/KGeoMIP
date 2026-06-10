import sys, time
sys.setrecursionlimit(10000)

import numpy as np
import pandas as pd

from src.models.base.application import aplicacion
aplicacion.profiler_habilitado = False
aplicacion.pagina_sample_network = "B"

from src.controllers.manager import Manager
from src.controllers.strategies.kgeometric import KGeometricSIA

ALPHABET = "ABCDEFGHIJKLMNO"
N = 15

def to_bits(letras: str) -> str:
    bits = ['0'] * N
    for c in str(letras):
        idx = ALPHABET.find(c)
        if idx >= 0:
            bits[idx] = '1'
    return ''.join(bits)

tpm = np.genfromtxt("../../data/samples/N15B.csv", delimiter=",")

df = pd.read_excel(
    "/home/mariana10l/proyecto-ada/projecto-analisis-20261/docs/DatosPruebas2026_1.xlsx",
    sheet_name="15B-Elementos", header=None
)
data = df.iloc[5:55, [0, 1, 2]].copy()
data.columns = ["prueba", "alcance", "mecanismo"]
data = data.dropna(subset=["prueba"])

estado = "1" + "0" * (N - 1)
condicion = "1" * N

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50

for df_idx, row in data.iterrows():
    p = int(row["prueba"])
    if p > limit:
        break
    alc = to_bits(str(row["alcance"]).strip())
    mec = to_bits(str(row["mecanismo"]).strip())
    n_v = alc.count("1") + mec.count("1")

    for k in [3, 4, 5]:
        t0 = time.time()
        try:
            sol = KGeometricSIA(Manager(estado_inicial=estado)).aplicar_estrategia(
                condicion, alc, mec, tpm, k=k
            )
            dt = time.time() - t0
            phi = sol.perdida
            par = sol.particion
            es_inf = np.isinf(phi)
            tiene_vacio = "∅" in (par or "")
            print(f"p={p:2d} N_v={n_v:2d} k={k}  phi={phi!s:>12}  t={dt:6.2f}s  "
                  f"INF={es_inf}  VACIO={tiene_vacio}  part={par}")
        except Exception as exc:
            dt = time.time() - t0
            print(f"p={p:2d} N_v={n_v:2d} k={k}  ERROR: {exc}  t={dt:6.2f}s")
