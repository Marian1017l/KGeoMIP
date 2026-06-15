import numpy as np
from pathlib import Path

from src.controllers.manager import Manager
from src.controllers.strategies.kgeometric import KGeometricSIA


# ───────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN DE LA PRUEBA  (lo único que el usuario suele tocar)
# ───────────────────────────────────────────────────────────────────────

# Red de muestra (TPM). Cambie el archivo por otra red de src/.samples/
# La longitud del estado y las máscaras debe coincidir con el tamaño de la red.
RED_CSV = "src/.samples/N20A.csv"

# Estado inicial del sistema (cadena binaria, 1=ON, 0=OFF).
ESTADO_INICIAL = "10000000000000000000"

# Máscaras del subsistema a analizar (misma longitud que el estado inicial).
CONDICION = "11111111111111111111"   # variables condicionadas (fondo)
ALCANCE   = "11110000000000000000"   # purview / futuro
MECANISMO = "00001111000000000000"   # mecanismo / presente

# Valores de k a probar. Recuerde: 2 <= k <= N_v (alcance + mecanismo).
VALORES_K = [2, 3, 4, 5]


# ───────────────────────────────────────────────────────────────────────
# 2. CARGA DE LA TPM
# ───────────────────────────────────────────────────────────────────────

def cargar_tpm(ruta_csv: str) -> np.ndarray:
    ruta = Path(ruta_csv)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró la red en '{ruta}'. "
            f"Verifique la ruta y que el archivo exista en src/.samples/."
        )
    return np.genfromtxt(ruta, delimiter=",")


# ───────────────────────────────────────────────────────────────────────
# 3. IMPRESIÓN DE RESULTADOS
# ───────────────────────────────────────────────────────────────────────

def imprimir_resultado(k: int, solucion) -> None:
    print(f"\n--- Resultado | k={k} ---")
    print(f"  k          : {k}")
    print(f"  Pérdida φ  : {solucion.perdida}")
    print(f"  Partición  :\n{solucion.particion}")
    print(f"  Tiempo (s) : {solucion.tiempo_total:.4f}")


# ───────────────────────────────────────────────────────────────────────
# 4. FLUJO PRINCIPAL DE PRUEBA
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Validación rápida de longitudes coherentes.
    n = len(ESTADO_INICIAL)
    for nombre, mascara in (("CONDICION", CONDICION),
                            ("ALCANCE", ALCANCE),
                            ("MECANISMO", MECANISMO)):
        if len(mascara) != n:
            raise ValueError(
                f"La máscara {nombre} tiene longitud {len(mascara)}, "
                f"pero el estado inicial tiene longitud {n}. Deben coincidir."
            )

    # Cargar la red.
    tpm = cargar_tpm(RED_CSV)
    print(f"TPM cargada desde: {RED_CSV}")
    print(f"Estado inicial   : {ESTADO_INICIAL}")
    print(f"Subsistema       : alcance={ALCANCE} | mecanismo={MECANISMO}")

    # Inicializar el sistema (configuración).
    config_sistema = Manager(estado_inicial=ESTADO_INICIAL)

    # Instanciar la estrategia KGeoMIP una sola vez (es reutilizable).
    estrategia = KGeometricSIA(config_sistema)

    # Ejecutar la búsqueda de k-particiones para cada valor de k.
    for k in VALORES_K:
        print("\n" + "=" * 60)
        print(f"=== Probando k = {k} ===")
        print("=" * 60)
        try:
            solucion = estrategia.aplicar_estrategia(
                CONDICION, ALCANCE, MECANISMO, tpm, k
            )
            imprimir_resultado(k, solucion)
        except ValueError as err:
            print(f"  [Saltado] k={k} no es válido: {err}")


if __name__ == "__main__":
    main()