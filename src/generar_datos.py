"""Genera el dataset simulado de jugadores juveniles (16-19 años).

Diseño: un score latente de talento = combinación ponderada de métricas
estandarizadas + ruido gaussiano. La etiqueta AltoPotencial marca el
tercio superior del score (prevalencia ~35%).
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)   # semilla fija -> reproducible
N = 40

def generar_metricas(n, rng):
    """Cada métrica ~ Normal(media realista, sd realista), recortada a rango plausible."""
    return pd.DataFrame({
        "PrecisionPase":  np.clip(rng.normal(78, 7, n),    55, 95),    # %
        "VelReaccion":    np.clip(rng.normal(0.62, 0.09, n), 0.42, 0.90),  # s (menor=mejor)
        "Efectividad1v1": np.clip(rng.normal(52, 11, n),   25, 85),    # %
        "VelSprint":      np.clip(rng.normal(29.5, 1.6, n), 25.5, 33.5),   # km/h
        "DistAltaInt":    np.clip(rng.normal(6.8, 1.2, n),  4.0, 10.0),    # x100 m / 90'
        "AciertoRegate":  np.clip(rng.normal(58, 10, n),   30, 85),    # %
        "Recuperaciones": np.clip(rng.normal(6.5, 2.0, n),  2, 12),    # /90'
        "TomaDecision":   np.clip(rng.normal(6.6, 1.1, n),  3.5, 9.5), # índice 0-10
    })

# Pesos "verdaderos" del talento (VelReaccion negativo: menos tiempo = mejor)
PESOS = {
    "PrecisionPase": 0.28, "VelReaccion": -0.30, "Efectividad1v1": 0.25,
    "VelSprint": 0.12, "DistAltaInt": 0.10, "AciertoRegate": 0.18,
    "Recuperaciones": 0.08, "TomaDecision": 0.26,
}

def score_latente(df, rng, ruido=0.55):
    z = (df - df.mean()) / df.std()          # estandarizar para que los pesos sean comparables
    señal = sum(PESOS[c] * z[c] for c in PESOS)
    return señal + rng.normal(0, ruido, len(df))

def ensuciar(df, rng):
    """Inyecta incidencias realistas de anotación de vídeo."""
    df = df.copy()
    df.loc[3,  "PrecisionPase"]  = np.nan   # clip inutilizable
    df.loc[11, "Efectividad1v1"] = np.nan
    df.loc[22, "VelReaccion"]    = np.nan
    df.loc[7,  "VelSprint"]      = 41.2     # error de tracking (imposible)
    return df

if __name__ == "__main__":
    met = generar_metricas(N, RNG)
    score = score_latente(met, RNG)
    df = pd.concat([
        pd.DataFrame({
            "Jugador":  [f"J{i+1:02d}" for i in range(N)],
            "Posicion": RNG.choice(["Delantero","Centrocampista","Defensa","Extremo"],
                                   N, p=[.25,.30,.25,.20]),
            "Edad":     RNG.integers(16, 20, N),
        }),
        met.round(2),
    ], axis=1)
    df["AltoPotencial"] = (score > np.quantile(score, 0.65)).astype(int)

    ensuciar(df, RNG).to_csv("data/raw/jugadores_raw.csv", index=False)
    print(df["AltoPotencial"].mean())   # comprueba prevalencia ~0.35

def generar_dataset(n, seed=42, ruido=0.55):
    """Genera un dataset limpio de n jugadores (sin ensuciar)."""
    rng = np.random.default_rng(seed)
    met = generar_metricas(n, rng)
    score = score_latente(met, rng, ruido)
    df = pd.concat([
        pd.DataFrame({
            "Jugador":  [f"J{i+1:03d}" for i in range(n)],
            "Posicion": rng.choice(["Delantero","Centrocampista","Defensa","Extremo"],
                                   n, p=[.25,.30,.25,.20]),
            "Edad":     rng.integers(16, 20, n),
        }),
        met.round(2),
    ], axis=1)
    df["AltoPotencial"] = (score > np.quantile(score, 0.65)).astype(int)
    return df