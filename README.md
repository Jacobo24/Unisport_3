# Unisport_3 — Análisis predictivo en el scouting deportivo

Tarea 3.1 del Módulo 3 del Máster en Big Data e IA en el Deporte (Unisport
Management School). Detección de talento en jóvenes futbolistas (16–19 años)
a partir de métricas de rendimiento extraídas de vídeo, con datos simulados
mediante código.

## Estructura del proyecto
├── data/  
│   ├── raw/          # Dataset bruto (con valores ausentes y un outlier)  
│   └── processed/    # Dataset limpio generado por el notebook 01  
├── src/  
│   └── generar_datos.py      # Generación del dataset simulado  
├── notebooks/  
│   ├── 01_exploracion_limpieza.ipynb   # Auditoría, limpieza y descriptivos  
│   ├── 02_modelado.ipynb               # Regresión logística vs Random Forest  
│   ├── 03_evaluacion.ipynb             # Métricas, ROC, validación cruzada, umbral  
│   └── 04_tamano_muestral.ipynb        # Curva de aprendizaje (AUC vs n)  
├── figures/          # Figuras en PDF que importa el informe LaTeX  
├── informe/  
│   ├── Documento.pdf  
│   └── tablas/   # Tablas exportadas a LaTeX desde los notebooks  
├── models/           # Pipelines entrenados y partición train/test (joblib)  
└── requirements.txt  

## Reproducir el análisis

```bash
python -m venv .venv
.venv\Scripts\activate        # En Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python src/generar_datos.py   # Genera data/raw/jugadores_raw.csv
```

Después, ejecutar los notebooks en orden (01 → 04). Cada uno guarda sus
salidas (dataset limpio, modelos, figuras y tablas), de modo que el informe
se regenera por completo desde el dato bruto.

## Resultados principales

- Modelo final: regresión logística (pipeline con estandarización),
  seleccionada frente a Random Forest por menor sobreajuste e
  interpretabilidad.
- AUC con n=40: 0,66 en test y 0,62 ± 0,18 en validación cruzada 5-fold.
- La curva de aprendizaje muestra que ampliando la muestra a 150–300
  jugadores el AUC alcanzaría ≈ 0,85, el techo estimado de la simulación.

## Informe

El documento final (LaTeX/PDF, elaborado en Overleaf) se encuentra en la
carpeta `informe/`.
