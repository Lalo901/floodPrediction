# Bitácora de Decisiones y Hallazgos (ADR Ligero)

Registro corto de decisiones técnicas verificadas durante el proyecto. Sirve como respaldo para la defensa de 15 minutos.

---

## 2026-11-23 — Auditoría inicial del proyecto y alineación al MVP

### Hallazgos verificados en disco
- **Dataset Sen1Floods11 completo:** ubicado en `dataset/data/` y `dataset/splits/flood_handlabeled/` (sin subcarpeta `v1.1`).
  - `HandLabeled`: 446 chips de `S1Hand` (`float32`, VV/VH en dB, con `NaN`s), `S2Hand` (`int16`, 13 bandas L1C), `LabelHand` (`int16`, `{-1, 0, 1}`), `JRCWaterHand` y `S1OtsuLabelHand`.
  - `WeaklyLabeled`: 4,384 chips con etiquetas automáticas.
  - Splits oficiales exactos: 252 train, 89 valid, 90 test, 15 Bolivia.
  - Metadatos completos en `dataset/Sen1Floods11_Metadata.geojson`.
- **Hardware local:** sin GPU NVIDIA (`CPU only`), lo que confirma que el entrenamiento de DeepLabV3+ debe correr en Google Colab.
- **Problemas detectados en la versión anterior (archivada en `_legacy/`):**
  - Conflicto de arquitectura (Gradio + DuckDB + 2 entornos mamba vs Streamlit + SQLite + Docker).
  - `random_forest_balanced.joblib` pesaba **2.0 GB** (inviable en Hugging Face Space).
  - Scripts antiguos tenían bugs críticos con `-1` (NoData) y `NaN`s en S1 (`CrossEntropyLoss` sin `ignore_index=-1`, bitwise `&`/`|` sobre `-1`, normalización incorrecta sobre dB).

### Decisiones tomadas
1. **SDD liviano en `Agent/`:** separar contrato estable (`spec.md`), plan vivo (`plan.md`) y bitácora (`decisions.md`), con `AGENTS.md` breve en la raíz.
2. **Solo `HandLabeled` (446 chips) para el MVP:** descartar los 4,384 `WeaklyLabeled` para evitar entrenar sobre ruido automático de umbrales y evitar multiplicar por 10× las descargas de terreno en GEE. Mantenerlos en disco sin borrar como trabajo futuro.
3. **Métrica principal congelada:** **Flood IoU** (clase 1 = agua), descartando `label == -1` y acumulando `TP/FP/FN` globalmente por split y por evento.
4. **Modelos livianos para CPU del Space:** limitar complejidad de RF para mantenerlo `< 50 MB`; usar `DeepLabV3+` con encoder `resnet34` exportado a **ONNX** (`onnxruntime`).
5. **Respaldo seguro:** código y modelos antiguos movidos a `_legacy/` (excluido de git) sin tocar `dataset/`.

---

## 2026-11-23 — EDA `HandLabeled` (446 chips) y Baseline Oficial S1 Otsu

Ejecutado con `training/eda_and_baseline.py` y `core/metrics.py` (ignorando estrictamente `label == -1`). **0 archivos faltantes** en los 446 chips (`S1Hand`, `S2Hand`, `LabelHand`, `JRCWaterHand`, `S1OtsuLabelHand`).

### Resumen por split (piso a superar)

| Split | Chips | S1 NaN % | Label `-1` % | Agua % (válidos) | Otsu Flood IoU | Otsu F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train` | 252 | 2.01% | 13.26% | 9.51% | 0.4513 | 0.6219 | 0.5691 | 0.6855 |
| `valid` | 89 | 0.94% | 13.01% | 11.03% | **0.5056** | 0.6716 | 0.6329 | 0.7155 |
| `test` | 90 | 2.38% | 13.04% | 12.51% | **0.5458** | 0.7062 | 0.6811 | 0.7332 |
| `bolivia` | 15 | 10.33% | 27.07% | 15.86% | **0.6972** | 0.8216 | 0.7809 | 0.8667 |

### Hallazgos clave para modelado y presentación
- **Fallo físico de SAR puro en zonas áridas:** `Pakistan` (IoU `0.1518`, Precision `0.1627`) y `Somalia` (IoU `0.1506`, Precision `0.1710`) sufren falsos positivos masivos porque la arena/suelo seco plano tiene baja retrodispersión radar similar al agua. Esto justifica físicamente incorporar **HAND/Slope (Config A)** y **Sentinel-2 NDWI/MNDWI (Config B)**.
- **`NaN`s críticos en S1:** 46 chips tienen `NaN`s en `S1Hand` (incluyendo **8 de los 15 chips de Bolivia**, 10.33% de píxeles). Todo modelo debe limpiar/enmascarar `NaN`s antes de inferencia.
- **Mapeo de nombres:** en `Sen1Floods11_Metadata.geojson` el evento figura como `Cambodia` (`Colombia` no tiene chips `HandLabeled`), mientras que los archivos usan el prefijo `Mekong_*`.

---

## 2026-11-23 — Prueba de Paridad GEE (Decisión Día 3): **APROBADA (PASS)**

Ejecutado con `training/gee_parity_check.py` (proyecto `flood-mapper-mvp`) comparando `S1Hand` y `S2Hand` locales contra `COPERNICUS/S1_GRD` y `COPERNICUS/S2_HARMONIZED` (L1C) vía `ee.data.computePixels` en grilla exacta 512×512:

| Chip | S1 VV corr (MAE dB) | S1 VH corr (MAE dB) | S2 B2–B12 corr |
|---|---:|---:|---:|
| `Bolivia_195474` | **1.0000** (`0.000`) | **1.0000** (`0.000`) | **0.9954 – 0.9982** |
| `Ghana_103272` | **1.0000** (`0.000`) | **1.0000** (`0.000`) | **1.0000** |
| `Spain_1167260` | 0.7367 *(3 escenas en ±1d)* | 0.6974 *(3 escenas en ±1d)* | **0.9718 – 0.9964** |

### Decisiones derivadas
- **GEE en vivo continúa en el MVP** (Fase 3), siempre manteniendo modo offline + subida de GeoTIFF como respaldo.
- Para inferencia S2 en vivo usar siempre **`COPERNICUS/S2_HARMONIZED` (L1C)** — no `S2_SR_HARMONIZED` (L2A) — para preservar paridad exacta con `S2Hand`.
- Para DEM usar el asset vigente **`COPERNICUS/DEM/GLO30_2024_1`** (el antiguo `GLO30` figura deprecado en GEE).

---

## 2026-11-23 — Entrenamiento y Ablación de Modelos de Árboles (Fase 2)

Entrenamiento ejecutado con `training/train_trees.py` optimizando **Flood IoU** en `valid` vía Optuna, calibrando threshold en 100% de `valid` y evaluando una sola vez en `test` (90 chips) y `bolivia` (15 chips).

### Tabla Comparativa de Modelos (Registrados en SQLite `data/flood.db`)

| Modelo | Configuración | Features | Val IoU | Test IoU | Test F1 | Test Prec | Test Rec | Bolivia IoU | Tamaño |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Baseline Otsu** | S1 (VH) | 1 | 0.5056 | 0.5458 | 0.7062 | 0.6811 | 0.7332 | 0.6972 | — |
| `lightgbm_S1_only` | Solo S1 | 8 | 0.5904 | 0.6217 | 0.7667 | 0.7930 | 0.7421 | 0.6039 | **0.34 MB** |
| `lightgbm_A` | S1 + Terreno | 11 | 0.5859 | 0.6318 | 0.7744 | 0.8185 | 0.7347 | 0.5644 | **0.43 MB** |
| `rf_A` | S1 + Terreno | 11 | 0.5925 | **0.6366** | 0.7779 | 0.8256 | 0.7355 | 0.5659 | **8.88 MB** |
| `lightgbm_B` | S1 + S2 + Terreno | 21 | 0.8752 | **0.8869** | 0.9401 | 0.9490 | 0.9313 | **0.8338** | **0.38 MB** |
| `rf_B` | S1 + S2 + Terreno | 21 | 0.8832 | **0.8926** | 0.9433 | 0.9526 | 0.9341 | **0.8338** | **7.24 MB** |

### Conclusiones de la Ablación (verificadas por evento)
1. **Terreno (Config A vs Solo S1 en árboles):** sumar `Slope + HAND + DEM_rel` mejora claramente llanuras aluviales húmedas como `Paraguay` (`0.56 -> 0.65`, `+9 pts`) y sube `+1.0 a +1.5 pts` el Test global (`0.622 -> 0.637`), **pero no resuelve zonas áridas planas** (`Pakistan`: `0.16`, `Somalia`: `0.27–0.28`) porque la arena seca también yace en zonas de `HAND` bajo, y reduce `-4 pts` en `Bolivia` (`0.604 -> 0.564`) por leve sobreajuste tabular a la topografía de otros países.
2. **Salto decisivo con Sentinel-2 (Config B):** agregar Sentinel-2 L1C + `NDWI`/`MNDWI`/`AWEI` resuelve el fallo físico del SAR en zonas áridas (`Pakistan`: `0.16 -> 0.87`, `Somalia`: `0.27 -> 0.91`) y eleva el Test global a **`0.8926`** (Bolivia **`0.8338`**).
3. **Contexto espacial en DeepLabV3+ (Config A):** supera ampliamente a los árboles en Test (`0.6889`) y recupera por completo la generalización en Bolivia (`0.6949`, `+12.9 pts` sobre árboles A).
4. **Modelos ultra livianos para CPU en Space:** Random Forest pasó de los **2.0 GB** legacy a **7.2–8.9 MB** y LightGBM pesa **< 0.5 MB**. Todos quedaron empaquetados bajo `FloodModel` en `models/` y registrados en SQLite.

---

## 2026-11-23 — DeepLabV3+ Config A (Colab GPU -> ONNX) y Validación Fase 3

### DeepLabV3+ (`resnet34`, Config A: 11 canales S1 + Terreno)
Entrenado en Google Colab (GPU T4, 40 épocas, crops 256×256, `MaskedBCEDiceLoss` ignorando `-1`):
- **Valid (thr=0.475):** `IoU = 0.6773`, `F1 = 0.8076`
- **Test (90 chips):** **`IoU = 0.6889`**, `F1 = 0.8158`, `Prec = 0.8157`, `Rec = 0.8159` (**+5.2 pts** sobre `rf_A` y **+14.3 pts** sobre Otsu)
- **Bolivia (15 chips, OOD):** **`IoU = 0.6949`**, `F1 = 0.8200`, `Prec = 0.8219`, `Rec = 0.8181` (**+12.9 pts** sobre árboles en Config A)
- **Exportación ONNX:** PyTorch 2.x con `dynamo=True` separaba los pesos en `.onnx.data` (0.27 MB + 85 MB). Se re-exportó con `dynamo=False` (`opset_version=17`) obteniendo un único archivo autocontenido `deeplabv3plus_A.onnx` (**89.8 MB**), verificado con `onnxruntime` CPU.
### DeepLabV3+ (`resnet34`, Config B: 21 canales S1 + S2 + Terreno)
Entrenado en Google Colab (GPU T4, 40 épocas, re-export autocontenido `dynamo=False`, `90.0 MB`):
- **Valid (thr=0.575):** `IoU = 0.8240`, `F1 = 0.9035`
- **Test (90 chips):** `IoU = 0.8311`, `F1 = 0.9078`, `Prec = 0.9144`, `Rec = 0.9013`
- **Bolivia (15 chips, OOD):** `IoU = 0.7937`, `F1 = 0.8850`, `Prec = 0.9414`, `Rec = 0.8350`
- **Conclusión honesta DL vs Árboles:** en **Config A (SAR ruidoso)** gana claramente DeepLabV3+ (`0.6889` test / `0.6949` Bolivia vs `0.637` / `0.566` de RF A); en **Config B (con índices ópticos S2 limpios sobre 252 chips)** ganan **RF B (`0.8926`)** y **LightGBM B (`0.8869`)** con `< 8 MB` y mucha mayor velocidad en CPU.

### Validación End-to-End Fase 3 y Correcciones de Auditoría
- Aplicadas todas las correcciones de `Agent/auditoria.md` sin requerir reentrenar ningún modelo:
  - `core/inference.py`: sliding window ONNX con solapamiento (`overlap=32`) y ventana coseno.
  - `core/data.py`: clip exacto al polígono en GEE, Cloud Score+ sobre el polígono para nubes S2, `dem_rel` a escala métrica fija (~510 m), unión JRC+MERIT para agua permanente, sanitización de nombres y `GEE_PROJECT_ID` solo desde entorno.
  - `app/main.py` + `app/layers.py`: capa **Mapa de Errores (TP/FP/FN)**, ocultado botón fantasma en modo GEE, `event_id` y rutas relativas en SQLite, rotación automática de `data/outputs/`.
  - `Dockerfile` actualizado con `.[gee,ai]`, links relativos portables en `AGENTS.md`/`Agent/plan.md`, `ruff check` limpio y suite `pytest` compatible con CI.

---

<!-- Próximas entradas esperadas:
## YYYY-MM-DD — Registro CPU de DeepLabV3+ A/B y Asistente IA + Deploy HF Space (Fase 4)
-->
