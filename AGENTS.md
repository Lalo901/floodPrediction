# AGENTS.md — Flood Mapper MVP

Sistema de detección de agua/inundaciones sobre imágenes Sentinel-1 y Sentinel-2 con terreno relativo (Slope, HAND, DEM), inferencia en Streamlit y despliegue en Docker / Hugging Face Space.

## Fuente de verdad (SDD liviano)

Antes de implementar o modificar arquitectura, leer:
- **Especificación y contratos:** [Agent/spec.md](Agent/spec.md)
- **Plan por fases y estado actual:** [Agent/plan.md](Agent/plan.md)
- **Bitácora de decisiones y métricas:** [Agent/decisions.md](Agent/decisions.md)
- **Auditoría técnica y de seguridad:** [Agent/auditoria.md](Agent/auditoria.md)

## Reglas del repositorio

1. **No modificar ni versionar `dataset/` ni `_legacy/`.** Los 446 chips `HandLabeled` están en `dataset/data/flood_events/HandLabeled/` y los splits en `dataset/splits/flood_handlabeled/` (sin subcarpeta `v1.1`).
2. **Solo usar `HandLabeled` (446 chips)** en entrenamiento, validación y test. `WeaklyLabeled` queda fuera del MVP.
3. **Siempre descartar `label == -1`** en loss, entrenamiento y métricas. Limpiar/enmascarar `NaN`s de `S1Hand` antes de calcular features o predecir.
4. **Métrica objetivo:** **Flood IoU** (clase 1 = agua, acumulando `TP/FP/FN` globalmente sobre píxeles válidos). Reportar además F1, Precision y Recall global y por evento (con **Bolivia** aparte). Nunca tunear sobre `test` ni `bolivia`.
5. **Un solo entorno Python (3.11/3.12 con `uv`)** compatible con el `Dockerfile`. No crear dependencias que requieran entornos separados.
6. **Modelos livianos para CPU:** RF `< 50 MB`, DeepLabV3+ (`resnet34`) entrenado en Google Colab y exportado a **ONNX** (`onnxruntime`).
7. **Nunca versionar secretos** (claves de GEE, API keys) ni pesos pesados (`*.joblib`, `*.onnx`, `*.pt`, `*.tif`, `data/flood.db`).
8. **Rigor científico:** la app detecta **agua presente en una escena**, no "zonas inundables" ni riesgo futuro. Reproyectar siempre a CRS métrico para calcular hectáreas.

## Estructura del proyecto

- `Agent/` — `spec.md`, `plan.md`, `decisions.md`
- `app/` — Interfaz Streamlit + `streamlit-folium`
- `core/` — `db.py`, `registry.py`, `metrics.py`, `features.py`, `data.py`, `inference.py`, `postproc.py`, `analytics.py`
- `ai/` — Clientes LLM (BYOK + OpenAI-compatible) y herramientas acotadas
- `training/` — EDA, baseline Otsu, entrenamiento genérico de árboles y scripts/notebooks Colab para DeepLabV3+
- `models/` — Metadatos/manifests (pesos en Hugging Face Hub)
- `data/` — `flood.db`, stacks precargados y salidas (ignorado en git)
- `plugins/` — Extensiones opcionales (solo `APP_MODE=local`)
- `_legacy/` — Código y modelos antiguos archivados (solo referencia)
