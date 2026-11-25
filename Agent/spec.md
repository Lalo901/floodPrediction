# Flood Mapper MVP — Especificación Técnica (Spec)

> Fuente de verdad arquitectónica y técnica del proyecto. Basado en `resumen_flood.md` y verificado contra los datos reales en disco (`2026-11-23`).
> Lo marcado como **[VERIFICAR]** debe confirmarse antes de depender de ello en producción.

---

## 1. Objetivo y Alcance

Aplicación web en **Streamlit** donde el usuario **dibuja un polígono** en un mapa o elige un evento precargado, y el sistema:

1. obtiene **Sentinel-1** (y **Sentinel-2** si existe una escena útil) junto con **DEM relativo, Slope y HAND** precalculados;
2. ejecuta un **modelo entrenado offline con Sen1Floods11** seleccionado por el usuario;
3. devuelve la extensión de agua/inundación en **capas** (imagen satelital, DEM/HAND, máscara predicha y opcional ground truth), con **estadísticas en hectáreas** y exportación **raster (GeoTIFF) y vectorial (GeoPackage / GeoJSON)**;
4. integra un **asistente de IA** con herramientas acotadas que opera sobre el análisis ya calculado y guardado en SQLite.

**Principio fundamental:** La app **solo hace inferencia**. Todo el entrenamiento y tuning con Optuna ocurre offline.

**Aclaración científica obligatoria (UI y presentación):** El sistema **detecta agua presente en una escena satelital** en una fecha dada; **no** predice riesgo de zonas inundables ni reemplaza un estudio hidrológico profesional.

**Plazo:** 3 semanas máximo (15 días hábiles). Presentación de 15 minutos con demo en vivo.

---

## 2. Criterios de Aceptación del MVP

El MVP está terminado si se cumple todo lo siguiente:

- [x] Al menos **6 modelos seleccionables** (7 registrados: RF liviano, LightGBM y DeepLabV3+ ONNX en A y B + `lightgbm_S1_only` ablación), con métricas registradas en SQLite (`models`).
- [x] Un evento de Sen1Floods11 funciona de punta a punta con **Flood IoU en vivo** contra `LabelHand`.
- [x] Un polígono libre funciona (GEE en vivo aprobado en paridad Día 3 + subida de GeoTIFF + modo offline).
- [x] Exportación funcional a **GeoTIFF** y **GeoPackage / GeoJSON**.
- [x] La UI siempre muestra: fecha de captura S1 (y S2 si aplica), % de nubes (S2), desfase S1–S2, modelo y configuración usada.
- [x] Asistente de IA responde sobre el resultado ejecutando herramientas registradas en `tool_calls` *(módulo `ai/` + `app/assistant_panel.py`, Bloque 1 completado)*.
- [ ] Repo en GitHub (`main` subido) y despliegue en **Hugging Face Space (Docker)** *(Bloque 2 — Fase 4)*.

---

## 3. Decisiones Técnicas Congeladas

| Tema | Decisión |
|---|---|
| UI y Mapa | Streamlit + `streamlit-folium` (con plugin Draw) |
| Base de datos | SQLite (`data/flood.db`), gestionada por `core/db.py` y `core/registry.py` |
| Entorno Python | **Un único entorno** Python 3.11/3.12 (`uv`), replicado en `Dockerfile` |
| Familias de modelos | LightGBM, Random Forest (**< 50 MB**), DeepLabV3+ (`resnet34` exportado a **ONNX**), XGBoost (opcional / primer corte) |
| Configuraciones | **A:** S1 + Slope + HAND + DEM_rel (operativa todo-clima). **B:** S1 + S2 + Slope + HAND + DEM_rel (máxima precisión despejada) |
| Dataset entrenamiento | **Solo los 446 chips `HandLabeled`** de Sen1Floods11 (`WeaklyLabeled` fuera del MVP) |
| Métrica objetivo | **Flood IoU** (clase 1 = agua, ignorando `-1`, micro-IoU global). Reportar además F1, Precision, Recall por split y por evento |
| DEM / Slope / HAND | **Estáticos precalculados** (`COPERNICUS/DEM/GLO30_2024_1` + MERIT Hydro `hnd`), nunca calculados por chip |
| Fuente S1 / S2 en vivo | Google Earth Engine (paridad aprobada `corr=1.0000`, Cloud Score+ sobre polígono), con **modo offline** y **subida de GeoTIFF** |
| Despliegue | Docker (uid 1000 no-root, puerto `8501`, `onnxruntime` CPU, `libgomp1`). Pesos en Hugging Face Hub |
| IA (módulo `ai/`) | Activación opcional; **cualquier proveedor BYOK** (Anthropic, Gemini, OpenAI, OpenRouter, Groq, OpenAI-compatible personalizado) + **Ollama/LM Studio local** (solo `APP_MODE=local`). Herramientas acotadas, protección anti-SSRF, sin ejecución libre de código |

---

## 4. Dataset Verificado en Disco

Ubicación local (excluida de git): `dataset/` (sin carpeta `v1.1/`).

- **Imágenes y máscaras `HandLabeled`** (`dataset/data/flood_events/HandLabeled/`):
  - `S1Hand/`: 446 GeoTIFFs `(2, 512, 512)`, `float32`, `EPSG:4326`, bandas `('VV', 'VH')` en dB. **Contienen `NaN`s** que deben enmascararse/imputarse explícitamente.
  - `S2Hand/`: 446 GeoTIFFs `(13, 512, 512)`, `int16`, `EPSG:4326`, Sentinel-2 **L1C** (`B1..B12, B8A`).
  - `LabelHand/`: 446 GeoTIFFs `(1, 512, 512)`, `int16`, valores **`{-1, 0, 1}`** (`-1` = NoData / nubes, `0` = no agua, `1` = agua).
  - `JRCWaterHand/`: 446 máscaras de agua permanente JRC (`uint8`).
  - `S1OtsuLabelHand/`: 446 máscaras del baseline Otsu oficial.
- **Splits oficiales** (`dataset/splits/flood_handlabeled/`, CSV sin cabecera `S1_file,Label_file`):
  - `flood_train_data.csv`: 252 chips
  - `flood_valid_data.csv`: 89 chips
  - `flood_test_data.csv`: 90 chips
  - `flood_bolivia_data.csv`: 15 chips (evaluación out-of-distribution aparte)
  - *Nota metodológica:* el split oficial de Sen1Floods11 es por chip, no por evento (excepto Bolivia). Debe declararse en la presentación.
- **Metadatos:** `dataset/Sen1Floods11_Metadata.geojson` (fechas S1/S2, órbita, polígonos, `VH_thresh`).
- **Terreno derivado (generado en Fase 1):** `dataset/derived/TerrainHand/` con `Slope`, `HAND` (MERIT Hydro `hnd`), `DEM_rel` y `merit_wat` alineados a la grilla 512×512 de los 446 chips (`446/446`, 0 errores).

---

## 5. Arquitectura y Estructura del Repositorio

```text
floodPrediction/
├── AGENTS.md                 # Guía operativa breve para agentes
├── DOCUMENTACION.md          # Guía de uso para usuarios finales
├── Agent/
│   ├── spec.md               # Este documento (contrato técnico)
│   ├── plan.md               # Plan por fases (Fases 0 a 4) y checkpoints
│   ├── decisions.md          # Bitácora de decisiones y hallazgos
│   └── auditoria.md          # Auditoría técnica, científica y de seguridad
├── app/                      # Streamlit UI: main.py, layers.py, assistant_panel.py
├── core/
│   ├── db.py                 # Conexión y esquema SQLite
│   ├── registry.py           # Consulta tabla models y carga artefactos FloodModel
│   ├── metrics.py            # Flood IoU, F1, Precision, Recall (ignorando -1)
│   ├── features.py           # Construcción determinista de stacks Config A y B
│   ├── data.py               # Fuentes: offline, GeoTIFF subido, GEE en vivo
│   ├── inference.py          # Artefacto FloodModel: árboles por lotes / DeepLab ONNX con overlap
│   ├── postproc.py           # Agua permanente, filtro mediana, área mínima, poligonizar
│   └── analytics.py          # Hectáreas (CRS métrico), zonal stats, focos críticos, desfase S1–S2
├── ai/                       # Módulo extensible de IA (agnóstico de UI):
│   ├── security.py           # Anti-SSRF, política APP_MODE, redacción de keys, sanitización
│   ├── providers.py          # BaseLLMProvider + Anthropic, OpenAI-compat (Gemini/OpenRouter/Groq/Custom), Ollama local
│   ├── tools.py              # ToolRegistry acotado + persistencia automática en SQLite `tool_calls`
│   └── agent.py              # FloodAssistant (loop acotado de herramientas + fallback para modelos locales)
├── plugins/                  # Extensiones opcionales (solo APP_MODE=local)
├── training/                 # Scripts genéricos de EDA, baseline, terreno, árboles y Colab DL
├── models/                   # Metadatos/manifests (.json); pesos en Hugging Face Hub
├── data/                     # flood.db, stacks demo precargados, salidas (ignorado en git)
├── dataset/                  # Sen1Floods11 local (ignorado en git)
├── _legacy/                  # Código y modelos antiguos archivados (ignorado en git)
├── Dockerfile
├── pyproject.toml            # Dependencias reproducibles (uv)
└── .github/workflows/        # CI: push a Hugging Face Space
```

---

## 6. Contratos Clave

### 6.1 Contrato de Features (a congelar en Fase 1)

- **Config A — Operativa todo-clima (`S1 + terreno`):**
  - Base SAR: `VV`, `VH`, `VV - VH`, `VV / (VH + eps)` (en dB, limpiando `NaN`s).
  - Contexto espacial (para árboles): media y desviación estándar en ventana (5×5 o 7×7) sobre `VV` y `VH`.
  - Terreno relativo: `Slope` (Copernicus GLO-30), `HAND` (MERIT Hydro `hnd`), `DEM_rel` (elevación menos media/percentil local; nunca elevación absoluta).
- **Config B — Máxima precisión (`S1 + S2 + terreno`):**
  - Todo Config A + bandas S2 L1C normalizadas (`B2, B3, B4, B8, B11, B12`) + índices espectrales de agua/vegetación (`NDWI`, `MNDWI`, `NDVI`, `AWEI`).
  - **Regla operativa:** si no existe escena S2 útil (% nubes alto o desfase S1–S2 > 3 días), **Config B se bloquea en la UI con explicación clara**.

### 6.2 Artefacto `FloodModel`

Cada modelo entrenado se empaqueta junto con su contrato de preprocesamiento:
- pesos del modelo (`.joblib` comprimido < 50 MB para árboles, o `.onnx` para DeepLabV3+);
- lista y **orden exacto** de features;
- estadísticas de normalización calculadas **exclusivamente sobre `train`**;
- `threshold` óptimo calibrado en `valid` para maximizar Flood IoU;
- política de manejo de `NaN` y configuración de postproceso.

### 6.3 Esquema SQLite (`data/flood.db`)

- `models`: `id`, `name`, `family`, `config` (`A`/`B`), `artifact_path`, `feature_names_json`, `threshold`, `hyperparams_json`, `val_iou`, `test_iou`, `test_f1`, `test_precision`, `test_recall`, `bolivia_iou`, `bolivia_f1`, `trained_at`, `data_version`.
- `events`: `id`, `name`, `iso_cc`, `s1_date`, `s2_date`, `orbit`, `bbox_json`, `vh_thresh`, `source` (`sen1floods11` / `custom`).
- `analyses`: `id`, `event_id`, `polygon_geojson`, `s1_date`, `s2_date`, `cloud_pct`, `s1_s2_lag_days`, `model_id`, `flooded_ha`, `perm_water_ha`, `live_iou`, `raster_path`, `vector_path`, `created_at`.
- `tool_calls`: `id`, `analysis_id`, `tool_name`, `args_json`, `result_summary`, `created_at`.

### 6.4 Postproceso y Analítica (igual para todos los modelos)

1. Enmascarar píxeles inválidos (`NaN` en S1 o nubes en S2).
2. Restar **agua permanente** (`JRCWaterHand` / MERIT `wat`).
3. Filtro de mediana (ej. 3×3) y eliminación de polígonos menores al área mínima configurable.
4. Reproyectar geometrías/máscara a **CRS métrico** (UTM automático por centroide) para calcular **hectáreas reales** (`flooded_ha`).
5. Calcular estadísticas zonales (`HAND`, `Slope`, `DEM_rel` dentro de la mancha inundada).
6. Exportar raster (GeoTIFF) y vector (GeoPackage / GeoJSON).

### 6.5 Modos de Ejecución (`APP_MODE=local|space`)

| Función | `local` | `space` |
|---|---|---|
| Modo offline (Sen1Floods11) + subida GeoTIFF | Sí (por defecto) | Sí (por defecto) |
| GEE en vivo | Sí (auth local) | Solo con cuenta de servicio en secrets |
| IA con clave propia (BYOK: Anthropic/Gemini) | Sí (solo en sesión) | Sí (solo en sesión) |
| IA local (Ollama / LM Studio) | Sí | No |
| Plugins (`plugins/`) | Sí | No (deshabilitado con aviso) |

---

## 7. Reglas de Entrenamiento Innegociables

1. **Descartar `label == -1`** en entrenamiento, validación, cálculo de loss y métricas.
2. **Nunca tunear hiperparámetros ni threshold sobre `test` ni `bolivia`.**
3. **Submuestreo solo en `train`** (nunca en `valid`/`test`). Elegir una sola estrategia de desbalance (submuestreo **o** pesos de clase, no mezclar a ciegas).
4. **Métrica de Optuna:** Flood IoU en `valid`. Ajustar `threshold` en `valid`.
5. **Ablación obligatoria a reportar (al menos con LightGBM):** `Solo S1` vs `S1 + terreno (A)` vs `S1 + S2 + terreno (B)`, más **baseline Otsu** oficial.
6. **DeepLabV3+ en Colab:** encoder `resnet34` (ImageNet), tiles 256×256, pérdida `BCEWithLogits + Dice` enmascarando `-1`, exportar mejor checkpoint a **ONNX** para inferencia rápida en CPU.

---

## 8. Línea de Corte (orden estricto si falta tiempo)

1. XGBoost (redundante con LightGBM).
2. DeepLabV3+ en Config B.
3. Proveedor Gemini (dejar Anthropic + OpenAI-compatible).
4. Modo GEE en vivo (si falla paridad Día 3 o cuota).
5. Plugins locales.
6. Capas extra (caso argentino / OSM).

**Nunca recortar:** pipeline de datos A/B, baseline Otsu, LightGBM + RF liviano en A y B, DeepLabV3+ en A, modo offline + GeoTIFF, exportación raster/vector, despliegue Docker y asistente IA básico.

---

## 9. Pendientes por Verificar

- [x] Elegibilidad y registro del proyecto GCP (`flood-mapper-mvp`) en Earth Engine — **verificado y autenticado**.
- [x] Paridad de histogramas entre `S1Hand`/`S2Hand` y `COPERNICUS/S1_GRD` + `COPERNICUS/S2_HARMONIZED` (L1C) en GEE — **aprobada (`corr = 1.0000`)**. Asset DEM vigente: `COPERNICUS/DEM/GLO30_2024_1`.
- [ ] **[VERIFICAR]** Límites actuales de descarga de GEE (~32 MB) y del plan gratuito de Hugging Face Spaces (CPU/RAM/disco efímero).
- [ ] **[VERIFICAR]** Tiempos reales de inferencia en CPU del Space (LightGBM, RF < 50 MB y DeepLab ONNX).
- [ ] **[VERIFICAR]** Licencias de capas externas (MERIT Hydro, Copernicus GLO-30) para citar correctamente en la presentación.
