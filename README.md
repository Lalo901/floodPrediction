---
title: Flood Mapper MVP
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
---

# 🌊 Flood Mapper MVP

Sistema de detección de **agua superficial e inundaciones** sobre imágenes **Sentinel-1 (SAR)** y **Sentinel-2 (óptico L1C)** combinadas con variables de terreno relativo (**Slope**, **HAND**, **DEM relativo**), entrenado offline sobre los 446 chips `HandLabeled` de **Sen1Floods11** y desplegable con **Streamlit + Docker / Hugging Face Space**.

> **Nota científica:** La aplicación detecta **agua superficial presente** en una escena satelital en una fecha dada; **no** predice zonas de riesgo futuro ni reemplaza un análisis hidrológico profesional.

---

## Resultados Verificados (Sen1Floods11 `HandLabeled`, ignorando `label == -1`)

Métrica principal: **Flood IoU** (índice Jaccard de la clase agua acumulado globalmente). Todos los hiperparámetros y umbrales se calibraron exclusivamente en `valid` (89 chips) y se evaluaron una sola vez en `test` (90 chips) y en `Bolivia` (15 chips, evento geográfico no visto).

| Modelo | Configuración | Features | Val IoU | **Test IoU** | Test F1 | **Bolivia IoU** | Peso |
|---|---|---:|---:|---:|---:|---:|---:|
| **Baseline Otsu** (oficial S1) | Solo S1 (VH) | 1 | 0.5056 | 0.5458 | 0.7062 | 0.6972 | — |
| `lightgbm_S1_only` | Solo S1 (ablación) | 8 | 0.5904 | 0.6217 | 0.7667 | 0.6039 | 0.34 MB |
| `lightgbm_A` | **Config A** (S1 + Terreno) | 11 | 0.5859 | 0.6318 | 0.7744 | 0.5644 | 0.43 MB |
| `rf_A` | **Config A** (S1 + Terreno) | 11 | 0.5925 | 0.6366 | 0.7779 | 0.5659 | 8.88 MB |
| **`deeplabv3plus_A`** (`resnet34` ONNX) | **Config A** (S1 + Terreno) | 11 | **0.6773** | **0.6889** | **0.8158** | **0.6949** | 89.8 MB |
| `deeplabv3plus_B` (`resnet34` ONNX) | **Config B** (S1 + S2 + Terreno) | 21 | 0.8240 | 0.8311 | 0.9078 | 0.7937 | 90.0 MB |
| `lightgbm_B` | **Config B** (S1 + S2 + Terreno) | 21 | 0.8752 | 0.8869 | 0.9401 | 0.8338 | 0.38 MB |
| **`rf_B`** | **Config B** (S1 + S2 + Terreno) | 21 | **0.8832** | **0.8926** | **0.9433** | **0.8338** | 7.24 MB |

- **Config A (operativa todo-clima):** funciona bajo nubes densas usando radar SAR + terreno (`COPERNICUS/DEM/GLO30_2024_1` + `MERIT Hydro hnd`). Aquí **`DeepLabV3+ A`** lidera claramente (`+5.2` pts en Test y `+12.9` pts en Bolivia sobre árboles) porque aprovecha el contexto espacial frente al ruido SAR.
- **Config B (máxima precisión despejada):** incorpora 6 bandas Sentinel-2 L1C + índices `NDWI`, `MNDWI`, `NDVI`, `AWEI`, resolviendo zonas áridas (`Pakistan`: `0.16 → 0.87`, `Somalia`: `0.27 → 0.91`). Con solo 252 chips y señal óptica muy limpia, **`RF B` / `LightGBM B`** superan a `DeepLabV3+ B` con una fracción del tamaño (`0.38–7.2 MB` vs `90 MB`). En la app Config B se bloquea automáticamente si las nubes dentro del polígono superan 30% o el desfase S1–S2 supera 3 días.

---

## Características de la App

- **3 fuentes de datos unificadas ([`core/data.py`](core/data.py)):**
  - **Modo Offline:** chips `HandLabeled` de Sen1Floods11 con cálculo de **Flood IoU en vivo** contra `LabelHand`.
  - **Subida de GeoTIFF:** S1 (VV, VH en dB) + opcional S2 y enriquecimiento automático de terreno vía GEE.
  - **Polígono en vivo (Google Earth Engine):** dibujo en mapa `streamlit-folium` (máx. 20 km/lado), consulta de escenas S1/S2 con paridad verificada (`corr = 1.0000`).
- **Postproceso uniforme y analítica rigurosa ([`core/postproc.py`](core/postproc.py), [`core/analytics.py`](core/analytics.py)):**
  - Separación de agua permanente (JRC / MERIT `wat`), filtro de mediana, área mínima configurable.
  - Cálculo exacto de **hectáreas en CRS métrico UTM** y estadísticas zonales (`Slope`, `HAND`, `DEM_rel`).
  - Exportación **GeoTIFF**, **GeoJSON** y **GeoPackage (`.gpkg`)** + registro en **SQLite (`data/flood.db`)**.

---

## Instalación y Ejecución Local

Requiere Python 3.11 o 3.12 y [`uv`](https://docs.astral.sh/uv/):

```bash
# 1. Crear entorno e instalar dependencias
uv venv --python 3.12
uv pip install -e ".[train,gee,ai,dev]"

# 2. Configurar variables de entorno (opcional para GEE en vivo)
cp .env.example .env

# 3. Inicializar base de datos SQLite con metadatos de eventos
.venv/bin/python -m core.db

# 4. Ejecutar tests unitarios y de integración
.venv/bin/pytest -v

# 5. Lanzar la aplicación Streamlit
APP_MODE=local .venv/bin/streamlit run app/main.py
```

## Ejecución con Docker

```bash
docker build -t flood-mapper .
docker run -p 8501:8501 -e APP_MODE=local flood-mapper
```

---

## Estructura del Repositorio

```text
├── AGENTS.md                 # Reglas operativas del repositorio
├── Agent/
│   ├── spec.md               # Especificación técnica y contratos
│   ├── plan.md               # Plan por fases y estado actual
│   └── decisions.md          # Bitácora de decisiones y resultados
├── app/                      # Interfaz Streamlit + streamlit-folium
├── core/                     # db, registry, metrics, features, data, inference, postproc, analytics
├── ai/                       # Asistente IA con herramientas acotadas (Fase 4)
├── training/                 # EDA, baseline Otsu, paridad GEE, terreno, árboles y script Colab DL
├── tests/                    # Suite pytest (unitarios + integración E2E)
├── Dockerfile
└── pyproject.toml
```

> Los pesos entrenados (`*.joblib`, `*.onnx`), el dataset `Sen1Floods11` y los secretos (`.env`, claves GEE) están excluidos de git vía `.gitignore`.

---

## Dataset y Atribución

Los modelos fueron entrenados y evaluados exclusivamente sobre los 446 chips etiquetados manualmente (`HandLabeled`) de **Sen1Floods11**:

- **Autores / Paper:** Derrick Bonafilia, Beth Tellman, Tyler Anderson y Erica Issenberg (2020). *"Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1"*, **IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW 2020)**, pp. 835–845.
  - Paper oficial: <https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html>
- **Repositorio oficial e instrucciones de descarga:** <https://github.com/cloudtostreet/Sen1Floods11>
- **Descarga directa (Google Cloud Storage público):** `gs://sen1floods11` (vía `gsutil -m rsync -r gs://sen1floods11/v1.1/ ./dataset/`)
- **Capas auxiliares de terreno y agua permanente:** Copernicus DEM GLO-30 (`COPERNICUS/DEM/GLO30_2024_1`, ESA) y MERIT Hydro v1.0.1 (`MERIT/Hydro/v1_0_1`, Yamazaki et al., 2019), consultadas vía Google Earth Engine.
