# Auditoría Técnica, Científica y de Seguridad — Flood Mapper MVP

> **Fecha:** 2026-11-23 | **Commit auditado:** `137b905` (`main`) | **Alcance:** Todos los `.md`, `core/`, `app/`, `training/`, `tests/`, `Dockerfile`, `pyproject.toml`, `.gitignore`, `.dockerignore`, `data/flood.db`, métricas generadas e historial completo de `git`.

---

## Resumen Ejecutivo (Semáforo Global)

| Área | Estado | Diagnóstico directo |
|---|---|---|
| **Seguridad y Exposición de Claves** | 🟢 Limpio (con 2 notas menores) | **0 claves privadas, 0 API keys, 0 Service Account JSON, 0 tokens en git.** Solo aparece el Project ID (`flood-mapper-mvp`, no es secreto) y 4 links `file:///home/...` a pasar a relativos. |
| **Datos y disciplina ML (Splits, `-1`, `NaN`, IoU)** | 🟢 Sólido | Cero data leakage, 446 `HandLabeled` íntegros, paridad GEE `1.0000`, `pytest` 8/8 en verde. |
| **Consistencia Docs vs Código** | 🟡 Parcial | Coherente en arquitectura general, pero hay 4 promesas de `spec.md` / `decisions.md` que el código o los números reales hoy contradicen. |
| **Producción Docker / Hugging Face Space** | 🔴 Bloqueado | El `Dockerfile` actual **no arranca con funcionalidad útil en Space** (faltan deps `gee`/`ai`, faltan modelos/manifest, falta `Metadata.geojson` y chips demo). |
| **GEE en vivo (`core/data.py`)** | 🔴 Incompleto / Riesgo | No enmascara nubes en S2, mide nubes del tile de 100 km (no del polígono), recorta por bounding box rectangular y cambia la escala física de `dem_rel`. |
| **Rigor científico para la defensa** | 🟠 Atención clave | La ablación real por evento refuta parte de la hipótesis escrita en `Agent/decisions.md` sobre `HAND/Slope` en árboles (y explica algo mucho más interesante). |

---

## 1. Hallazgos Críticos 🔴 (Bloqueantes antes de desplegar o defender)

### 1.1 El `Dockerfile` no puede arrancar la app en Hugging Face Space
- **Evidencia:**
  - `Dockerfile` (línea 24) ejecuta `RUN uv pip install --system --no-cache .` (sin extras `[gee,ai]`).
  - En `pyproject.toml`, `earthengine-api`, `anthropic` y `openai` están en dependencias opcionales.
  - `.dockerignore` excluye `dataset/`, `data/flood.db`, y los pesos (`models/*.joblib`, `models/*.onnx`) no están en GitHub.
- **Qué ocurre hoy si haces deploy al Space:**
  1. `data/flood.db` se crea **vacía** (`0` modelos, `0` eventos porque `dataset/Sen1Floods11_Metadata.geojson` no existe en la imagen).
  2. `app/main.py` detecta `not models` y se frena en pantalla roja (`st.error`).
  3. Aunque configures los secrets de GEE en el Space, `import ee` falla silenciosamente dentro de `init_gee()` porque `earthengine-api` no fue instalado.
  4. Todavía no existe un `models/manifest.json` ni un downloader automático desde Hugging Face Hub, ni los 6–8 chips demo empaquetados en `data/stacks/`.
- **Severidad:** Bloqueante para el criterio de aceptación *"Space en Hugging Face desplegado"*.

---

### 1.2 Cuatro fallos ocultos en GEE en vivo (`core/data.py`)
1. **Nubes medidas sobre 10,000 km², no sobre tu polígono:** `fetch_gee_scene` usa `CLOUDY_PIXEL_PERCENTAGE` (metadato del tile completo de Sentinel-2 de 100×100 km). Puede aceptar Config B cuando una nube cubre justo tu polígono de 10 km, o rechazarlo cuando tu polígono está despejado.
2. **Sin máscara de nubes por píxel en S2:** `Agent/spec.md` promete enmascarar nubes en Sentinel-2 en vivo, pero `fetch_gee_scene` descarga la imagen S2 cruda sin aplicar **Cloud Score+** (`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`) ni `QA60`.
3. **Recorte por Bounding Box, no por el polígono dibujado:** `fetch_gee_scene` usa `west, south, east, north = poly.bounds`. Si el usuario dibuja un polígono irregular o diagonal siguiendo un río, se procesa y se reportan hectáreas sobre todo el rectángulo exterior.
4. **Escala física inconsistente en `dem_rel`:** En entrenamiento (chips `512×512` a 10 m), `_compute_dem_rel` usa `window_px=51` (~510 m). En GEE en vivo, si un polígono mide 18 km y se limita a `max_dim_px=1024`, cada píxel mide ~18 m y `window_px=51` pasa a ser ~920 m. Debe calcularse en **metros** (`window_px = odd(round(510 / pixel_res_m))`).

---

### 1.3 Alerta Científica: tus propios números refutan una frase de `Agent/decisions.md`
En `Agent/decisions.md` quedó escrito:
> *"Sumar HAND/Slope (Config A) y Sentinel-2 NDWI/MNDWI (Config B) ataca directamente ese problema físico [en Pakistan y Somalia]."*

**Pero mira lo que realmente ocurrió por evento en `data/metrics_*.json`:**

| Modelo | Config | **Test IoU** | **Bolivia IoU** | **Pakistan** | **Somalia** | **Paraguay** |
|---|---|---:|---:|---:|---:|---:|
| Baseline Otsu | Solo S1 | 0.546 | 0.697 | 0.152 | 0.151 | 0.440 |
| `lightgbm_S1_only` | Solo S1 (8 feat) | 0.622 | **0.604** | 0.160 | **0.300** | 0.560 |
| `lightgbm_A` | S1 + Terreno (11 feat) | 0.632 (+0.010) | **0.564 (-0.040)** | **0.160** | **0.270** | **0.650 (+0.090)** |
| `rf_A` | S1 + Terreno (11 feat) | 0.637 (+0.015) | **0.566 (-0.038)** | **0.160** | **0.280** | **0.650 (+0.090)** |
| `lightgbm_B` | S1 + S2 + Terreno | 0.887 | 0.834 | **0.850** | **0.900** | 0.800 |
| `rf_B` | S1 + S2 + Terreno | **0.893** | **0.834** | **0.870** | **0.910** | **0.810** |

**Lectura crítica honesta (y excelente argumento para el jurado):**
1. **El terreno SÍ ayuda en llanuras aluviales húmedas** (`Paraguay` sube de `0.56` a `0.65`, `+9 pts`), pero **en zonas áridas planas (`Pakistan`, `Somalia`) la arena seca también está en llanuras de `HAND` y `Slope` bajos**, así que el terreno **no corrige** la confusión radar de arena vs agua en árboles. **Es Sentinel-2 (`Config B`) quien resuelve el problema (`Pakistan 0.16 -> 0.87`, `Somalia 0.27 -> 0.91`)**.
2. **En Bolivia (evento no visto), el terreno empeoró `-4 pts` a los árboles** (`0.604 -> 0.564`) por leve sobreajuste a la distribución de `HAND` de otros países; en cambio **DeepLabV3+ A** logró **`0.6949` en Bolivia** porque aprovecha contexto espacial de gran escala.
3. **Qué corregir en docs:** No afirmar que "HAND resolvió Pakistán y Somalia"; explicar el hallazgo con esta tabla — demuestra rigor científico real.

---

## 2. Hallazgos Altos 🟠 (Bugs de paridad, UX y mantenimiento)

### 2.1 Riesgo de "efecto cuadrícula" y paridad en DeepLabV3+ ONNX (`core/inference.py`)
- En Colab (`training/train_deeplab_colab.py`), la evaluación se hizo pasando el chip entero `512×512` de una vez.
- En producción (`core/inference.py`), `_predict_onnx_tiled` parte el chip en **4 tiles de `256×256` sin solapamiento** (`overlap = 0`). Las convoluciones dilatadas de DeepLabV3+ sufren efectos de borde en `x=256` e `y=256`.
- **Recomendación:** Inferencia con *sliding window* con solapamiento (ej. `tile_size=256, stride=192` con promedio en zonas solapadas). **No requiere reentrenar en Colab.**

### 2.2 Script de Colab en GitHub aún tiene el bug de `dynamo=True`
- `training/train_deeplab_colab.py` (línea 253) llama a `torch.onnx.export(...)` sin `dynamo=False`. Corregirlo en el repo para reproducibilidad futura (**no requiere reentrenar A ni B** porque ambos ya fueron exportados con `dynamo=False`).

### 2.3 Falsedad visual de `cloud_pct = 0.0%` en chips Offline (`core/data.py`)
- `load_offline_scene` devuelve `cloud_pct=0.0 if s2 is not None else None`, pese a que varios chips `HandLabeled` (Ghana 33.8% `-1`, Bolivia 27.1% `-1`) tienen nubes visibles en S2 RGB. Mostrar `N/D (Sen1Floods11)` o `% NoData/nubes (label=-1)`.

### 2.4 UX confusa en modo GEE en vivo y fuga de disco en `app/main.py`
- **Botón fantasma:** En modo `"Polígono en vivo (GEE)"`, el sidebar sigue mostrando `"🚀 Ejecutar Detección"`, pero no hace nada (`selected_scene is None`); el botón real está dentro de la pestaña principal.
- **Desacople visual entre mapa e IoU en vivo:** El mapa muestra `post.flood_mask` (sin agua permanente y filtrada por `min_area_ha`), mientras `live_iou` evalúa `post.raw_water_mask` (contra `LabelHand`, que marca toda el agua superficial). Añadir nota/tooltip explicativo en UI.
- **Disco efímero en Space:** Cada corrida escribe 3 archivos nuevos (`tif`, `gpkg`, `geojson`) en `data/outputs/` sin limpieza/rotación automática.
- **`analyses.event_id` siempre `NULL`:** `save_analysis_record` nunca pasa `event_id` y guarda rutas absolutas del disco local en vez de relativas al proyecto.

### 2.5 Agua permanente incompleta cuando `JRCWaterHand` viene vacío
- En `core/data.py`, si `JRCWaterHand` existe, descarta `merit_wat`. En `Bolivia_103757`, `JRCWaterHand` es 100% ceros mientras `merit_wat` sí detecta cauces permanentes. Usar la unión `(jrc == 1) | (merit_wat > 0.5)` o hacerla seleccionable.

---

## 3. Hallazgos Medios 🟡 (Deuda técnica, CI, Tests y Estilo)

1. **`pytest` fallará en GitHub Actions CI:** `tests/test_pipeline_e2e.py` depende de `dataset/` y `models/lightgbm_B.joblib` (fuera de git). Añadir `@pytest.mark.skipif(...)` cuando no existan.
2. **Huecos de cobertura en tests:** Faltan tests unitarios para `core/db.py`, `core/registry.py`, `FloodModel.save/load` + ONNX tiling en `core/inference.py`, `load_uploaded_geotiff` y `app/layers.py`.
3. **25 avisos de `ruff`** (20 auto-reparables con `ruff check --fix`): imports sin usar (`json`, `tempfile`, `io`, `pytest`, `FEATURE_NAMES_BY_CONFIG`), `datetime.utcfromtimestamp()` deprecado en Python 3.12 (`core/data.py`), `import pandas as pd` fuera de cabecera (`core/postproc.py`) y `except Exception: return False` silencioso en `init_gee()`.
4. **Inconsistencias menores en documentación:** `README.md` ya lista `deeplabv3plus_A` antes de registrarlo en `data/flood.db`; `lightgbm_S1_only` conviene etiquetarlo como `(Solo ablación)` en la UI; `Dockerfile` aún no se ha probado con `docker build`.

---

## 4. Ideas Creativas de Alto Impacto 💡 (Para destacar en los 15 minutos)

1. **Mapa de Errores `TP / FP / FN` en 3 colores:** Verde (`TP`), Rojo (`FP`), Amarillo (`FN`) cuando hay `LabelHand`. Mostrar `Pakistan` en Config A (rojo en arena) vs Config B (verde limpio) cuenta toda la historia física en 10 segundos.
2. **Comparación Lado a Lado ("Config A vs Config B" o "Árbol vs DeepLab"):** Botón *"Comparar con otro modelo"* con `Δ ha`, `Δ IoU` y capa de discrepancia (conecta directo con `compare_with_model` en Fase 4).
3. **Panel de Explicabilidad (XAI):** Gráfico de barras con `feature_importances_` de `lightgbm_A` y `lightgbm_B` en la pestaña de modelos.
4. **Semáforo Automático de Confiabilidad de Escena:** Badge visual arriba del mapa (🟢 Config B despejado / 🟡 Config A bajo nubes / ⚠️ Alerta de incertidumbre).
5. **"Demo Pack" liviano (~35 MB) en Hugging Face Hub:** 6 chips curados (`Bolivia`, `Pakistan`, `Somalia`, `Spain`, `Ghana`) + `Sen1Floods11_Metadata.geojson` + todos los modelos descargados por `core/bootstrap.py` al arrancar Docker.

---

## 5. Orden Sugerido de Correcciones (Antes de Fase 4)

1. Copiar `deeplabv3plus_A.onnx` y `deeplabv3plus_B.onnx` (+ `.meta.json` y `metrics_*.json`) a `models/` y registrarlos con `training/register_onnx.py` midiendo tiempo real en CPU. (**No hace falta reentrenar en Colab.**)
2. Aplicar correcciones de inferencia/app/GEE/Docker (`core/inference.py` overlap ONNX, `core/data.py` clip polígono + Cloud Score+ + `dem_rel` en metros, `app/main.py` UX y limpieza `outputs/`, `Dockerfile` con `.[gee,ai]`, `ruff check --fix`, `skipif` en CI, actualización honesta de ablación en `Agent/decisions.md`).
3. Pasar a **Fase 4** (Asistente IA + `core/bootstrap.py` con HF Hub + prueba real de `docker build` + deploy en Space).

---

## 6. Auditoría de Seguridad, Exposición de Secretos y Vulnerabilidades

Escaneo ejecutado sobre **todos los archivos versionados y el historial completo de `git log -p`**:

| Vector revisado | Resultado | Detalle |
|---|---|---|
| **Claves privadas / Service Account JSON / OAuth tokens** | 🟢 **0 expuestos** | Ningún `PRIVATE KEY`, `private_key_id`, `.json` de credenciales, `.env` ni token (`AIza...`, `sk-...`, `hf_...`, `ghp_...`) en el repo ni en el historial git. |
| **GCP Project ID (`flood-mapper-mvp`)** | 🟡 **Higiene (no es secreto)** | Aparece como valor por defecto en `core/data.py:237`, `training/extract_terrain.py:144`, `training/gee_parity_check.py:122` y `Agent/*.md`. **No permite acceso ni consumo de cuota sin credenciales**, pero por buena práctica conviene quitar el default hardcodeado y leer solo de `GEE_PROJECT_ID`. |
| **Rutas locales con usuario (`/home/farfaneduardo/...`)** | 🟡 **Higiene / Privacidad** | Presentes en 4 enlaces `file:///` de `AGENTS.md` (L8–10) y `Agent/plan.md` (L3). Cambiar a enlaces relativos (`Agent/spec.md`) para no exponer el usuario local y para que funcionen en GitHub. |
| **Inyección SQL (`core/db.py`, `registry.py`, `analytics.py`, `app/main.py`)** | 🟢 **Seguro** | El 100% de las consultas SQL usan placeholders parametrizados (`?`). |
| **Path Traversal en subida de GeoTIFF (`app/main.py`, `core/data.py`)** | 🟡 **Bajo** | `Path(filename).stem` neutraliza `/` y `..`, pero conviene sanitizar `scene_id` con `re.sub(r"[^A-Za-z0-9_-]", "_", ...)` antes de escribir en `data/outputs/`. |
| **Deserialización `joblib.load` (`core/inference.py`)** | 🟡 **Riesgo controlado** | `joblib` usa `pickle`. Regla obligatoria: cargar **solo** artefactos propios confiables (nunca permitir que usuarios suban `.joblib`). |
| **`enableCORS=false` y `enableXsrfProtection=false` (`Dockerfile`)** | 🟡 **Tradeoff aceptable** | Necesario para que `st.file_uploader` no arroje 403 dentro del iframe de Hugging Face Spaces; aceptable porque la app no tiene sesiones autenticadas por cookies. En Fase 4 mantener API keys BYOK **solo en `st.session_state`** (nunca en SQLite ni logs). |
