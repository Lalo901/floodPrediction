# Flood Mapper MVP — Plan de Ejecución por Fases

> Tablero vivo de tareas. Referencia técnica en [spec.md](spec.md).
> Plazo: **15 días hábiles** (3 semanas). Aplicar la línea de corte sin dudar en los puntos de decisión.

---

## Fase 0 — Limpieza, Estructura y Cimientos (Día 1)

**Objetivo:** Repositorio limpio, un solo entorno reproducible, DB inicializada y Space "hola mundo" arriba.

- [x] Archivar código/modelos heredados en `_legacy/` sin tocar `dataset/`
- [x] Crear estructura limpia (`Agent/`, `app/`, `core/`, `ai/`, `training/`, `models/`, `data/`, `plugins/`)
- [x] Crear `.gitignore` estricto y `.dockerignore`
- [x] Crear SDD liviano (`Agent/spec.md`, `Agent/plan.md`, `Agent/decisions.md`) y actualizar `AGENTS.md`
- [x] Inicializar `git` y verificar que `dataset/` y `_legacy/` no entran en el tracking
- [x] Crear entorno único Python 3.12 con `uv` (`pyproject.toml` + `.venv` instalado y verificado)
- [x] Implementar `core/db.py` (esquema SQLite) y poblar tabla `events` desde `dataset/Sen1Floods11_Metadata.geojson` (12 eventos)
- [x] Crear `Dockerfile` + `README.md` (front matter HF Space) + `app/main.py` ("hola mundo" Streamlit)
- [x] Subir repo limpio a GitHub (`origin/main`) *(deploy de Space unificado en Fase 4 Bloque 2)*
- [x] Registrar proyecto GCP (`flood-mapper-mvp`) en Earth Engine y probar autenticación local

**Entregable:** Repo limpio en GitHub, DB `data/flood.db` con eventos cargados. **(Fase 0 COMPLETADA; deploy Space pasa a Fase 4 Bloque 2)**

---

## Fase 1 — EDA, Baseline Real, Paridad GEE y Stacks A/B (Días 2–3)

**Objetivo:** Conocer los datos reales, fijar el piso de IoU con Otsu, decidir si GEE en vivo entra y dejar los stacks listos.

- [x] Implementar `core/metrics.py` (Flood IoU, F1, Precision, Recall ignorando `-1`, global y por evento) + tests unitarios (`4 passed`)
- [x] Correr `training/eda_and_baseline.py` sobre los 446 `HandLabeled`: 0 archivos faltantes, conteo de `NaN`, `% de -1` y `% de agua` por split y evento
- [x] Evaluar baseline oficial `S1OtsuLabelHand` vs `LabelHand` en `valid` (IoU `0.5056`), `test` (IoU `0.5458`) y `bolivia` (IoU `0.6972`), registrado en `Agent/decisions.md`
- [x] **Prueba de paridad GEE:** **APROBADA** (`corr = 1.0000` en S1 y `0.995–1.000` en S2 L1C `S2_HARMONIZED`) -> GEE en vivo sigue en el MVP
- [x] Extraer una sola vez `Slope` (`GLO30_2024_1`), `HAND` (`MERIT Hydro`), `DEM_rel` y `merit_wat` alineados a los 446 chips en `dataset/derived/TerrainHand/` (`446/446`, `0 errors`, 403 MB)
- [x] Implementar `core/features.py` y congelar orden exacto de features para `S1_only` (8), `Config A` (11) y `Config B` (21) + tests unitarios (`6 passed`)

> **🚦 Punto de Decisión — Día 3: RESUELTO (PASS)**
> GEE pasó paridad exacta con `COPERNICUS/S1_GRD` y `COPERNICUS/S2_HARMONIZED` (L1C). GEE en vivo entra en Fase 3 (después del modo offline).

**Entregable:** Baseline Otsu medido, terreno precalculado para los 446 chips, `core/features.py` congelado. **(Fase 1 COMPLETADA)**

---

## Fase 2 — Entrenamiento y Evaluación Disciplinada (Días 4–6)

**Objetivo:** Entrenar modelos livianos y trazables, empaquetados como `FloodModel`, con métricas en SQLite.

- [x] Implementar artefacto `FloodModel` (`core/inference.py` + `core/registry.py`) con preprocesamiento y `threshold` incluidos
- [x] Escribir script genérico `training/train_trees.py` (submuestreo solo en `train`, Optuna optimizando Flood IoU en `valid`, calibración de `threshold` en 100% `valid`, evaluación única en `test` y `bolivia`)
- [x] Entrenar y registrar **LightGBM** (ablación `S1_only` IoU 0.6217, `Config A` IoU 0.6318, `Config B` IoU 0.8869)
- [x] Entrenar y registrar **Random Forest liviano (< 50 MB)** (`Config A` 8.88 MB / IoU 0.6366, `Config B` 7.24 MB / IoU 0.8926)
- [ ] *(Opcional / corte #1)* Entrenar **XGBoost** (Config A y Config B)
- [x] En paralelo en **Google Colab (GPU)**: entrenar **DeepLabV3+ (`resnet34`)** Config A (`Test IoU 0.6889`, `Bolivia IoU 0.6949`, `85.68 MiB` ONNX, **`0.79s` CPU**, registrado en SQLite id=6)
- [x] *(Opcional / corte #2)* Entrenar **DeepLabV3+** Config B en Colab (`Test IoU 0.8311`, `Bolivia IoU 0.7937`, `85.80 MiB` ONNX, **`0.76s` CPU**, registrado en SQLite id=7)

**Entregable:** 7 modelos entrenados, livianos (`0.34–85.8 MiB`), probados en CPU (`< 0.8s`) y registrados en `data/flood.db`. **(Fase 2 COMPLETADA)**

---

## Fase 3 — Motor de Inferencia y App Streamlit (Días 7–10)

**Objetivo:** Pipeline de inferencia end-to-end validado contra ground truth y app usable primero en modo offline.

- [x] Implementar `core/postproc.py` (enmascarado de `NaN`, resta de agua permanente, filtro mediana, área mínima, poligonización a GeoPackage/GeoJSON)
- [x] Implementar `core/analytics.py` (reproyección automática a UTM para **hectáreas exactas**, zonal stats de `HAND`/`Slope`/`DEM_rel`, desfase S1–S2 y bloqueo Config B)
- [x] Implementar `core/data.py` unificado (`SceneBundle`: Offline Sen1Floods11, Subida GeoTIFF y GEE en vivo)
- [x] Validar pipeline completo (`data -> features -> inference -> postproc -> analytics -> export`) con `pytest` (`12 passed`, `ruff` 0 avisos)
- [x] Construir `app/main.py` + `app/layers.py` en Streamlit + `streamlit-folium`:
  - [x] Modo **offline** (selector de evento/split Sen1Floods11, selector de modelo, capas conmutables, metadatos obligatorios siempre visibles, Flood IoU en vivo)
  - [x] Subida de GeoTIFF propio
  - [x] Descarga de GeoTIFF, GeoJSON y GeoPackage (`.gpkg`)
  - [x] Modo **GEE en vivo** con dibujo de polígono (límite 20 km con validación instantánea y escala métrica), clip exacto, Cloud Score+ y guardado offline en `data/stacks/`
  - [x] Mejoras operativas para desastres: paleta azul Copernicus EMS, capas `NDWI`/`MNDWI`/`Probabilidad`/`HAND < 2 m`/`Slope`, **Comparación de 2 modelos**, **Semáforo de Confiabilidad**, **Top 5 Focos Críticos** y guía de usuario (`DOCUMENTACION.md`)

> **🚦 Punto de Decisión — Día 10: ADELANTADO Y CUMPLIDO**
> Pipeline end-to-end y app Streamlit (offline + upload + GEE en vivo) listos y testeados.

**Entregable:** App Streamlit completamente funcional en local. **(Fase 3 COMPLETADA)**

---

## Fase 4 — Asistente IA, Despliegue en Space y Demo (Días 11–15)

**Objetivo:** Asistente con herramientas acotadas, despliegue reproducible en Hugging Face Space y defensa ensayada.

- [x] Implementar módulo desacoplado `ai/` (`security.py`, `providers.py`, `tools.py`, `agent.py`) + `app/assistant_panel.py` acoplado a la derecha con scroll independiente, pre-configuración previa a detección, selector de imagen procesada (<=512 px), soporte multi-proveedor BYOK y Ollama/LM Studio local (`disable_thinking` para Gemma 4, bloqueado en `APP_MODE=space`)
- [x] Implementar 7 herramientas acotadas en whitelist (`get_analysis_summary`, `get_top_hotspots`, `get_zonal_stats`, `filter_by_min_area`, `compare_models`, `list_registered_models`, `get_export_info`) con log en SQLite `tool_calls` y fallback automático (texto + visión)
- [x] Rediseño UX GIS de escritorio sin iconos (`app/main.py`, `app/plugins_panel.py`, `core/exports_gis.py`, `core/geocoding.py`, `plugins/manager.py`): mapa base siempre visible desde inicio (`Satélite | Calles | Relieve | Oscuro`), buscador país/ciudad/coords, LayerControl compacto con scroll, opacidad rápida, enfoque de Focos #1–#5, menús desplegables uniformes (`Herramientas`, `Complementos`, `Ayuda`), conectores QGIS/MapLibre/Buffer UTM/Reporte `.md`, administrador open-source con lista blanca (`APP_MODE=local`) e `Historial` clicable con `Cargar en el Mapa` sin duplicar SQLite
- [x] Controles de seguridad y pruebas: anti-SSRF, path-traversal guard en descargas del historial, lista blanca estricta de plugins, API keys solo en memoria de sesión + redacción en errores, caché de overlays/buffers, `pytest` (`26 passed`) y `ruff` (`0` avisos)
- [ ] Subir pesos finales + Demo Pack a Hugging Face Hub y cargar en arranque (`core/bootstrap.py`)
- [ ] Configurar GitHub Actions -> Hugging Face Space y secrets del Space
- [ ] Medir tiempos reales de inferencia y RAM en CPU del Space gratuito

> **🚦 Punto de Decisión — Día 13:**
> ¿El Space despliega y responde en tiempos aceptables?
> - Si hay bloqueo del Space, congelar demo local en Docker y grabar respaldo.

- [ ] Generar tabla comparativa final de métricas (Otsu, LightGBM, RF, DeepLab, ablación, Bolivia)
- [ ] Preparar guion de 15 minutos y grabar video de respaldo completo

**Entregable:** MVP desplegado, métricas finales verificadas y presentación lista.
