# Flood Mapper MVP — Plan de Ejecución por Fases

> Tablero vivo de tareas. Referencia técnica en [spec.md](file:///home/farfaneduardo/Documentos/floodPrediction/Agent/spec.md).
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
- [ ] Subir a GitHub y desplegar Space "hola mundo" en Hugging Face
- [x] Registrar proyecto GCP (`flood-mapper-mvp`) en Earth Engine y probar autenticación local

**Entregable:** Repo limpio en GitHub, Space vivo en Docker, DB `data/flood.db` con eventos cargados.

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
- [x] En paralelo en **Google Colab (GPU)**: entrenar **DeepLabV3+ (`resnet34`)** Config A (`Test IoU 0.6889`, `Bolivia IoU 0.6949`, `deeplabv3plus_A.onnx` autocontenido `89.8 MB`) *(pendiente copiar a `models/` y registrar con `training/register_onnx.py`)*
- [ ] *(Opcional / corte #2)* Entrenar **DeepLabV3+** Config B en Colab y exportar a ONNX *(en curso en Colab)*

**Entregable:** Al menos 5–6 modelos entrenados, livianos (< 50 MB / ONNX) y registrados en `data/flood.db`.

---

## Fase 3 — Motor de Inferencia y App Streamlit (Días 7–10)

**Objetivo:** Pipeline de inferencia end-to-end validado contra ground truth y app usable primero en modo offline.

- [x] Implementar `core/postproc.py` (enmascarado de `NaN`, resta de agua permanente, filtro mediana, área mínima, poligonización a GeoPackage/GeoJSON)
- [x] Implementar `core/analytics.py` (reproyección automática a UTM para **hectáreas exactas**, zonal stats de `HAND`/`Slope`/`DEM_rel`, desfase S1–S2 y bloqueo Config B)
- [x] Implementar `core/data.py` unificado (`SceneBundle`: Offline Sen1Floods11, Subida GeoTIFF y GEE en vivo)
- [x] Validar pipeline completo (`data -> features -> inference -> postproc -> analytics -> export`) con `pytest` (`8 passed`, `Bolivia_103757` IoU `0.9037`)
- [x] Construir `app/main.py` + `app/layers.py` en Streamlit + `streamlit-folium`:
  - [x] Modo **offline** (selector de evento/split Sen1Floods11, selector de modelo, capas conmutables, metadatos obligatorios siempre visibles, Flood IoU en vivo)
  - [x] Subida de GeoTIFF propio
  - [x] Descarga de GeoTIFF, GeoJSON y GeoPackage (`.gpkg`)
  - [x] Modo **GEE en vivo** con dibujo de polígono (límite 20 km) y bloqueo explicativo de Config B si no hay S2 útil (nubes > 30% o desfase > 3 días)

> **🚦 Punto de Decisión — Día 10: ADELANTADO Y CUMPLIDO**
> Pipeline end-to-end y app Streamlit (offline + upload + GEE en vivo) listos y testeados.

**Entregable:** App Streamlit completamente funcional en local. **(Fase 3 COMPLETADA)**

---

## Fase 4 — Asistente IA, Despliegue en Space y Demo (Días 11–15)

**Objetivo:** Asistente con herramientas acotadas, despliegue reproducible en Hugging Face Space y defensa ensayada.

- [ ] Implementar `ai/` con 1 proveedor API BYOK (Anthropic) + endpoint compatible OpenAI (Ollama local); *(Gemini opcional / corte #3)*
- [ ] Implementar herramientas acotadas sobre análisis guardados en SQLite (`summary`, `area_by_class`, `zonal_stats`, `filter_min_area`, `compare_with_model`, `export`) con log en `tool_calls`
- [ ] Aviso de privacidad cuando se usa proveedor externo; nunca loguear ni persistir API keys
- [ ] Subir pesos finales a Hugging Face Hub y cargar en arranque / bajo demanda
- [ ] Configurar GitHub Actions -> Hugging Face Space y secrets del Space
- [ ] Medir tiempos reales de inferencia y RAM en CPU del Space gratuito

> **🚦 Punto de Decisión — Día 13:**
> ¿El Space despliega y responde en tiempos aceptables?
> - Si hay bloqueo del Space, congelar demo local en Docker y grabar respaldo.

- [ ] Generar tabla comparativa final de métricas (Otsu, LightGBM, RF, DeepLab, ablación, Bolivia)
- [ ] Preparar guion de 15 minutos y grabar video de respaldo completo

**Entregable:** MVP desplegado, métricas finales verificadas y presentación lista.
