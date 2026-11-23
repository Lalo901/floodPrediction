# Guía de Uso — Flood Mapper MVP (`DOCUMENTACION.md`)

Manual práctico para operar la aplicación web **Flood Mapper MVP** (Streamlit), interpretar sus mapas y métricas, y exportar resultados geoespaciales.

> **Alcance científico importante:** Esta herramienta detecta **agua superficial presente en una fecha satelital específica**. **No** predice riesgo de inundación futuro ni reemplaza un estudio hidrológico o hidráulico profesional.

---

## Vista general de la interfaz

La pantalla se divide en dos zonas:
- **Panel izquierdo (Sidebar):** configuración paso a paso (**1. Fuente de Datos → 2. Modelo → 3. Postproceso**).
- **Panel principal:** 3 pestañas (**🗺️ Mapa y Resultados**, **📊 Comparativa de Modelos** y **🕒 Historial (SQLite)**).

---

## 1. Fuente de Datos (Panel izquierdo)

Permite elegir de dónde provienen las imágenes satelitales a analizar:

1. **Offline (Sen1Floods11):**
   - Carga escenas reales preprocesadas del benchmark internacional Sen1Floods11 (chips de 512×512 píxeles a ~10 m).
   - Permite elegir el **Split** (`bolivia`, `test`, `valid` o `demo`), filtrar por país/evento y seleccionar un chip concreto.
   - **Ventaja:** incluye etiqueta real verificada manualmente (`LabelHand`), por lo que la app calcula el **Flood IoU en vivo** y el **Mapa de Errores** al instante.
2. **Subir GeoTIFF:**
   - Permite analizar tus propios recortes satelitales:
     - **Sentinel-1 (obligatorio):** archivo `.tif` de 2 bandas (`VV` y `VH` en decibelios, dB).
     - **Sentinel-2 (opcional):** archivo `.tif` de 6 bandas (`B2, B3, B4, B8, B11, B12`) o 13 bandas L1C.
     - **Descargar Slope/HAND desde GEE (opcional):** si Earth Engine está activo, descarga automáticamente el terreno relativo alineado a tu GeoTIFF para habilitar modelos Config A y B.
3. **Polígono en vivo (GEE):** *(disponible cuando Google Earth Engine está autenticado)*
   - En la pestaña principal aparece un mapa interactivo con herramienta de dibujo.
   - Dibuja un polígono de interés (**máximo 20 km por lado**), selecciona un rango de fechas y pulsa **🔎 Buscar escenas Sentinel-1 en GEE**.
   - Elige una fecha/órbita y pulsa **⬇️ Descargar recorte GEE y ejecutar modelo**. El sistema recorta exactamente al polígono dibujado, descarga `Slope`/`HAND`/`DEM_rel` y busca la escena Sentinel-2 más cercana (±3 días) aplicando máscara de nubes **Cloud Score+** dentro de tu polígono.

---

## 2. Modelo (Panel izquierdo)

Todos los modelos fueron entrenados previamente offline; la app solo ejecuta inferencia. Se agrupan según los sensores que necesitan:

| Configuración | Sensores usados | Cuándo usarla | Mejor modelo recomendado |
|---|---|---|---|
| **Config A** | Sentinel-1 (SAR) + Terreno (`Slope`, `HAND`, `DEM_rel`) | **Con nubes, tormentas o de noche** (el radar atraviesa nubes). | **`deeplabv3plus_A`** *(mejor contexto espacial y generalización)* |
| **Config B** | Sentinel-1 + Sentinel-2 (óptico + `NDWI`/`MNDWI`/`NDVI`/`AWEI`) + Terreno | **Cielo despejado** (máxima precisión, ideal en zonas áridas donde el radar confunde arena con agua). | **`rf_B`** o **`lightgbm_B`** *(Test IoU ~0.89, muy rápidos)* |
| **Solo ablación** | Solo Sentinel-1 (sin terreno ni S2) | Referencia científica para medir cuánto aporta el terreno. | `lightgbm_S1_only` |

- **Bloqueo automático de Config B:** si la escena no tiene Sentinel-2 disponible, tiene **> 30% de nubes sobre el polígono**, o el desfase entre Sentinel-1 y Sentinel-2 supera **3 días**, la app bloquea Config B y explica el motivo para evitar resultados engañosos (en modo GEE en vivo cambia automáticamente a Config A).
- **⚖️ Comparar con un segundo modelo:** activa este checkbox en el sidebar para ejecutar **dos modelos simultáneamente** sobre la misma escena (ej. `deeplabv3plus_A` vs `rf_B`), obteniendo tabla comparativa (`Δ ha`, `Δ Flood IoU`, tiempos CPU) y una capa de **Consenso / Discrepancia** en el mapa.

---

## 3. Postproceso (Panel izquierdo)

Se aplica **exactamente igual a todos los modelos** para limpiar la máscara cruda y convertirla en polígonos utilizables en SIG:

- **Restar agua permanente (JRC + MERIT):** separa cuerpos de agua permanentes (ríos, lagos, embalses) de la inundación temporal detectada.
- **Filtro mediana (`1`, `3` o `5` px):** suaviza píxeles aislados ("ruido sal y pimienta") sin desplazar los bordes principales. `3 px` es el valor equilibrado recomendado; `1` desactiva el filtro.
- **Área mínima de polígono (ha):** descarta manchas muy pequeñas (por defecto `0.10 ha`, equivalente a ~10 píxeles de 10×10 m). El área se calcula reproyectando automáticamente a coordenadas métricas **UTM**.

---

## 4. Pestaña "🗺️ Mapa y Resultados"

### Semáforo de Confiabilidad y Barra de Metadatos
- **Semáforo operativo:** clasifica la confiabilidad de la escena (🟢 Alta confiabilidad en Config B despejado / 🟡 Modo operativo todo-clima en Config A / ⚠️ Alerta de incertidumbre si >12% del área presenta probabilidades intermedias `0.35–0.65`).
- **Barra de metadatos obligatoria:** siempre muestra escena, modelo y configuración, fecha y órbita Sentinel-1, fecha Sentinel-2, `%` de nubes sobre el polígono, desfase S1–S2 en días y tiempo de ejecución en CPU.

### Tarjetas de indicadores (KPIs)
- **Inundación (ha):** superficie total de agua detectada tras postproceso (en hectáreas métricas UTM), con el desglose de **Alta confianza (`P ≥ 80%`)** vs **Confianza moderada**.
- **Agua permanente (ha):** superficie de cauces/lagos permanentes dentro del área analizada.
- **% Área válida inundada:** porcentaje respecto a los píxeles con dato válido (excluyendo bordes `NaN` o zonas fuera del polígono).
- **Focos / Polígonos:** cantidad de parches/polígonos de inundación resultantes.
- **Flood IoU (en vivo)** *(solo cuando existe etiqueta `LabelHand`)*: índice Jaccard del agua ($\frac{TP}{TP+FP+FN}$) y F1 calculados en el acto contra la verdad terreno (ignorando píxeles `-1`). *Nota: como `LabelHand` marca toda el agua superficial, esta métrica evalúa el agua total detectada (incluyendo agua permanente), por lo que no varía al activar/desactivar "Restar agua permanente".*

### Capas del mapa interactivo (control arriba a la derecha)
Puedes encender y apagar cada capa individualmente:
- **Satélite Base (Esri) / OpenStreetMap:** mapa de referencia (con barra de escala métrica en la esquina inferior izquierda).
- **Sentinel-1 (Falso Color SAR):** composición `R=VV, G=VH, B=VV-VH`. Las zonas oscuras suelen corresponder a agua calma.
- **Sentinel-2 (RGB Natural):** vista óptica en color verdadero (si la escena dispone de S2).
- **Índice de Agua MNDWI / NDWI (S2):** índices espectrales de agua en paleta divergente (marrón = suelo seco, blanco = neutro, azul = agua) para auditoría visual directa.
- **HAND (m, MERIT Hydro):** altura sobre el drenaje más cercano (azul = zonas bajas; marrón = terreno elevado).
- **Zonas Bajas Topográficas (`HAND < 2 m`)** y **Pendiente del Terreno (`Slope °`)**: contexto topográfico estático derivado del DEM.
- **Agua Permanente (🟦 Azul marino `#0F3D91`):** cauces y cuerpos de agua permanentes (JRC + MERIT).
- **Inundación Detectada (🩵 Azul cian `#00B4FF`):** máscara final de agua nueva tras postproceso.
- **Probabilidad / Confianza:** mapa continuo `0–100%` para distinguir núcleos seguros de bordes inciertos.
- **Comparación (M1 vs M2)** *(cuando está activo el modo comparación)*: **Azul** = Consenso (ambos detectan agua), **Naranja** = Solo Modelo 1, **Magenta** = Solo Modelo 2.
- **Ground Truth (`LabelHand`, verde)** y **Mapa de Errores** *(solo en modo offline)*:
  - 🟢 **Verde (`TP`):** agua correctamente detectada.
  - 🔴 **Rojo (`FP`):** falsa alarma (predijo agua donde había tierra seca).
  - 🟡 **Amarillo (`FN`):** inundación omitida (había agua real que el modelo no detectó).

### Panel de Priorización (Top 5 Focos Críticos) y Estadísticas Zonales
- **📍 Top 5 Focos Críticos:** lista los 5 polígonos de inundación más grandes con su superficie en hectáreas, coordenadas exactas del centroide (`lat, lon`), marcadores naranjas en el mapa y enlace directo a **Google Maps**.
- **⛰️ Estadísticas Zonales:** resumen del terreno **únicamente dentro de la zona inundada detectada** (`mean`, `median`, `p90`, `max`) para `slope_deg`, `hand_m` y `dem_rel_m`.

### Exportación y Guardado Offline
- **GeoTIFF (`.tif`):** raster de 2 bandas comprimido (`banda 1 = flood_mask` binaria `0/1`, `banda 2 = water_proba` probabilidad `0..1`), listo para QGIS o ArcGIS.
- **GeoJSON (`.geojson`):** polígonos vectoriales en `EPSG:4326` con atributo `area_ha`.
- **GeoPackage (`.gpkg`):** formato vectorial estándar moderno para QGIS con atributo `area_ha`.
- **💾 Guardar recorte en `data/stacks/`:** disponible en modo GEE en vivo o GeoTIFF subido para almacenar la escena y reabrirla luego sin conexión desde **Offline → Split `demo`**.

---

## 5. Pestaña "📊 Comparativa de Modelos"

Muestra la tabla oficial registrada en SQLite (`models`) con las métricas obtenidas durante la evaluación offline:
- **`Test IoU` (columna principal de referencia):** desempeño sobre 90 chips nunca vistos en entrenamiento ni tuning.
- **`Bolivia IoU`:** desempeño sobre 15 chips de un país/evento completo jamás visto durante el entrenamiento (mide generalización geográfica real).
- **`Test F1`, `Precision`, `Recall`:** equilibrio entre falsas alarmas (`Precision`) e inundación omitida (`Recall`).
- **`Threshold` y `Size (MB)`:** umbral de probabilidad calibrado en validación y peso real del artefacto en disco.
- Al pie se muestra la referencia del **baseline oficial Otsu** (`Test IoU = 0.5458`, `Bolivia IoU = 0.6972`).

---

## 6. Pestaña "🕒 Historial (SQLite)"

Cada análisis ejecutado se registra automáticamente en la base de datos local `data/flood.db` (tabla `analyses`), guardando fecha de ejecución, evento, fechas satelitales, cobertura de nubes, modelo utilizado, hectáreas inundadas, agua permanente y `live_iou` (si aplica).

> *Nota en despliegue Hugging Face Space:* el disco del Space gratuito es efímero, por lo que el historial y los archivos generados se reinician si el contenedor se suspende o reinicia.

---

## 7. Inicio rápido en 3 pasos (para un usuario nuevo)

1. En **1. Fuente de Datos**, deja **Offline (Sen1Floods11)** → Split `bolivia` → Chip `Bolivia_103757`.
2. En **2. Modelo**, elige **`deeplabv3plus_A`** (solo SAR + Terreno) o **`rf_B`** (SAR + S2 + Terreno) y pulsa **🚀 Ejecutar Detección**.
3. En el mapa, abre el menú de capas (arriba a la derecha), activa **Mapa de Errores** y **Sentinel-2 (RGB Natural)** para inspeccionar aciertos y errores, revisa las **Estadísticas Zonales** y prueba descargar el **GeoTIFF** o **GeoPackage**.
