"""Curated, security-audited plugin manager for Flood Mapper MVP.

Allows users to:
  1. Use built-in GIS connectors (QGIS QML, MapLibre JSON, UTM Safety Buffer, Executive Report).
  2. Discover and install curated, whitelisted open-source geospatial packages (WhiteboxTools,
     OSMnx, Rasterstats, Contextily, Leafmap) strictly in `APP_MODE=local`.

Security controls:
  - Strict whitelist: only hardcoded `pip_package` entries in `PLUGIN_CATALOG` can be installed.
  - Never executes arbitrary URLs, shell commands (`shell=False`), or unverified scripts.
  - Completely blocks installation when `APP_MODE != "local"` (e.g., Hugging Face Space).
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PLUGINS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PluginSpec:
    """Descriptor for a built-in or whitelisted open-source plugin."""

    id: str
    name: str
    category: str  # "Exportación" | "Cartografía" | "Vector" | "Hidrología" | "Análisis" | "Imágenes"
    source_type: str  # "builtin" | "opensource"
    license_name: str
    summary: str
    value_for_flood_image: str
    homepage_url: str
    import_name: str = ""
    pip_package: str = ""


PLUGIN_CATALOG: tuple[PluginSpec, ...] = (
    # --- Built-in connectors (always installed) ---
    PluginSpec(
        id="qgis_connector",
        name="Conector QGIS (.qml + .gpkg)",
        category="Exportación",
        source_type="builtin",
        license_name="MIT",
        summary="Exporta simbología oficial .qml compatible con QGIS 3.x junto al GeoPackage de inundación.",
        value_for_flood_image="Abre la máscara procesada en QGIS con la paleta Copernicus EMS ya estilizada.",
        homepage_url="https://qgis.org",
    ),
    PluginSpec(
        id="maplibre_connector",
        name="Conector MapLibre / GeoLibre (JSON)",
        category="Cartografía",
        source_type="builtin",
        license_name="BSD-3",
        summary="Genera el bloque JSON de source + layers (fill y outline) listo para MapLibre GL JS.",
        value_for_flood_image="Publica los polígonos inundados directamente en visores web GeoLibre/MapLibre.",
        homepage_url="https://maplibre.org",
    ),
    PluginSpec(
        id="safety_buffer",
        name="Buffer de Seguridad Operativo (UTM)",
        category="Vector",
        source_type="builtin",
        license_name="MIT",
        summary="Calcula perímetros preventivos (+50 m, +100 m o +250 m) en proyección métrica UTM.",
        value_for_flood_image="Delimita franjas de evacuación alrededor de la mancha de agua detectada.",
        homepage_url="https://shapely.readthedocs.io",
    ),
    PluginSpec(
        id="executive_report",
        name="Reporte Ejecutivo de Emergencia (.md)",
        category="Análisis",
        source_type="builtin",
        license_name="MIT",
        summary="Compila métricas, confiabilidad, top 5 focos críticos y estadísticas zonales en Markdown.",
        value_for_flood_image="Entrega un informe técnico listo para compartir con equipos de Defensa Civil.",
        homepage_url="https://github.com/Lalo901/floodPrediction",
    ),
    # --- Curated Open-Source Geospatial Ecosystem (whitelisted, local install only) ---
    PluginSpec(
        id="whitebox",
        name="WhiteboxTools (Hidrología DEM)",
        category="Hidrología",
        source_type="opensource",
        license_name="MIT",
        summary="Motor open-source en Rust con >500 herramientas de hidrología y geomorfología.",
        value_for_flood_image="Permite rellenar depresiones DEM, trazar dirección de flujo y cuencas sobre tu escena.",
        homepage_url="https://www.whiteboxgeo.com",
        import_name="whitebox",
        pip_package="whitebox",
    ),
    PluginSpec(
        id="osmnx",
        name="OSMnx (Infraestructura OpenStreetMap)",
        category="Vector",
        source_type="opensource",
        license_name="MIT",
        summary="Descarga redes viales, puentes y edificios desde OpenStreetMap en GeoDataFrames.",
        value_for_flood_image="Cruza tu polígono inundado con calles, hospitales y puentes potencialmente afectados.",
        homepage_url="https://osmnx.readthedocs.io",
        import_name="osmnx",
        pip_package="osmnx",
    ),
    PluginSpec(
        id="rasterstats",
        name="Python Rasterstats (Estadísticas Zonales)",
        category="Análisis",
        source_type="opensource",
        license_name="BSD-3",
        summary="Cálculo rápido de estadísticas zonales sobre rasters GeoTIFF por polígono.",
        value_for_flood_image="Resume probabilidad media, HAND y pendiente por cada foco crítico individual.",
        homepage_url="https://pythonhosted.org/rasterstats/",
        import_name="rasterstats",
        pip_package="rasterstats",
    ),
    PluginSpec(
        id="contextily",
        name="Contextily (Mapas Base Cartográficos)",
        category="Cartografía",
        source_type="opensource",
        license_name="BSD-3",
        summary="Descarga teselas web (OSM, Stamen, CartoDB) para componer láminas estáticas.",
        value_for_flood_image="Genera mapas impresos/PDF de alta resolución con la máscara de agua superpuesta.",
        homepage_url="https://contextily.readthedocs.io",
        import_name="contextily",
        pip_package="contextily",
    ),
    PluginSpec(
        id="leafmap",
        name="Leafmap (Visor y COG Open-Source)",
        category="Imágenes",
        source_type="opensource",
        license_name="MIT",
        summary="Ecosistema open-source para mapas interactivos, Cloud Optimized GeoTIFF y mosaicos.",
        value_for_flood_image="Amplía el manejo de rasters COG y exportación cartográfica avanzada en Python.",
        homepage_url="https://leafmap.org",
        import_name="leafmap",
        pip_package="leafmap",
    ),
)

_CATALOG_BY_ID: dict[str, PluginSpec] = {p.id: p for p in PLUGIN_CATALOG}


def is_plugin_installed(spec: PluginSpec) -> bool:
    """Return True if a built-in plugin is active or an open-source package is importable."""
    if spec.source_type == "builtin":
        return True
    if not spec.import_name:
        return False
    return importlib.util.find_spec(spec.import_name) is not None


def list_plugins(
    *,
    category: str | None = None,
    query: str = "",
    only_installed: bool | None = None,
) -> list[dict[str, object]]:
    """List catalog plugins with live installation status, filtered by category/search."""
    q = query.strip().lower()
    results: list[dict[str, object]] = []
    for spec in PLUGIN_CATALOG:
        installed = is_plugin_installed(spec)
        if only_installed is True and not installed:
            continue
        if only_installed is False and installed:
            continue
        if category and category != "Todas" and spec.category != category:
            continue
        if q:
            haystack = f"{spec.name} {spec.category} {spec.summary} {spec.value_for_flood_image}".lower()
            if q not in haystack:
                continue
        results.append({"spec": spec, "installed": installed})
    return results


def list_local_plugin_scripts() -> list[str]:
    """List optional user-created Python scripts placed in `plugins/` (informational only)."""
    if not PLUGINS_DIR.exists():
        return []
    return sorted(
        p.name
        for p in PLUGINS_DIR.glob("*.py")
        if p.name not in ("__init__.py", "manager.py")
    )


def install_whitelisted_plugin(
    plugin_id: str,
    *,
    app_mode: str = "local",
    timeout_s: int = 180,
) -> tuple[bool, str]:
    """Install a strictly whitelisted open-source package into the active Python env (local only)."""
    if app_mode != "local":
        raise PermissionError(
            f"La instalación de complementos está bloqueada en APP_MODE={app_mode} por seguridad."
        )

    spec = _CATALOG_BY_ID.get(plugin_id)
    if spec is None:
        raise ValueError(f"Complemento no reconocido en la lista blanca: `{plugin_id}`.")
    if spec.source_type != "opensource" or not spec.pip_package:
        return True, f"`{spec.name}` ya viene integrado en la aplicación."
    if is_plugin_installed(spec):
        return True, f"`{spec.name}` ya está instalado en el entorno activo."

    uv_bin = shutil.which("uv")
    if uv_bin:
        cmd = [uv_bin, "pip", "install", "--python", sys.executable, spec.pip_package]
    else:
        cmd = [sys.executable, "-m", "pip", "install", spec.pip_package]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Tiempo de espera agotado instalando `{spec.pip_package}`."
    except OSError as exc:
        return False, f"No se pudo invocar el instalador: {exc}"

    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout or "").strip()[-400:]
        return False, f"Error instalando `{spec.pip_package}`: {err_tail}"

    return True, f"`{spec.name}` instalado correctamente en el entorno actual."
