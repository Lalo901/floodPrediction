"""Lightweight geocoding helper (direct lat/lon parsing + OpenStreetMap Nominatim).

Zero Streamlit dependencies, hardcoded HTTPS endpoint (no SSRF), strict timeouts.
"""

from __future__ import annotations

import re
from typing import Any

import requests

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "FloodMapperMVP/1.0 (https://github.com/Lalo901/floodPrediction)"
_COORD_RE = re.compile(
    r"^\s*([+-]?\d{1,2}(?:\.\d+)?)\s*[,;\s]\s*([+-]?\d{1,3}(?:\.\d+)?)\s*$"
)


def parse_direct_coordinates(query: str) -> tuple[float, float] | None:
    """Parse direct `lat, lon` strings (e.g., `-14.8, -64.9`) without network calls."""
    m = _COORD_RE.match(query.strip())
    if not m:
        return None
    lat = float(m.group(1))
    lon = float(m.group(2))
    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
        return lat, lon
    return None


def _estimate_zoom(place_type: str, addresstype: str) -> int:
    kind = (addresstype or place_type or "").lower()
    if kind in ("country", "continent"):
        return 5
    if kind in ("state", "region", "province"):
        return 7
    if kind in ("county", "municipality", "district"):
        return 10
    if kind in ("city", "town", "village", "suburb", "hamlet"):
        return 12
    return 11


def geocode_place(
    query: str,
    *,
    limit: int = 5,
    timeout_s: float = 6.0,
) -> list[dict[str, Any]]:
    """Search a country, province, city, or direct coordinates (`lat, lon`).

    Returns a list of normalized dicts:
      `{"display_name", "lat", "lon", "bounds", "zoom", "place_type"}`
    where `bounds` is `[[south, west], [north, east]]` for Folium `fit_bounds`, or `None`.
    """
    clean = query.strip()[:120]
    if not clean:
        return []

    coords = parse_direct_coordinates(clean)
    if coords is not None:
        lat, lon = coords
        return [
            {
                "display_name": f"Coordenadas ({lat:.4f}, {lon:.4f})",
                "lat": lat,
                "lon": lon,
                "bounds": None,
                "zoom": 12,
                "place_type": "coordinates",
            }
        ]

    try:
        resp = requests.get(
            NOMINATIM_SEARCH_URL,
            params={
                "q": clean,
                "format": "jsonv2",
                "limit": max(1, min(limit, 8)),
                "addressdetails": 0,
            },
            headers={"User-Agent": USER_AGENT, "Accept-Language": "es,en"},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        raw_items = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"No se pudo consultar el buscador de ubicaciones (Nominatim): {exc}"
        ) from None

    results: list[dict[str, Any]] = []
    for item in raw_items:
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
        except (KeyError, TypeError, ValueError):
            continue

        bbox = item.get("boundingbox")
        bounds: list[list[float]] | None = None
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                south, north, west, east = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
                bounds = [[south, west], [north, east]]
            except (TypeError, ValueError):
                bounds = None

        place_type = str(item.get("type", ""))
        addresstype = str(item.get("addresstype", ""))
        results.append(
            {
                "display_name": str(item.get("display_name", clean)),
                "lat": lat,
                "lon": lon,
                "bounds": bounds,
                "zoom": _estimate_zoom(place_type, addresstype),
                "place_type": addresstype or place_type,
            }
        )
    return results
