"""Optional plugins and curated open-source GIS package manager (local mode)."""

from plugins.manager import (
    PLUGIN_CATALOG,
    PluginSpec,
    install_whitelisted_plugin,
    is_plugin_installed,
    list_local_plugin_scripts,
    list_plugins,
)

__all__ = [
    "PLUGIN_CATALOG",
    "PluginSpec",
    "install_whitelisted_plugin",
    "is_plugin_installed",
    "list_local_plugin_scripts",
    "list_plugins",
]
