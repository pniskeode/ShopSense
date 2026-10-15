"""M6's customer-scoped Model Context Protocol integration."""

from typing import Any

__all__ = ["MockApiGateway", "create_mcp_server"]


def __getattr__(name: str) -> Any:
    """Avoid importing the server before ``python -m shopsense.mcp.server`` runs it."""

    if name in __all__:
        from .server import MockApiGateway, create_mcp_server

        return {"MockApiGateway": MockApiGateway, "create_mcp_server": create_mcp_server}[name]
    raise AttributeError(name)
