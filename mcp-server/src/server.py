"""MCP server exposing structured first-aid lookup and triage tools.

Runs as its own service (own container, own port) and is called by Agent
System A over the network using the Streamable HTTP MCP transport — never
imported as a Python module inside Agent System A.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from protocols import assess_burn_severity as _assess_burn_severity
from protocols import get_first_aid_protocol as _get_first_aid_protocol

mcp = FastMCP(
    "first-aid-tools",
    host="0.0.0.0",  # noqa: S104 - must be reachable from other containers
    port=8002,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "127.0.0.1:*",
            "localhost:*",
            "[::1]:*",
            "mcp-server:*",
            "first_aid_mcp_server:*",
        ],
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
        ],
    ),
)


@mcp.tool()
def get_first_aid_protocol(condition: str) -> dict:
    """Look up the ordered first-aid steps for a named condition or injury.

    Args:
        condition: The condition to look up, e.g. "choking", "severe bleeding",
            "burns", "cpr", "fracture", "seizure", "shock", or "unconscious".
            Matching is case-insensitive and tolerates partial phrases.

    Returns:
        A dict with ``found`` (bool). When found, also includes
        ``condition_id``, ``title``, an ordered list of ``steps``, a list of
        safety ``warnings``, and a ``source`` citation (doc_id and reference)
        into the underlying first-aid manual corpus. When not found, includes
        ``available_conditions`` so the caller can retry with a valid name.
    """
    return _get_first_aid_protocol(condition)


@mcp.tool()
def assess_burn_severity(percent_body_surface_area: float, degree: str) -> dict:
    """Classify burn severity and recommend an action from BSA percentage and depth.

    Args:
        percent_body_surface_area: Estimated percentage (0-100) of total body
            surface area affected by the burn, e.g. from the rule of nines.
        degree: Burn depth classification — one of "first", "second", or
            "third".

    Returns:
        A dict with ``valid`` (bool). When valid, also includes ``severity``
        ("minor", "moderate", or "critical") and a ``recommended_action``
        string. This is a deterministic triage calculation, not a medical
        diagnosis, and does not replace emergency medical assessment.
    """
    return _assess_burn_severity(percent_body_surface_area, degree)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
