# mcp-server

A standalone [MCP](https://modelcontextprotocol.io) server exposing structured,
deterministic first-aid lookup tools. It runs in its own Docker container and is
called by Agent System A over the network (Streamable HTTP transport) — it is
never imported as a Python module inside another service.

This complements Agent System A's RAG specialist rather than duplicating it: the
RAG specialist answers open-ended questions by retrieving and citing passages
from the manual corpus, while these tools return fixed, structured,
deterministic output (an ordered protocol checklist, or a rule-based severity
calculation) from a small curated knowledge base.

## Tools

- **`get_first_aid_protocol(condition: str)`** — looks up the ordered first-aid
  steps and safety warnings for a named condition (choking, severe bleeding,
  burns, CPR, fracture, seizure, shock, unconscious/recovery position), with a
  citation into the underlying manual corpus (`data/source_registry.json`
  `doc_id`s). Unknown conditions return the list of available condition names
  instead of raising an error.
- **`assess_burn_severity(percent_body_surface_area: float, degree: str)`** — a
  deterministic triage calculator that classifies burn severity (minor /
  moderate / critical) and a recommended action from the affected body-surface
  percentage and burn depth.

## Data

`data/first_aid_protocols.json` is a small, hand-curated knowledge base (not
extracted from the PDFs at runtime). Each entry cites the manual it is
consistent with, but the wording itself is written for this tool, not copied
from a source document.

## Running standalone

```bash
pip install -r requirements.txt
python src/server.py
```

Serves the MCP Streamable HTTP endpoint at `http://0.0.0.0:8002/mcp`.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest tests/
```
