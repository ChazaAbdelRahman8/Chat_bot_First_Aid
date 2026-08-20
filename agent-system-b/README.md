# Agent System B: Synthetic Appointment Agent

Agent System B is an independent Google ADK service for synthetic mental-health
appointment search and booking. Agent System A is the primary user-facing
system and calls this service only over HTTP.

## Responsibilities

- understand language, budget, specialty, consultation mode, and availability;
- search and rank a synthetic psychologist catalog;
- create a synthetic appointment after explicit user selection;
- persist its catalog, request context, and appointments in MongoDB;
- send an authenticated reminder webhook to Agent System A five minutes before
  the appointment; and
- provide both JSON and SSE chat endpoints.

This is a demonstration workflow. It does not contact real providers, create
real clinical appointments, or provide mental-health diagnosis.

## HTTP API

- `GET /health`
- `POST /v1/chat`
- `POST /v1/chat/stream`
- `GET /v1/psychologists`
- `GET /v1/appointments/{appointment_id}`

Example:

```powershell
$body = @{
    conversation_id = "demo-conversation"
    query = "Find an Arabic online psychologist under $50 for anxiety"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
    -Uri http://127.0.0.1:8001/v1/chat `
    -ContentType "application/json" `
    -Body $body
```

Run the whole project from the repository root with:

```powershell
docker compose up -d --build
```

Set `GROQ_API_KEY` and a long random `INTERNAL_SERVICE_TOKEN` in the local
`.env`. The same internal token is injected into both agent containers and is
never committed.
