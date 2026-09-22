# OpenAI engineering pilot

The first live run uses `gpt-4.1-mini-2025-04-14`, temperature 0, 512 maximum output tokens, strict JSON schema and `store=false`. Standard token prices checked on 2026-09-22: $0.40/M input and $1.60/M output. Source: https://developers.openai.com/api/docs/models/gpt-4.1-mini

Credentials are loaded locally from `.env` or the `OPENAI_API_KEY` environment variable. They are never written into request artifacts. Requests go only to `https://api.openai.com/v1/responses`. No provider switch, retry, external tools, or account billing changes are performed.

Five households are selected by a fixed hash ranking, independently of outcomes. All eligible test snapshots for those households are retained. A structured-history forecast, a persona summary from the same information, and a persona-conditioned forecast require three calls per snapshot. Baselines are evaluated on precisely the same matched rows. This sample is an engineering check, not evidence for a general research conclusion.

## Local cost guard

The pilot cap cannot exceed $0.50. Each call reserves a conservative amount before transmission, using UTF-8 request bytes plus framing allowance as an input token bound, and maximum output tokens. Reservations are written and fsynced before each request; unused amounts are not reclaimed during the run. Unknown/failed calls retain their reservation. API errors stop the run without retry. Existing output directories cannot be reused, preventing accidental repeat charges on a simple rerun.

This is a per-run application guard based on verified pricing, not an OpenAI account-level spending limit. Repeated runs in different directories would create additional costs. Usage-derived costs may differ from the final bill (e.g. cached input discounts); the account balance is not read. A completed run must be inspected before authorizing expansion.

## Commands

```sh
uv run python -m behaviorlens.prepare_llm --run outputs/my_run --output outputs/pilot_plan --households 5
uv run python -m behaviorlens.openai_pilot --plan outputs/pilot_plan --output outputs/pilot --baseline-run outputs/my_run
```

Every prompt, raw response, usage count, latency, source hash, parsed prediction and redacted failure event stays in the local run folder. Future labels are joined only after all predictions complete. Incomplete runs are not silently scored on successful subsets.

The generated persona may omit or distort information; preserving source information is an instruction, not an enforced guarantee. Summary quality needs separate failure analysis before interpreting a difference as a representation effect. No inference about stability follows from a single run at temperature zero.
