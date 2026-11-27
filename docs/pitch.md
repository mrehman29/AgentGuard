# AgentGuard: 90-second pitch

| Time | Say | Show |
|---|---|---|
| 0:00-0:15 | "Agents are easy to prototype and risky in production. Injections leak data, unsandboxed code harms hosts, and runaway loops drain budgets. AgentGuard is a firewall and cost circuit breaker, with a web console and an MCP server." | Console, idle |
| 0:15-0:35 | "Every action passes a budget gate, a regex pre-filter, then an OWASP classifier using Gemini structured output. Only an ALLOW reaches the sandbox. The sandbox is swappable: Docker today, with a Cloud Run provider written but not yet deployed. Any error means block." | Gate chips |
| 0:35-0:50 | "A safe query: it is screened, it runs in an isolated container, and its token cost is metered. In our test runs a classifier call took 0.6 to 1.1 seconds and cost about two hundredths of a penny." | Preset 1, output and token P&L |
| 0:50-1:00 | "A prompt injection: stopped as LLM01 before any model or sandbox call." | Preset 2, verdict card |
| 1:00-1:15 | "A runaway loop: each call must reserve its cost first. When the next one would pass the cap, the breaker opens and that call is never made." | Preset 3, budget meter and ledger |
| 1:15-1:30 | "Add it to any MCP client with a short config block: the same policy, one shared ledger and global cap." | MCP config card, **Copy config** |

Reset the session between rehearsal and pitch.

## Before presenting

- All three presets ran as expected in one live session with the real model and the Docker sandbox on 2026-11-27; preset 3, at its £0.0017 cap, ran 3 steps and refused step 4 on budget. Its step count depends on token usage, so still rehearse, check that `.env` does not still set `0.002`, and describe only what you see.
- **Rate limits:** after about a dozen calls within seconds the API returned `429 RESOURCE_EXHAUSTED` and the guard blocked. Use a billing-enabled key, or pause between demo runs, including between rehearsal and pitch.
- Presets 1 and 3 need a working Gemini key. If one does not behave as expected, say so: show preset 2, which needs no model, or a clearly labelled recording of a successful run if one exists.
- Quote only measured cost and latency: 0.6-1.1 s per call and about £0.0002 per call at the configured rates (about 400 input and 50 output tokens), measured on 2026-11-27 with `gemini-3.5-flash-lite`. The spoken line rounds these. A run of preset 1 makes more than one call.
- Detection: 3 benign actions were allowed and 2 injection prompts without regex signatures were blocked as LLM01. That is a small sample; claim no accuracy figure.
- The Docker sandbox runs locally, but nothing executes without the classifier, which needs internet access.
- If asked: the caps limit recorded spend only, the regex gate can be bypassed, and the sandbox is the security boundary. See [Known limits](../README.md#known-limits).
