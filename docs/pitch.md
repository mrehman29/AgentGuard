# AgentGuard: 90-second pitch

| Time | Say | Show |
|---|---|---|
| 0:00-0:15 | "Agents are easy to prototype and risky in production. Injections leak data, unsandboxed code harms hosts, and runaway loops drain budgets. AgentGuard is a firewall and cost circuit breaker, with a web console and an MCP server." | Console, idle |
| 0:15-0:35 | "Every action passes a budget gate, a regex pre-filter, then an OWASP classifier using Gemini structured output. Only an ALLOW reaches the sandbox. The sandbox is swappable: Docker today, with a Cloud Run provider written but not yet deployed. Any error means block." | Gate chips |
| 0:35-0:50 | "A safe query: it is screened, it runs in an isolated container, and its token cost is metered." | Preset 1, output and token P&L |
| 0:50-1:00 | "A prompt injection: stopped as LLM01 before any model or sandbox call." | Preset 2, verdict card |
| 1:00-1:15 | "A runaway loop: each call must reserve its cost first. When the next one would pass the cap, the breaker opens and that call is never made." | Preset 3, budget meter and ledger |
| 1:15-1:30 | "Add it to any MCP client with a short config block: the same policy, one shared ledger and global cap." | MCP config card, **Copy config** |

Reset the session between rehearsal and pitch.

## Before presenting

- **The live classifier is not yet verified.** Presets 1 and 3 need a working Gemini key; run all three presets end to end first. If preset 1 or 3 does not behave as expected, do not describe it as working: show preset 2, which needs no model, and say what was not verified, or show a clearly labelled recording of a successful run if one exists.
- Quote only measured cost and latency. None have been measured yet, so quote none. Figures on screen during a live run may be described as that run's recorded values, computed from configured rates.
- The Docker sandbox runs locally, but nothing executes without the classifier, which needs internet access.
- If asked: the caps limit recorded spend only, the regex gate can be bypassed, and the sandbox is the security boundary. See [Known limits](../README.md#known-limits).
