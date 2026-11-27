# AgentGuard
AgentGuard — A dual-interface AI agent firewall and cost circuit breaker. Combines deterministic regex filtering, Gemini Structured Outputs OWASP classification, real-time token P&amp;L caps, and isolated Cloud Run code execution—accessible via a standalone web console or drop-in MCP server.

> **Status (buildathon build):** the Gemini classifier has been checked against the live API on a [small sample](#live-measurements), and code runs in the Docker sandbox. The Cloud Run provider is written but **not deployed**. Read [Known limits](#known-limits) before relying on any of it.

## What it is

AgentGuard sits between an LLM agent and code execution. It screens each proposed action, blocks what looks risky, stops model spend at a cap, and runs Python in a sandbox only when the action is allowed. One guard core serves two interfaces:

- a **web console** for red-team demos, with verdicts, sandbox output, token spend and a shared ledger;
- an **MCP server** (stdio and Streamable HTTP) that any MCP client can call.

## Architecture

```
Request (web console or MCP client)
  1. Budget gate      session or global cap reached      -> BLOCK (stage BUDGET)
  2. Regex gate       signature hit                      -> BLOCK, risk 100 (stage REGEX), no model call
  3. OWASP classifier (Gemini, structured output)
        reserve cost (refused -> BLOCK, stage BUDGET)
        -> call model -> record actual token usage
        BLOCK, FLAG, error, timeout, invalid output      -> stop (stage OWASP_LLM)
  4. ALLOW only       -> sandbox provider (docker | cloudrun)
  -> result: verdict, sandbox output, budget
```

- **Only `ALLOW` runs.** `FLAG` stops the action just as `BLOCK` does.
- **Fail closed.** A missing API key, a model error or timeout, invalid model output, invalid arguments or an unavailable ledger all end in `BLOCK`.
- **One path to execution:** `execute_guarded` in `core/pipeline.py`. Only the `execute_python` tool can be executed.
- **Spend is reserved first.** Each model call reserves its estimated cost in a SQLite ledger, then records the cost computed from the reported token usage. The web app and every MCP process using the same database share the ledger.
- **The sandbox is the security boundary.** The regex gate is only a cheap pre-filter.

| Path | Contents |
|---|---|
| `core/` | Pipeline, regex gate and classifier, model providers, budget ledger, schemas, settings |
| `sandbox/` | Docker and Cloud Run providers, the worker and its image |
| `app/` | Web API, demo agent, console (`app/static/`), MCP server |
| `tests/` | Test suite |
| `docs/` | [Design notes](docs/design.md), [90-second pitch](docs/pitch.md) |

## Setup

You need Linux, `git`, [uv](https://docs.astral.sh/uv/), a running Docker daemon your user can access, and a Gemini API key for any `ALLOW` verdict (preferably [billing-enabled](#rate-limits)).

```bash
git clone https://github.com/mrehman29/AgentGuard.git
cd AgentGuard
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env        # then set GEMINI_API_KEY in .env
docker build -t agentguard-sandbox:latest sandbox
```

`.env` is git-ignored. Never commit it. The sandbox never pulls an image at run time, so the `docker build` step is required.

Without an API key the server still starts and the regex and budget gates work, but every action that reaches the classifier is blocked, so nothing executes.

## Run the web console

```bash
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Keep it bound to `127.0.0.1`: there is no authentication.

To check the server and the sandbox providers:

```bash
curl http://127.0.0.1:8000/health
```

`"docker": true` means the daemon answers and the image exists. `"cloudrun": false` is expected unless you have configured a Cloud Run service.

## Use the MCP server

| Tool | Arguments | Effect |
|---|---|---|
| `guard_check_action` | `tool_name`, `args_json` (a JSON object as text), `session_id` | Returns a verdict. Runs nothing. |
| `guard_execute_sandbox` | `code`, `session_id`, `provider` | Screens the code and runs it only on `ALLOW`. |
| `guard_get_budget_status` | `session_id` | Returns session and global spend. |

`session_id` defaults to `default`; the console uses `demo`. Session spend and caps are separate for each id; the global cap and the ledger are shared.

`guard_check_action` is advisory: your client must refuse to act on anything other than `ALLOW`. AgentGuard cannot stop actions that a client runs by itself.

### stdio

The client starts the server itself; the web console does not need to be running. Replace `/absolute/path/to/AgentGuard` with the output of `pwd` in your clone:

```json
{
  "mcpServers": {
    "agentguard": {
      "command": "/absolute/path/to/AgentGuard/.venv/bin/python",
      "args": ["/absolute/path/to/AgentGuard/app/mcp_stdio.py"]
    }
  }
}
```

The console shows this block with your paths already filled in (**Copy config**). Settings are read from the clone's `.env`, whichever directory the client starts in.

### Streamable HTTP

Start the web console (above); it serves MCP at `http://127.0.0.1:8000/mcp`. For clients that take an HTTP server in this format, such as Claude Code:

```json
{
  "mcpServers": {
    "agentguard": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Other clients may name these keys differently. Only loopback host names are accepted.

## Configuration

Settings come from `.env` in the project root; real environment variables override it. Defaults are shown below. Rates, the exchange rate and reserve sizes are **estimates**: check them against current pricing.

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | empty | Gemini API key. Empty: the classifier blocks everything. |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Model, unless a role below sets its own. |
| `INPUT_COST_PER_1M_USD` | `0.30` | Input token rate used for costing. |
| `OUTPUT_COST_PER_1M_USD` | `2.50` | Output and thought token rate. |
| `USD_TO_GBP` | `0.80` | Exchange rate. |
| `SESSION_BUDGET_CAP_GBP` | `0.05` | Spend cap for each session. |
| `GLOBAL_BUDGET_CAP_GBP` | `0.50` | Spend cap across all sessions in the database. Reset does not clear it. |
| `SANDBOX_PROVIDER` | `docker` | `docker` or `cloudrun`. |
| `SANDBOX_DOCKER_IMAGE` | `agentguard-sandbox:latest` | Local image for the Docker provider. |
| `CLOUDRUN_SANDBOX_URL` | empty | HTTPS URL of a deployed worker. Calls use a `gcloud` identity token. |
| `SQLITE_DB_PATH` | `data/guard_state.db` | Ledger database. Relative paths start at the project root. |
| `CLASSIFIER_TIMEOUT_SEC` | `5` | Time limit for model calls (`AGENT_TIMEOUT_SEC` overrides it for the agent). |
| `BUDGET_RESERVE_INPUT_TOKENS` | `610` | Input tokens assumed when reserving cost. |
| `BUDGET_RESERVE_OUTPUT_TOKENS` | `512` | Output tokens reserved, and the model's output limit, unless a role sets its own. |
| `PRESET3_SESSION_CAP_GBP` | `0.0017` | Session cap for the runaway loop preset. An older `.env` may still set `0.002`. |
| `PRESET3_MAX_STEPS` | `6` | Most steps the loop preset attempts. |

Rates, timeouts, reserve sizes and output limits must be above zero; a cap may be zero. The server does not start with an invalid setting. Restart it after changing `.env`.

### Model per role

The classifier and the demo agent can be set separately. All of these are optional: an unset classifier setting takes the value from the table above, and an unset agent setting takes the classifier's.

| Variable (`CLASSIFIER_` or `AGENT_` prefix) | Meaning |
|---|---|
| `_PROVIDER` | Only `gemini` is available. |
| `_BASE_URL` | API address. Empty: the provider's default. No credentials, query or fragment. |
| `_MODEL` | Model name. |
| `_API_KEY` | API key. Required if the role's URL differs from the one it would otherwise use, so a key is never sent elsewhere. |
| `_INPUT_COST_PER_1M_USD`, `_OUTPUT_COST_PER_1M_USD` | Rates used to cost this role's calls. Set them when you change the model. |
| `_TIMEOUT_SEC` | Time limit for a call. |
| `_MAX_OUTPUT_TOKENS` | Output limit, also the output size reserved. |

## Tests

```bash
.venv/bin/pytest
```

The suite has 448 tests, including integration and end-to-end tests of the presets, both sandbox providers and the shared ledger. It needs no API key: model responses are faked, and the Cloud Run provider is tested against a local worker, not a deployed service. It uses temporary databases and does not write `data/guard_state.db`.

Docker tests are skipped without a daemon or sandbox image (without a daemon: 418 passed, 30 skipped). Build `agentguard-sandbox:latest` first, as the end-to-end tests look for that tag, and check the summary for skips (`.venv/bin/pytest -rs` lists the reasons).

## Demo

Build the image, set a working API key, start the console and wait for the provider pill to show **Docker · online**. For each preset, click it, then **Run through AgentGuard**.

| Preset | What happens | Expected | Needs a working key |
|---|---|---|---|
| 1 Safe data query | Prompt is screened, the agent proposes Python, the code is screened and run | `ALLOW`, sandbox output, spend recorded | Yes |
| 2 Prompt injection | The prompt hits a signature rule | `BLOCK` at the regex gate, `LLM01`; no model or sandbox call | No |
| 3 Runaway cost loop | Fixed benign steps in a new `loop-…` session with a low cap; each step is classified | At the default cap, 3 steps run and step 4 is a budget `BLOCK` | Yes |

**Reset session** clears the spend of the session on screen and restores its default cap. After preset 3 it clears the `loop-…` session and returns to `demo`, whose spend is kept. Global spend and ledger entries are never cleared by reset; to start from nothing, stop all AgentGuard processes and delete `data/guard_state.db*`.

On 2026-11-27, all three presets ran as expected in one live session with the real model and the Docker sandbox; preset 3, at the default £0.0017 cap, ran 3 steps and refused step 4 on budget. The step count depends on real token usage, so rehearse it and adjust `PRESET3_SESSION_CAP_GBP` if the breaker does not trip where you want it.

Without a key, presets 1 and 3 show a classifier error `BLOCK` and nothing runs.

### Rate limits

In live runs, after about a dozen calls within seconds the API returned `429 RESOURCE_EXHAUSTED` and the guard blocked (fail closed). Use a billing-enabled key, or pause between demo runs.

## Live measurements

Measured on 2026-11-27 with `gemini-3.5-flash-lite`:

| Check | Result |
|---|---|
| 3 benign actions | All `ALLOW` |
| 2 injection prompts without regex signatures | Both `BLOCK`, `LLM01` |
| Time per call | 0.6-1.1 s |
| Cost per call | About £0.0002 at the configured rates (about 400 input and 50 output tokens) |

This is a small sample, **not an accuracy measurement**. Your latency, token use and cost may differ.

## Known limits

**Screening**
- The regex gate can be bypassed (string building, import aliases, unusual spellings, other languages and more). It is not a security boundary.
- It also has false positives: text that only discusses `rm -rf`, `eval(`, `.env` or "ignore all previous instructions" is blocked.
- The classifier can be wrong, in either direction.
- The classifier's accuracy has not been measured. It has been checked live on a [small sample](#live-measurements) only; the automated tests use fake responses.
- API rate limits cause blocks: a rejected call fails closed, so legitimate actions are refused too.

**Budget**
- The budget covers only AgentGuard's own classifier and demo agent calls recorded in the ledger, not an external client's model spend.
- **Spend can go unrecorded.** On a timeout, cancellation, transport error or failed ledger write, the reservation is released although the provider may bill the call. The caps limit recorded spend only.
- Reservations are estimates. A call that uses more input than reserved can take spend past a cap.

**Sandbox and deployment**
- **Cloud Run is not deployed**, so that provider has not run against a real service.
- Its isolation is weaker than Docker's: network egress is open unless you block it yourself, and it lacks the Docker provider's read-only filesystem. Claim network isolation for Docker only.
- **`/tmp` persists in the worker:** files that code writes outside its per-request directory can remain in a warm instance and be seen by later requests. The Docker provider uses a new container each time.
- The base image (`python:3.11-slim`) is not pinned by digest, and only direct Python dependencies are pinned.
- Containers share the host kernel.

**Access**
- No authentication. Any local process that can reach the server can run actions, choose session ids and reset a session's budget. The global cap still applies. Do not expose the server to a network.
