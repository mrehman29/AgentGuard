/*
 * AgentGuard console (PLAN.md sections 5 and 9, design handoff).
 * Everything that comes from the model, the sandbox, the ledger or the server
 * is written with textContent. No markup is ever built from strings.
 */
'use strict';

const HOME_SESSION = 'demo';
const PROVIDER_NAMES = { docker: 'Docker', cloudrun: 'Cloud Run' };
const GATES = { budget: 'BUDGET', regex: 'REGEX', owasp: 'OWASP_LLM' };
const CHIP_NAMES = { budget: 'Budget', regex: 'Regex', owasp: 'OWASP', sandbox: 'Sandbox' };

const TEXT = {
  idleIntent: 'No request yet.',
  mcpIntent: 'No web request. MCP calls appear in the ledger.',
  idleReason: 'Every action passes Budget, Regex and OWASP gates before it can execute.',
  noOutput: 'No output yet.',
  waiting: 'Waiting for the verdict…',
  notRun: 'Not run. Only an ALLOW verdict reaches the sandbox.',
  noTool: 'No tool was proposed, so nothing ran.',
  failedReason: 'No verdict. Treat as not executed. Check the server and run again.',
};

const $ = (id) => document.getElementById(id);
const wideLayout = window.matchMedia('(min-width: 1900px) and (min-height: 1060px)');

const state = {
  session: HOME_SESSION,
  config: null,
  provider: null,          // null until the config or the user chooses one
  providerHealth: null,    // null until /health answers
  presetId: null,
  running: false,
  resetting: false,
  hasRun: false,           // a web request is on screen
  breaker: null,           // null | 'session' | 'global'
  budget: null,
  budgetVersion: 0,        // bumps whenever newer budget data is shown
  ledger: [],
  ledgerRequest: 0,
  ledgerShown: 0,
  pendingRow: null,        // console-only row: {session, label} or {session, failed}
  idleIntent: null,        // card 1 text of the current idle state
  shownSandbox: null,      // the sandbox result on card 3
  run: null,
};

/* ---------- Formatting ---------- */

const isNumber = (value) => typeof value === 'number' && Number.isFinite(value);
const money = (value) => (isNumber(value) ? `£${value.toFixed(5)}` : '£—');
const sessionCap = (value) => (isNumber(value) ? `£${value.toFixed(4)}` : '£—');
const count = (value) => (isNumber(value) ? value.toLocaleString('en-GB') : '—');

function globalCap(value) {
  if (!isNumber(value)) return '£—';
  const short = value.toFixed(2);
  return `£${Number(short) === value ? short : value.toFixed(4)}`;
}

const providerName = (id) => PROVIDER_NAMES[id] ?? id;
const selectedProvider = () => state.provider ?? 'docker';

async function getJson(url, options) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  if (!response.ok) throw new Error(`HTTP ${response.status} from ${url}`);
  return response.json();
}

/* ---------- Provider pill ---------- */

function renderProviders() {
  const pill = $('provider-pill');
  const ids = state.config?.providers ?? Object.keys(PROVIDER_NAMES);
  ids.forEach((id, index) => {
    let button = pill.children[index];
    if (!button || button.dataset.provider !== id) {
      button = document.createElement('button');
      button.type = 'button';
      button.className = 'segment';
      button.dataset.provider = id;
      button.append(document.createElement('img'), document.createElement('span'));
      button.addEventListener('click', () => selectProvider(id));
      pill.insertBefore(button, pill.children[index] ?? null);
    }
    const selected = id === selectedProvider();
    let health = null;
    if (selected) {
      health = state.providerHealth === null ? 'checking'
        : state.providerHealth[id] === true ? 'online' : 'offline';
    }
    button.setAttribute('aria-pressed', String(selected));
    if (health) button.dataset.health = health;
    else delete button.dataset.health;

    const [icon, label] = button.children;
    icon.alt = '';
    icon.hidden = health !== 'online' && health !== 'offline';
    if (!icon.hidden) icon.src = `/static/icons/pill-${health}.svg`;
    label.textContent = health ? `${providerName(id)} · ${health}` : providerName(id);
  });
  while (pill.children.length > ids.length) pill.lastElementChild.remove();
}

function selectProvider(id) {
  state.provider = id;
  renderProviders();
  if (!state.hasRun && !state.running) $('c3-provider').textContent = id;
  checkHealth();
}

async function checkHealth() {
  try {
    const data = await getJson('/health');
    state.providerHealth = data.providers ?? {};
  } catch {
    state.providerHealth = {};
  }
  renderProviders();
}

/* ---------- Budget: header meter and card 4 ---------- */

function showBudget(budget) {
  if (!budget || budget.session_id !== state.session) return;
  state.budget = budget;
  state.budgetVersion += 1;
  renderBudget();
}

function renderBudget() {
  const budget = state.budget?.session_id === state.session ? state.budget : null;
  const breaker = state.breaker;

  $('meter').classList.toggle('breaker', breaker !== null);
  $('meter-text').textContent = budget
    ? `${state.session} · ${money(budget.spent_gbp)} / ${sessionCap(budget.cap_gbp)}`
      + (breaker ? ' · BREAKER OPEN' : '')
    : `${state.session} · loading budget…`;

  const bar = $('meter-bar');
  const spent = budget?.spent_gbp ?? 0;
  const percent = budget && budget.cap_gbp > 0 ? (spent / budget.cap_gbp) * 100
    : spent > 0 ? 100 : 0;
  bar.style.width = `${Math.min(percent, 100)}%`;
  bar.style.minWidth = spent > 0 ? '6px' : '0';
  // A global breaker keeps the session bar; only the text turns red.
  bar.classList.toggle('red', breaker === 'session');
  bar.classList.toggle('amber', breaker !== 'session' && percent >= 15);

  const globalText = budget
    ? `Global ${money(budget.global_spent_gbp)} / ${globalCap(budget.global_cap_gbp)}`
    : 'Global —';
  $('meter-global').textContent = globalText;
  $('c4-global').textContent = `${globalText} · all sessions`;

  $('c4-session').textContent = `session ${state.session}`;
  $('c4-spent').textContent = budget ? money(budget.spent_gbp) : '—';
  $('c4-reserved').textContent = budget ? money(budget.reserved_gbp) : '—';
  $('c4-remaining').textContent = budget ? money(budget.remaining_gbp) : '—';
  $('c4-cap').textContent = budget ? sessionCap(budget.cap_gbp) : '—';
  $('c4-prompt').textContent = count(budget?.prompt_tokens);
  $('c4-candidates').textContent = count(budget?.candidates_tokens);
  $('c4-thoughts').textContent = count(budget?.thoughts_tokens);
  $('c4-total').textContent = count(budget?.total_tokens);

  const note = $('c4-note');
  note.classList.toggle('alert-text', breaker !== null);
  $('card-4').classList.toggle('alert', breaker !== null);
  if (breaker) {
    note.textContent = `Circuit breaker open: next call would exceed the ${breaker} cap`;
  } else if (!budget) {
    note.textContent = 'Loading budget…';
  } else {
    note.textContent = budget.is_exceeded ? 'Budget cap reached' : 'Within budget';
  }
}

/* The API does not name the cap that refused a call: it is the tighter one. */
function refusingCap(budget) {
  if (!budget) return 'session';
  const globalLeft = budget.global_cap_gbp - budget.global_spent_gbp;
  return globalLeft < budget.remaining_gbp ? 'global' : 'session';
}

async function pollBudget() {
  const session = state.session;
  const version = state.budgetVersion;
  try {
    const budget = await getJson(`/api/budget/${encodeURIComponent(session)}`);
    // Drop answers that were overtaken by a run event or a session change.
    if (version === state.budgetVersion) showBudget(budget);
  } catch (error) {
    console.error('Budget poll failed:', error);
  }
}

/* ---------- Ledger ---------- */

function ledgerLine(source, parts) {
  const wide = wideLayout.matches;
  const lead = wide || source !== 'web' ? `${source} ` : '';
  return lead + parts.join(wide ? '  ' : ' ');
}

function renderLedger() {
  const lines = state.ledger.map((entry) => ledgerLine(entry.source, [
    entry.session_id, entry.label, entry.outcome, money(entry.cost_gbp),
  ]));
  const pending = state.pendingRow;
  if (pending) {
    lines.push(ledgerLine('web', pending.failed
      ? [pending.session, 'request failed']
      : [pending.session, pending.label, 'checking…']));
  }
  if (lines.length === 0) lines.push('No activity yet.');

  const box = $('ledger-rows');
  const key = lines.join('\n');
  if (box.dataset.key === key) return;   // keep the reader's scroll position
  box.dataset.key = key;
  box.replaceChildren(...lines.map((line) => {
    const row = document.createElement('div');
    row.className = 'ledger-row';
    row.textContent = line;
    return row;
  }));
  box.scrollTop = box.scrollHeight;
}

async function pollLedger() {
  const request = ++state.ledgerRequest;
  try {
    const entries = await getJson('/api/ledger/recent?limit=50');
    if (request < state.ledgerShown) return;
    state.ledgerShown = request;
    state.ledger = entries;
    renderLedger();
    if (!state.hasRun && !state.running && state.idleIntent === TEXT.idleIntent) {
      $('c1-intent').textContent = entries.at(-1)?.source === 'mcp'
        ? TEXT.mcpIntent : TEXT.idleIntent;
    }
  } catch (error) {
    console.error('Ledger poll failed:', error);
  }
}

/* ---------- Card 1: agent intent ---------- */

function setIntent({ tool, intent, label, text }) {
  $('c1-tool').textContent = tool;
  $('c1-intent').textContent = intent;
  $('c1-label').textContent = label;
  $('c1-label').dataset.base = label;
  const code = $('c1-code');
  code.textContent = text;
  code.scrollTop = 0;
  fitOutputs();
}

function showAction(tool, intent, args) {
  const values = args ?? {};
  let label = 'Arguments';
  let text = Object.keys(values).length ? JSON.stringify(values, null, 2) : '–';
  if (tool === 'execute_python' && typeof values.code === 'string') {
    label = 'Code';
    text = values.code;
  } else if (tool === 'agent_prompt' && typeof values.prompt === 'string') {
    label = 'Prompt';
    text = values.prompt;
  }
  setIntent({ tool, intent, label, text });
}

/* ---------- Card 2: verdict and stage chips ---------- */

function setVerdict(kind, word, tag, risk, reason) {
  $('c2-badge').dataset.verdict = kind;
  $('c2-icon').src = `/static/icons/verdict-${kind}.svg`;
  $('c2-word').textContent = word;
  $('c2-tag').textContent = tag;
  $('c2-risk').textContent = risk;
  $('c2-reason').textContent = reason;
  $('card-2').classList.toggle('alert', kind === 'block' || kind === 'failed');
  $('card-2').querySelector('.card-body').scrollTop = 0;
}

function setChip(name, chipState) {
  const chip = $(`chip-${name}`);
  const word = chipState[0].toUpperCase() + chipState.slice(1);
  chip.dataset.state = chipState;
  chip.querySelector('img').src = `/static/icons/chip-${chipState}.svg`;
  chip.title = `${CHIP_NAMES[name]}: ${word}`;
  chip.setAttribute('aria-label', chip.title);
}

function setAllChips(chipState) {
  Object.keys(CHIP_NAMES).forEach((name) => setChip(name, chipState));
}

function showGuard(guard, sandbox, sandboxPending) {
  const decision = guard.decision;
  const latency = `${count(guard.latency_ms)} ms · ${guard.stage}`;
  const risk = guard.owasp_tag === 'N/A'
    ? `Risk not assessed · ${latency}`
    : `Risk ${guard.risk_score} / 100 · ${latency}`;
  const kind = decision === 'ALLOW' ? 'allow' : decision === 'FLAG' ? 'flag' : 'block';
  setVerdict(kind, decision, guard.owasp_tag, risk, guard.reason);

  const passed = guard.gates_passed ?? [];
  for (const [name, gate] of Object.entries(GATES)) {
    let chipState = passed.includes(gate) ? 'passed' : 'skipped';
    if (decision !== 'ALLOW' && guard.stage === gate) {
      chipState = decision === 'FLAG' ? 'flagged' : 'stopped';
    }
    setChip(name, chipState);
  }
  if (decision !== 'ALLOW') setChip('sandbox', 'skipped');
  else if (sandbox) setChip('sandbox', ranCleanly(sandbox) ? 'passed' : 'stopped');
  else setChip('sandbox', sandboxPending ? 'waiting' : 'skipped');
}

/* ---------- Card 3: sandbox execution ---------- */

const ranCleanly = (sandbox) => sandbox.exit_code === 0 && !sandbox.timed_out;

function trimmedOutput(text) {
  return (text ?? '').replace(/\r?\n$/, '');
}

/* Idle, waiting and not-run states: no metrics, one muted line. */
function showNoExecution(note, message) {
  state.shownSandbox = null;
  $('card-3').classList.remove('alert', 'failed', 'split');
  $('c3-provider').textContent = note;
  $('c3-headline').hidden = true;
  $('c3-headline').textContent = '';
  for (const id of ['c3-exit', 'c3-duration', 'c3-timeout', 'c3-truncated']) {
    $(id).textContent = '—';
  }
  $('c3-stdout-box').hidden = false;
  $('c3-stdout-box').classList.add('placeholder');
  $('c3-stdout-label').textContent = 'stdout';
  $('c3-stdout').textContent = message;
  $('c3-stderr-box').hidden = true;
  $('c3-stderr').textContent = '';
  $('c3-lines').hidden = true;
  $('c3-lines').textContent = '';
}

function showSandbox(sandbox) {
  showNoExecution(sandbox.provider, '');
  state.shownSandbox = sandbox;
  const failed = !ranCleanly(sandbox);
  const stdout = trimmedOutput(sandbox.stdout);
  const stderr = trimmedOutput(sandbox.stderr);

  const card = $('card-3');
  card.classList.toggle('alert', failed);
  card.classList.toggle('failed', failed);
  card.classList.toggle('split', Boolean(stdout && stderr));

  if (failed) {
    let headline = `FAILED: exit code ${sandbox.exit_code}`;
    if (sandbox.timed_out) {
      headline = 'TIMED OUT';
    } else if (sandbox.exit_code === 125) {
      headline = 'SANDBOX OFFLINE: NOTHING RAN';
    }
    $('c3-headline').textContent = headline;
    $('c3-headline').hidden = false;
  }

  $('c3-exit').textContent = String(sandbox.exit_code);
  $('c3-duration').textContent = `${count(sandbox.duration_ms)} ms`;
  $('c3-timeout').textContent = sandbox.timed_out ? 'yes' : 'no';
  $('c3-truncated').textContent = sandbox.truncated ? 'YES' : 'no';

  $('c3-stdout-box').classList.remove('placeholder');
  $('c3-stdout-box').hidden = !stdout && Boolean(stderr);
  $('c3-stdout-box').classList.toggle('placeholder', !stdout);
  $('c3-stdout').textContent = stdout || '(no output)';
  $('c3-stderr-box').hidden = !stderr;
  $('c3-stderr').textContent = stderr;
  fitOutputs();
}

/* Scroll hints depend on the rendered height, so they are measured. */
function fitOutputs() {
  const lineCount = (element) => ({
    shown: Math.round(element.clientHeight / 22),
    total: Math.round(element.scrollHeight / 22),
  });
  const overflows = (element) => element.scrollHeight > element.clientHeight + 1;

  const code = $('c1-code');
  const base = $('c1-label').dataset.base ?? 'Arguments';
  $('c1-label').textContent = base;
  if (overflows(code)) {
    const { shown, total } = lineCount(code);
    $('c1-label').textContent = `${base} · showing ${shown} of ${total} lines · scroll`;
  }

  const sandbox = state.shownSandbox;
  if (!sandbox) return;
  const stdout = $('c3-stdout');
  const stderr = $('c3-stderr');
  const both = !$('c3-stdout-box').hidden && !$('c3-stderr-box').hidden;
  const cut = sandbox.truncated ? ' · cut at 16 KB' : '';
  const note = $('c3-lines');
  note.hidden = true;

  let label = `stdout${cut}`;
  if (!$('c3-stdout-box').hidden && overflows(stdout)) {
    if (both) {
      label += ' · scroll';
    } else {
      const { shown, total } = lineCount(stdout);
      note.textContent = `Showing ${shown} of ${total} lines · scroll`;
      note.hidden = false;
    }
  }
  $('c3-stdout-label').textContent = label;

  let errorLabel = $('c3-stdout-box').hidden ? `stderr${cut}` : 'stderr';
  if (!$('c3-stderr-box').hidden) {
    const hidden = overflows(stderr);
    if (hidden) errorLabel += wideLayout.matches ? ' · scroll for more' : ' · scrolled to end';
    $('c3-stderr-label').textContent = errorLabel;
    stderr.scrollTop = stderr.scrollHeight;
  }
}

/* ---------- Whole-card states ---------- */

function clearCards(intent = TEXT.idleIntent, reason = TEXT.idleReason) {
  state.idleIntent = intent;
  $('c1-session').textContent = `session ${state.session}`;
  setIntent({ tool: '—', intent, label: 'Arguments', text: '–' });
  setVerdict('idle', 'NO REQUEST', 'No verdict yet', 'Waiting for a request', reason);
  setAllChips('waiting');
  showNoExecution(selectedProvider(), TEXT.noOutput);
}

function showFailure(detail) {
  state.pendingRow = { session: state.session, failed: true };
  state.breaker = null;
  clearCards('The request did not complete.');
  setVerdict('failed', 'FAILED', 'Request failed', detail, TEXT.failedReason);
  renderBudget();
  renderLedger();
}

/* ---------- Run events ---------- */

const handlers = {
  run_started(event) {
    state.session = event.session_id;
    state.run.loop = event.preset === 'loop';
    state.run.totalSteps = event.total_steps;
    $('c1-session').textContent = `session ${state.session}`;
    renderBudget();
    pollBudget();
  },

  step_started(event) {
    const run = state.run;
    const previous = run.loop && event.index > 1 ? state.shownSandbox : null;
    showAction(event.tool_name, event.intent, event.tool_args);
    setVerdict('checking', 'CHECKING', 'Checking…', 'Budget and regex gates', '');
    setAllChips('waiting');
    if (previous) {
      // A loop keeps the last result on screen, labelled with its step.
      $('c3-provider').textContent = `${previous.provider} · step ${event.index - 1} result`;
    } else {
      const provider = run.provider;
      showNoExecution(`${provider} · waiting`, TEXT.waiting);
    }
    const label = event.index === 0 ? 'screen'
      : `step ${event.index}/${run.loop ? run.totalSteps : 1}`;
    state.pendingRow = { session: state.session, label };
    renderLedger();
  },

  budget_reserved(event) {
    showBudget(event.budget);
    setChip('budget', 'passed');
    setChip('regex', 'passed');
    setChip('owasp', 'running');
    const amount = `${money(event.budget.reserved_gbp)} reserved for`;
    let reason = `${amount} this check.`;
    if (!state.run.loop) {
      reason = event.index === 0
        ? `${amount} the prompt screen.`
        : `Prompt screened and agent turn done. ${reason}`;
    }
    setVerdict('checking', 'CHECKING', 'Classifying…', 'OWASP classifier', reason);
  },

  step_finished(event) {
    const { result } = event;
    const { guard, sandbox } = result;
    // The prompt screen passed: the agent turn and its tool are still to come.
    const pending = event.index === 0 && guard.decision === 'ALLOW'
      && result.tool_name === 'agent_prompt';
    state.breaker = guard.stage === 'BUDGET' ? refusingCap(result.budget) : null;
    state.pendingRow = null;

    showAction(result.tool_name, result.intent, result.tool_args);
    showGuard(guard, sandbox, pending);
    if (sandbox) showSandbox(sandbox);
    else if (pending) showNoExecution(`${state.run.provider} · waiting`, TEXT.waiting);
    else showNoExecution('not run', TEXT.notRun);

    showBudget(result.budget);
    renderBudget();
    renderLedger();
    pollLedger();
  },

  agent_reply(event) {
    setIntent({
      tool: 'no tool proposed',
      intent: 'The agent answered in text.',
      label: 'Agent reply',
      text: event.text,
    });
    setChip('sandbox', 'skipped');
    showNoExecution('not run', TEXT.noTool);
  },

  run_finished(event) {
    state.run.ended = true;
    state.pendingRow = null;
    showBudget(event.budget);
    renderLedger();
  },

  run_error(event) {
    state.run.ended = true;
    showFailure(event.message);
  },
};

function handleLine(line) {
  if (!line.trim() || state.run.ended) return;
  const event = JSON.parse(line);
  handlers[event.type]?.(event);
}

async function readStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { done, value } = await reader.read();
    buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = done ? '' : lines.pop();
    lines.forEach(handleLine);
    if (done) break;
  }
}

/* ---------- Controls ---------- */

function updateControls() {
  const busy = state.running || state.resetting;
  const prompt = $('prompt-input');
  prompt.disabled = busy;
  $('btn-run').disabled = busy || prompt.value.trim() === '';
  $('btn-run').textContent = state.running ? 'Running…' : 'Run through AgentGuard';
  $('btn-reset').disabled = busy;
  for (const button of document.querySelectorAll('.preset')) {
    button.disabled = busy;
    button.setAttribute('aria-pressed', String(button.dataset.preset === state.presetId));
  }
}

const presetById = (id) => state.config?.presets.find((preset) => preset.id === id);

async function startRun() {
  const prompt = $('prompt-input').value.trim();
  if (state.running || state.resetting || !prompt) return;

  // A preset is sent only while the text is still its own prompt.
  const preset = presetById(state.presetId);
  const presetId = preset && preset.prompt.trim() === prompt ? preset.id : null;

  state.running = true;
  state.hasRun = true;
  state.session = HOME_SESSION;   // only run_started moves away from it
  state.breaker = null;
  state.budgetVersion += 1;
  state.pendingRow = null;
  state.run = { loop: false, totalSteps: 1, provider: selectedProvider(), ended: false };
  updateControls();
  clearCards('Sending the request…', '');
  setVerdict('checking', 'CHECKING', 'Checking…', 'Sending the request', '');
  showNoExecution(`${state.run.provider} · waiting`, TEXT.waiting);
  renderBudget();
  renderLedger();
  pollBudget();

  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        preset: presetId,
        session_id: state.session,
        provider: state.provider,
      }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status} from /api/run`);
    await readStream(response);
    if (!state.run.ended) throw new Error('The result stream ended early');
  } catch (error) {
    console.error('Run failed:', error);
    if (!state.run.ended) showFailure(error.message);
  } finally {
    state.running = false;
    updateControls();
    pollBudget();
    pollLedger();
  }
}

async function resetSession() {
  if (state.running || state.resetting) return;
  const session = state.session;
  state.resetting = true;
  updateControls();
  try {
    await getJson(`/api/budget/${encodeURIComponent(session)}/reset`, {
      method: 'POST',
      headers: { 'X-Requested-With': 'AgentGuard' },
    });
    state.session = HOME_SESSION;
    state.breaker = null;
    state.budget = null;
    state.budgetVersion += 1;
    state.presetId = null;
    state.pendingRow = null;
    state.hasRun = false;
    $('prompt-input').value = '';
    if (session === HOME_SESSION) {
      clearCards(`${session} reset.`, 'Session demo was cleared. Global total is kept.');
    } else {
      clearCards(
        `${session} reset. Back to session demo.`,
        'The loop session was cleared. Session demo keeps its spend. Global total is kept.',
      );
    }
    renderBudget();
    renderLedger();
  } catch (error) {
    console.error('Reset failed:', error);
    clearCards('The reset did not complete.');
    setVerdict('failed', 'FAILED', 'Reset failed', error.message,
      'The session was not reset. Check the server and try again.');
    state.hasRun = true;
  } finally {
    state.resetting = false;
    updateControls();
    pollBudget();
    pollLedger();
  }
}

/* ---------- Configuration: presets and MCP ---------- */

function renderPresets() {
  $('presets').replaceChildren(...state.config.presets.map((preset, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'preset';
    button.dataset.preset = preset.id;

    const title = document.createElement('span');
    title.className = 'preset-title';
    title.textContent = `${index + 1}  ${preset.title}`;

    const description = document.createElement('span');
    description.className = 'preset-desc';
    let text = preset.description;
    if (preset.session_cap_gbp !== null && preset.session_cap_gbp !== undefined) {
      const cap = sessionCap(preset.session_cap_gbp);
      if (!text.includes(cap)) text = `Cap ${cap}. ${text}`;
    }
    description.textContent = text;

    button.append(title, description);
    button.addEventListener('click', () => {
      if (state.running || state.resetting) return;
      state.presetId = preset.id;
      $('prompt-input').value = preset.prompt;
      updateControls();
      startRun();
    });
    return button;
  }));
}

/* The script path works from any directory; module form needs the cwd. */
function mcpConfigText(mcp) {
  const server = mcp.script_path
    ? { command: mcp.command, args: [mcp.script_path] }
    : { command: mcp.command, args: mcp.args, cwd: mcp.cwd };
  const lines = Object.entries(server)
    .map(([key, value]) => `    ${JSON.stringify(key)}: ${JSON.stringify(value)}`);
  return `{ "mcpServers": { "agentguard": {\n${lines.join(',\n')} } } }`;
}

function showMcpConfig() {
  for (const element of document.querySelectorAll('.mcp-json')) {
    element.textContent = mcpConfigText(state.config.mcp);
  }
  for (const button of document.querySelectorAll('.btn-copy, #btn-show-config')) {
    button.disabled = false;
  }
}

async function loadConfig() {
  try {
    state.config = await getJson('/api/config');
  } catch (error) {
    console.error('Config load failed, retrying:', error);
    $('presets-loading').textContent = 'Presets unavailable. Retrying…';
    setTimeout(loadConfig, 2000);
    return;
  }
  state.provider ??= state.config.default_provider ?? null;
  renderPresets();
  showMcpConfig();
  renderProviders();
  if (!state.hasRun && !state.running) $('c3-provider').textContent = selectedProvider();
  updateControls();
}

function setOverlay(open) {
  $('mcp-overlay').hidden = !open;
  $('btn-show-config').setAttribute('aria-expanded', String(open));
  if (open) $('btn-hide-config').focus();
  else if (!wideLayout.matches) $('btn-show-config').focus();
}

async function copyConfig(button) {
  try {
    await navigator.clipboard.writeText(mcpConfigText(state.config.mcp));
    button.textContent = 'Copied';
    button.classList.add('copied');
  } catch (error) {
    console.error('Copy failed:', error);
    button.textContent = 'Copy failed';
    if (!wideLayout.matches) setOverlay(true);   // so it can be selected by hand
  }
  clearTimeout(button.resetTimer);
  button.resetTimer = setTimeout(() => {
    button.textContent = 'Copy config';
    button.classList.remove('copied');
  }, 2000);
}

/* ---------- Start ---------- */

$('prompt-input').addEventListener('input', () => {
  const preset = presetById(state.presetId);
  if (preset && $('prompt-input').value !== preset.prompt) state.presetId = null;
  updateControls();
});
$('btn-run').addEventListener('click', startRun);
$('btn-reset').addEventListener('click', resetSession);
$('btn-show-config').addEventListener('click', () => setOverlay($('mcp-overlay').hidden));
$('btn-hide-config').addEventListener('click', () => setOverlay(false));
for (const button of document.querySelectorAll('.btn-copy')) {
  button.addEventListener('click', () => copyConfig(button));
}
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !$('mcp-overlay').hidden) setOverlay(false);
});
wideLayout.addEventListener('change', () => {
  if (wideLayout.matches && !$('mcp-overlay').hidden) setOverlay(false);
  renderLedger();
});
window.addEventListener('resize', fitOutputs);
document.fonts.ready.then(fitOutputs);

for (const element of document.querySelectorAll('.mcp-url')) {
  element.textContent = `${window.location.origin}/mcp`;
}

clearCards();
renderProviders();
renderBudget();
updateControls();
loadConfig();
checkHealth();
pollBudget();
pollLedger();
setInterval(() => { pollBudget(); pollLedger(); }, 2000);
setInterval(checkHealth, 10000);
