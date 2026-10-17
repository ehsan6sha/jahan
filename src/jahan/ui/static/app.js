/* Jahan browser interface: every action runs a `jahan` command on this computer (see server.py).
   Nothing is simulated here; this file only builds commands from clicks and shows their output. */
"use strict";
(() => {
  // ------------------------------------------------------------------ helpers
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style") el.style.cssText = v; // CSSOM, allowed by the page's content security policy
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "value") el.value = v;
      else if (k === "checked") el.checked = !!v;
      else el.setAttribute(k, v === true ? "" : String(v));
    }
    const add = (c) => {
      if (c === null || c === undefined || c === false) return;
      if (Array.isArray(c)) c.forEach(add);
      else el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    };
    kids.forEach(add);
    return el;
  }
  const $ = (sel) => document.querySelector(sel);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : n);
  const pad4 = (n) => String(n).padStart(4, "0");
  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem("jahan:" + k)); } catch { return null; } },
    set(k, v) { try { localStorage.setItem("jahan:" + k, JSON.stringify(v)); } catch { /* private mode */ } },
  };
  const quote = (s) => "'" + String(s).replace(/'/g, "") + "'";
  // Free text passed as an option value must not start with a dash (the server refuses option-like values).
  const clean = (s) => String(s).replace(/^[\s-]+/, "").trim();
  const LEANS = [["", "From a resident"], ["0.8", "Firmly with the governing camp"], ["0.5", "Leans to the governing camp"], ["0", "No strong lean"], ["-0.5", "Leans to the opposition"], ["-0.8", "Firmly with the opposition"]];

  // ------------------------------------------------------------------ state
  const S = {
    token: null,
    info: null,
    world: null, // e.g. "runs/live"
    status: null,
    view: "start",
    jobs: [],
    busy: false,
    flash: null, // {kind, text} shown once at the top of the next render
    whatif: { tab: "words", text: "", future: "", preview: null, questions: null, answers: {} },
    social: { country: "", kind: "all", rows: null, loading: null, per: 2, includeX: store.get("includeX") !== false },
    people: { form: { country: "", sex: "", ageMin: "", ageMax: "", info: "", status: "", area: "", figures: "" },
      advanced: null, limit: 25, result: null, selected: null, profile: null },
    ask: { mode: "group", agent: "", form: { country: "", sex: "", ageMin: "", ageMax: "", info: "" }, advanced: null,
      question: "", sample: 5, noLlm: false, result: null },
    play: { as: null, brief: null, circle: null, tab: "message", replies: true, target: "", autopilotDays: 7 },
    results: { what: "", seeds: 4, horizon: "", report: null, reportPath: null },
    run: { autonomy: null, keepIdle: false, custom: 14 },
    watch: { data: null, key: null, loading: null, days: 60, seeds: null, seedsFor: null, force: false,
      custom: { question: "", prior: "0.5", resolves: "", countries: "" }, live: null, liveQuery: "Iran", liveTag: "" },
  };

  const GLOSSARY = {
    prompt: "Anything you add to the world: news, a product, a market move, or an action by someone you play. Prompts are numbered, and you can undo the world to just before any of them.",
    day: "One simulated day. Day 0 is the world's start date.",
    experiment: "Re-runs the world from just before a prompt, once with it and once without, on several random seeds, and compares the two.",
    seed: "A starting point for chance. Different seeds give different but equally plausible histories; several of them show the spread.",
    "information level": "How much news someone follows: minimal, major news, well informed or specialist.",
    visibility: "How many people someone's public statements can reach, on a log scale: 1 is family, 8 is a head of state.",
    repost: "An ordinary person passing a story on in their social feed. It adds a short, weak push on the social channels of their country; it never creates a new story.",
    trending: "Topics many fetched posts were about, counted per country. Shown as context; trending has no effect on the simulation.",
    "political lean": "Where someone stands on their country's main political divide: with the camp of the current head of government, with the main opposition, or neither. It sets how much they trust each public figure to begin with. Camp sizes come from polls or election results; Iran and China use placeholders.",
    trust: "How much one person believes another (0 to 1). It grows when their advice works out and falls faster when it does not.",
    autopilot: "While you are not acting, the simulator decides for your character the way it does for everyone else.",
    "public figure": "A real office holder or opinion leader under their real name. Everything generated for them is labelled [simulated] and is not a real quote.",
    circle: "The people who listen to someone: household, relatives, colleagues, friends and acquaintances.",
    recordings: "Every web fetch and Claude answer is saved inside the world, so undo and replay give identical results without paying again.",
    session: "Your time playing one or more people. Each action is saved, so it can be replayed after an undo.",
    baseline: "The same world without your prompt, with the same random draws, so differences come from the prompt alone.",
    watch: "A series you follow while time moves: a market, a currency, a country's mood or the odds of an event. The change is counted from the day you started watching.",
    question: "A yes/no question about the future, such as whether the US-Iran ceasefire holds. Simulated traders (about one in ten adults online) hold a belief about it, starting near the real market price; the simulated odds are their average. News that bears on it moves the beliefs of the traders who hear it.",
    "people's trading": "Simulated US adults who hold stock (shares from the Federal Reserve's Survey of Consumer Finances) buy or sell when they hear news that makes the economy look better or worse (weekend news is traded on Monday). Their part of the US stock index is shown separately from the part moved by announcements and scripted what-ifs. It fades as the news fades: they buy back what they sold.",
  };
  const term = (key, label) => h("span", { class: "term", title: GLOSSARY[key] || "", tabindex: 0 }, label || key);

  // ------------------------------------------------------------------ token and API
  function readToken() {
    const m = location.hash.match(/t=([A-Za-z0-9_-]+)/);
    if (m) {
      try { sessionStorage.setItem("jahan-token", m[1]); } catch { /* ignore */ }
      history.replaceState(null, "", location.pathname);
      return m[1];
    }
    try { return sessionStorage.getItem("jahan-token"); } catch { return null; }
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      method: opts.method || "GET",
      headers: { "X-Jahan-Token": S.token || "", "Content-Type": "application/json" },
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    let data = null;
    try { data = await res.json(); } catch { /* not JSON */ }
    if (!res.ok) {
      const err = new Error((data && data.error) || res.statusText);
      err.status = res.status;
      throw err;
    }
    return data;
  }

  // ------------------------------------------------------------------ running commands (one at a time)
  let chain = Promise.resolve();
  function run(argv, opts = {}) {
    const p = chain.then(() => exec(argv, opts));
    chain = p.catch(() => {});
    return p;
  }

  async function exec(argv, { quiet = false } = {}) {
    setBusy(true);
    let job;
    try {
      for (let attempt = 0; ; attempt++) {
        try {
          job = await api("/api/jobs", { method: "POST", body: { argv } });
          break;
        } catch (e) {
          if (e.status === 409 && attempt < 900) {
            consoleState("waiting: another command is using this world", true);
            await sleep(1000);
            continue;
          }
          // The server refused or is gone: report it like a failed command so every caller handles it the same way.
          const rec = { id: 0, cmdline: "jahan " + argv.join(" "), argv, lines: [`The page could not start the command: ${e.message}`], status: "failed", started: Date.now(), ended: Date.now() };
          S.jobs.unshift(rec);
          return rec;
        }
      }
      const rec = { id: job.id, cmdline: job.cmdline, argv, lines: [], status: "running", started: Date.now(), quiet };
      S.jobs.unshift(rec);
      consoleStart(rec);
      let since = 0;
      for (;;) {
        await sleep(250);
        const v = await api(`/api/jobs/${job.id}?since=${since}`);
        if (v.lines.length) {
          rec.lines.push(...v.lines);
          if (!quiet) consoleAppend(v.lines);
        }
        since = v.next;
        if (v.status !== "running") {
          rec.status = v.status;
          rec.exit = v.exit_code;
          rec.ended = Date.now();
          break;
        }
      }
      consoleDone(rec);
      return rec;
    } finally {
      setBusy(false);
      if (S.view === "activity") render();
    }
  }

  function lastJson(rec) {
    for (let i = rec.lines.length - 1; i >= 0; i--) {
      const t = rec.lines[i].trim();
      if (t.startsWith("{")) return JSON.parse(t);
    }
    throw new Error("the command printed no JSON");
  }

  async function runJSON(argv) {
    const rec = await run(argv, { quiet: true });
    if (rec.status !== "ok") throw jobError(rec);
    return lastJson(rec);
  }

  // Typer prints errors in a box; this turns the box into one readable sentence.
  function friendly(lines) {
    const boxed = lines.filter((l) => l.includes("│")).map((l) => l.replace(/[─-╿]/g, " ").trim()).filter(Boolean);
    let text = boxed.join(" ").replace(/\s+/g, " ").replace(/^Invalid value:\s*/i, "").replace(/^Error\s*/i, "");
    if (!text) {
      const tail = lines.filter((l) => l.trim()).slice(-3);
      text = tail.join(" ") || "The command failed without a message.";
    }
    if (/^Aborted!?$/.test(text)) text = "Cancelled.";
    return text;
  }

  function jobError(rec) {
    const e = new Error(rec.status === "cancelled" ? "Cancelled." : friendly(rec.lines));
    e.rec = rec;
    return e;
  }

  function errorNotice(err) {
    const rec = err.rec;
    return h("div", { class: "notice error", role: "alert" },
      h("strong", {}, "That did not work"),
      h("div", {}, err.message),
      rec ? h("details", {}, h("summary", {}, "Full output"), h("pre", { class: "out" }, rec.lines.join("\n")), h("div", { class: "cmd" }, rec.cmdline)) : null);
  }

  function outputBlock(rec, title) {
    return h("div", {},
      title ? h("h3", {}, title) : null,
      h("pre", { class: "out" }, rec.lines.join("\n") || "(no output)"),
      h("div", { class: "cmd", title: "The command this ran" }, rec.cmdline));
  }

  // Runs a command that changes the world, then refreshes the status. Success or the error is shown in `box`;
  // with `rerender` the page is redrawn and the success message is carried to the top of the new page.
  async function act(box, argv, { ok, then, rerender } = {}) {
    if (box) box.replaceChildren(h("div", { class: "notice" }, "Working… (see the command output below)"));
    try {
      const rec = await run(argv);
      if (rec.status !== "ok") throw jobError(rec);
      await refreshStatus();
      if (rerender) {
        S.flash = { kind: "ok", text: ok || "Done.", cmd: rec.cmdline, extra: rerender === true ? null : rerender(rec) };
        render();
      } else if (box) {
        box.replaceChildren(h("div", { class: "notice ok" }, h("strong", {}, ok || "Done."), h("div", { class: "cmd" }, rec.cmdline)));
      }
      if (then) await then(rec);
      return rec;
    } catch (e) {
      if (box) box.replaceChildren(errorNotice(e));
      return null;
    }
  }

  // ------------------------------------------------------------------ console drawer
  // The drawer shows the last command you asked for; background reads (status, lists) only show "reading…".
  function consoleStart(rec) {
    const cancel = $("#console-cancel");
    cancel.hidden = false;
    cancel.onclick = () => {
      consoleState("stopping…", true);  // the world is free again once the command has actually stopped
      api(`/api/jobs/${rec.id}/cancel`, { method: "POST", body: {} }).catch(() => {});
    };
    if (rec.quiet) {
      consoleState("reading the world…", true);
      return;
    }
    $("#console-cmd").textContent = rec.cmdline;
    $("#console-out").textContent = "";
    consoleState("running", true);
    if (store.get("consoleOpen") !== false) setConsole(true);
  }
  function consoleAppend(lines) {
    const out = $("#console-out");
    const atEnd = out.scrollTop + out.clientHeight >= out.scrollHeight - 30;
    out.textContent += lines.join("\n") + "\n";
    if (atEnd) out.scrollTop = out.scrollHeight;
  }
  let lastShown = null;
  function consoleDone(rec) {
    $("#console-cancel").hidden = true;
    if (!rec.quiet) lastShown = rec;
    const shown = lastShown || rec;
    const secs = ((shown.ended - shown.started) / 1000).toFixed(1);
    consoleState(rec.quiet && !lastShown ? "" : { ok: `finished in ${secs}s`, failed: `failed after ${secs}s`, cancelled: "cancelled" }[shown.status] || shown.status, false);
  }
  function consoleState(text, running) {
    const el = $("#console-state");
    el.textContent = text;
    el.classList.toggle("running", !!running);
  }
  function setConsole(open) {
    $("#console").classList.toggle("open", open);
    $("#console-toggle").setAttribute("aria-expanded", String(open));
  }
  function setBusy(b) {
    S.busy = b;
    document.body.classList.toggle("busy", b);
  }

  // ------------------------------------------------------------------ status and chrome
  async function refreshInfo() {
    S.info = await api("/api/info");
    const sel = $("#world-select");
    sel.replaceChildren(
      h("option", { value: "" }, S.info.worlds.length ? "Choose a world…" : "No worlds yet"),
      ...S.info.worlds.map((w) => h("option", { value: w.path }, `${w.name}  (day ${w.day})`)));
    sel.value = S.world || "";
  }

  async function refreshStatus() {
    if (!S.world) { S.status = null; renderChrome(); return; }
    try {
      S.status = await runJSON(["status", "--run", S.world, "--json"]);
    } catch (e) {
      // A world this version cannot read (older format, changed data files): leave it, say why, go to Start.
      S.flash = { kind: "error", text: `Could not open ${S.world}: ${e.message}` };
      S.status = null;
      S.world = null;
      store.set("world", null);
      $("#world-select").value = "";
      S.view = "start";
    }
    renderChrome();
  }

  async function openWorld(path, view) {
    S.world = path || null;
    store.set("world", S.world);
    S.people.result = null; S.people.selected = null; S.people.profile = null;
    Object.assign(S.play, { brief: null, briefing: null, circle: null, circling: null, target: "" });
    S.results.report = null; S.results.reportPath = null; S.results.what = ""; S.whatif.preview = null; S.ask.result = null;
    $("#world-select").value = S.world || "";
    go(S.world ? view || "world" : "start");
    await refreshStatus();
    render();
  }

  function renderChrome() {
    const chips = $("#chips");
    chips.replaceChildren();
    const st = S.status;
    if (st) {
      const ahead = st.days_ahead_of_today;
      const when = ahead === 0 ? "today" : ahead > 0 ? `${ahead} days after today` : `${-ahead} days before today`;
      chips.append(h("span", { class: "chip", title: "World date" }, `Day ${st.day} · ${st.date} · ${when}`));
      const me = st.session.agents;
      if (me.length) chips.append(h("span", { class: "chip ok", title: "You are playing" }, "Playing: " + me.map((a) => a.name).join(", ")));
      const spent = st.llm.spent_in_this_world;
      chips.append(st.llm.ready
        ? h("span", { class: "chip ok", title: "Claude answers recorded in this world, at list prices" }, `Claude ready · $${spent.usd.toFixed(3)} spent here`)
        : h("span", { class: "chip warn" }, "Claude not set up ", h("button", { onclick: () => go("start") }, "how?")));
    } else if (S.info) {
      const ready = S.info.llm.api_key_set && S.info.llm.sdk_installed;
      chips.append(h("span", { class: ready ? "chip ok" : "chip warn" }, ready ? "Claude ready" : "Claude not set up"));
    }
    renderNav();
  }

  // ------------------------------------------------------------------ navigation
  const VIEWS = [
    { id: "start", title: "Start", hint: "Create or open a world" },
    { id: "world", title: "World", hint: "Where things stand; move time", world: true },
    { id: "whatif", title: "What-if", hint: "Add news, products, market moves", world: true },
    { id: "social", title: "Social", hint: "Real posts, trending, posts in the world", world: true },
    { id: "people", title: "People", hint: "Find anyone; see their view", world: true },
    { id: "ask", title: "Ask", hint: "Interview a person or a group", world: true },
    { id: "play", title: "Play", hint: "Become someone and act", world: true },
    { id: "results", title: "Results", hint: "Measure what a prompt changed", world: true },
    { id: "watch", title: "Watch", hint: "Follow markets and odds as time moves", world: true },
    null,
    { id: "activity", title: "Activity", hint: "Every command that ran" },
    { id: "guide", title: "Guide & tools", hint: "How it works; checks" },
  ];

  function renderNav() {
    const nav = $("#sidebar");
    nav.replaceChildren(...VIEWS.map((v) => {
      if (!v) return h("div", { class: "nav-sep" });
      const disabled = v.world && !S.world;
      return h("button", {
        class: "nav-item", "aria-current": S.view === v.id ? "page" : null, disabled,
        title: disabled ? "Open or create a world first" : v.hint,
        onclick: () => { go(v.id); $("#sidebar").classList.remove("open"); },
      }, h("span", { class: "nav-title" }, v.title), h("span", { class: "nav-hint" }, v.hint));
    }), h("p", { class: "nav-disclaimer" },
      "Educational and testing use only. Not financial or any other advice; not a forecast. No warranty; use at your own risk (Apache-2.0)."));
  }

  function go(view) {
    S.view = view;
    store.set("view", view);
    render();
    const main = $("#main");
    main.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }

  function render() {
    renderNav();
    const main = $("#main");
    const flash = S.flash;
    S.flash = null;
    const body = (RENDER[S.view] || RENDER.start)();
    const note = flash
      ? h("div", { class: `notice ${flash.kind}`, role: "status" }, h("strong", {}, flash.text), flash.extra || null, flash.cmd ? h("div", { class: "cmd" }, flash.cmd) : null)
      : "";
    main.replaceChildren(note, body);
  }

  const page = (title, lead, ...kids) => h("div", {}, h("h1", {}, title), lead ? h("p", { class: "lead" }, lead) : null, ...kids);
  const card = (title, hint, ...kids) => h("section", { class: "card" }, title ? h("h3", {}, title) : null, hint ? h("p", { class: "hint" }, hint) : null, ...kids);
  const tile = (title, text, onclick) => h("button", { class: "tile", onclick }, h("strong", {}, title), h("span", {}, text));
  const loading = (text) => h("p", { class: "muted" }, text || "Loading…");

  function field(label, input, hint) {
    return h("label", { class: "field" }, h("span", { class: "lbl" }, label), input, hint ? h("span", {}, hint) : null);
  }
  function select(options, value, onchange, attrs = {}) {
    return h("select", { ...attrs, onchange: (e) => onchange(e.target.value) },
      ...options.map(([v, label]) => h("option", { value: v, selected: String(v) === String(value) ? true : null }, label)));
  }
  function suggestions(list, onpick) {
    return h("div", { class: "suggest" }, h("span", { class: "muted small" }, "Try:"), ...list.map((s) => h("button", { type: "button", onclick: () => onpick(s) }, s)));
  }

  // ------------------------------------------------------------------ Start
  function viewStart() {
    const info = S.info;
    const out = h("div", {});
    const llmCard = () => {
      const l = info ? info.llm : null;
      if (!l) return loading();
      if (l.sdk_installed && l.api_key_set) {
        return h("div", {}, h("p", {}, "Claude is set up for this session. Commands that need it print their approximate cost."),
          healthButtons());
      }
      return h("div", {},
        h("div", { class: "notice warn" },
          h("strong", {}, "Claude is not set up yet"),
          "Everything except free-text prompts, interviews, prose and public figures' own decisions works without it."),
        !l.sdk_installed ? h("p", {}, "1. Install the Claude add-on, then restart this page: ", h("code", {}, "uv sync --extra llm")) : null,
        !l.api_key_set ? h("p", {}, (l.sdk_installed ? "" : "2. ") + "Stop this page (Ctrl+C in its terminal) and start it with your own Anthropic API key: ",
          h("code", {}, "uv run jahan ui --key-file $HOME\\.anthropic_key"),
          " (a file holding only the key), or set ", h("code", {}, "ANTHROPIC_API_KEY"), " before starting.") : null,
        healthButtons());
    };
    out.append(page("Welcome to Jahan",
      "Jahan simulates thousands of people in eight countries: their families, work, news, conversations, trust and life events. You add what-ifs, move time forward, look through anyone's eyes, and measure what changed. Every button here runs a jahan command on this computer; the exact command is shown below each result, so you can also type it yourself later."));
    const next = [];
    if (!info || !info.worlds.length) next.push(tile("1. Create a world", "Takes a few seconds. Start below.", () => $("#world-name") && $("#world-name").focus()));
    else next.push(tile("Open your latest world", `${info.worlds[0].name}, day ${info.worlds[0].day}`, () => openWorld(info.worlds[0].path)));
    next.push(tile("Read the guide", "Concepts in two minutes", () => go("guide")));
    out.append(h("div", { class: "next-steps" }, next));
    out.append(card("Your worlds", "Each world is a folder under " + (info ? info.runs : "runs") + ". Open one to continue where you left off.", worldsTable()));
    out.append(createWorldCard());
    out.append(card("Claude (optional)", null, llmCard()));
    return out;
  }

  function healthButtons() {
    const box = h("div", {});
    return h("div", {},
      h("div", { class: "row" },
        h("button", { class: "btn act", onclick: async () => box.replaceChildren(outputBlock(await run(["doctor"]), "Health check")) }, "Run health check"),
        h("button", { class: "btn act", title: "Sends one tiny request per model (a fraction of a cent)", onclick: async () => box.replaceChildren(outputBlock(await run(["doctor", "--check-models"]), "Model check")) }, "Check Claude models")),
      box);
  }

  function worldsTable() {
    const info = S.info;
    if (!info) return loading();
    if (!info.worlds.length) return h("p", { class: "muted" }, "No worlds yet. Create one below.");
    return h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "World"), h("th", { class: "num" }, "Day"), h("th", {}, "Started"), h("th", { class: "num" }, "Prompts"), h("th", {}, ""))),
      h("tbody", {}, ...info.worlds.map((w) => h("tr", { class: "clickable", onclick: () => openWorld(w.path) },
        h("td", {}, h("strong", {}, w.name), h("div", { class: "muted small" }, w.path)),
        h("td", { class: "num" }, w.day), h("td", {}, w.start_date || ""), h("td", { class: "num" }, w.prompts),
        h("td", {}, h("button", { class: "btn small" }, "Open")))))));
  }

  function createWorldCard() {
    const info = S.info;
    const countries = info ? info.countries : ["USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"];
    const today = info ? info.today : new Date().toISOString().slice(0, 10);
    const f = { name: "world-" + today.replace(/-/g, ""), size: "10000", countries: new Set(countries), seed: "", start: today, replace: false };
    const box = h("div", {});
    const nameInput = h("input", { id: "world-name", value: f.name, oninput: (e) => (f.name = e.target.value.trim()) });
    const boxes = countries.map((c) => h("label", { class: "check" }, h("input", { type: "checkbox", checked: true, onchange: (e) => (e.target.checked ? f.countries.add(c) : f.countries.delete(c)) }), c));
    const submit = async () => {
      if (!/^[A-Za-z0-9_-]+$/.test(f.name)) { box.replaceChildren(h("div", { class: "notice error" }, "Use letters, digits, - or _ for the name.")); return; }
      if (!f.countries.size) { box.replaceChildren(h("div", { class: "notice error" }, "Pick at least one country.")); return; }
      const path = `${info ? info.runs : "runs"}/${f.name}`;
      const all = f.countries.size === countries.length;
      const argv = ["gen", "--run", path];
      if (f.size === "10000" && all) argv.push("--config", "configs/m1.yaml");
      else argv.push("--agents", f.size, "--countries", [...f.countries].join(","));
      if (f.seed) argv.push("--seed", f.seed);
      if (f.start) argv.push("--start-date", f.start);
      if (f.replace) argv.push("--force");
      const rec = await act(box, argv, { ok: "World created." });
      if (rec) { await refreshInfo(); await openWorld(path, "world"); }
    };
    return card("Create a new world", "People, households, jobs, friendships and media are generated from published statistics. A full world (10,000 people) takes a few seconds.",
      h("div", { class: "form-grid" },
        field("Name", nameInput),
        field("Size", select([["1000", "1,000 people (quick)"], ["3000", "3,000 people"], ["10000", "10,000 people (full)"]], f.size, (v) => (f.size = v))),
        field("Start date", h("input", { type: "date", value: f.start, onchange: (e) => (f.start = e.target.value) }), "Today means real news can be added."),
        field("Seed (optional)", h("input", { type: "number", placeholder: "random", oninput: (e) => (f.seed = e.target.value) }), "Same seed, same world.")),
      h("fieldset", {}, h("legend", {}, "Countries"), h("div", { class: "row" }, boxes)),
      h("div", { class: "row" },
        h("button", { class: "btn primary act", onclick: submit }, "Create world"),
        h("label", { class: "check small" }, h("input", { type: "checkbox", onchange: (e) => (f.replace = e.target.checked) }), "Replace a world with the same name")),
      box);
  }

  // ------------------------------------------------------------------ World
  function viewWorld() {
    const st = S.status;
    if (!st) return page("World", null, loading("Reading the world…"));
    const name = S.world.split("/").pop();
    const spent = st.llm.spent_in_this_world;
    const out = page(name, `Day ${st.day} is ${st.date}. ${describeDate(st)}`);
    out.append(h("div", { class: "grid four" },
      stat("People", fmt(st.agents_alive), `${st.countries.length} countries`),
      stat("Prompts", st.prompts.length, st.prompts.length ? `last on day ${st.prompts[st.prompts.length - 1].day}` : "none yet"),
      stat("Products", st.products.length, st.products.join(", ") || "none launched"),
      stat("Claude spent here", `$${spent.usd.toFixed(3)}`, `${fmt(spent.input_tokens + spent.output_tokens)} tokens`)));
    out.append(h("div", { style: "height:14px" }));
    out.append(timeCard());
    out.append(h("h3", {}, "What next?"), h("div", { class: "next-steps" }, nextSteps(st)));
    out.append(promptsCard(st));
    out.append(card("People by country", "How many simulated people live in each country, and the languages they speak (news reaches only people who understand it).",
      h("div", { class: "bars" }, ...st.countries.map((c) => {
        const max = Math.max(...st.countries.map((x) => x.agents));
        return h("div", { class: "bar-row" }, h("strong", {}, c.iso3), h("div", { class: "bar", title: c.name }, h("i", { style: `width:${(100 * c.agents) / max}%` })), h("span", { class: "muted" }, fmt(c.agents)));
      })),
      h("p", { class: "muted small", style: "margin-top:10px" }, st.countries.map((c) => `${c.iso3}: ${c.languages.filter((l) => l !== "other").join(", ")}`).join(" · "))));
    return out;
  }

  function describeDate(st) {
    const a = st.days_ahead_of_today;
    if (a === 0) return "That is today, so today's real headlines can be added.";
    if (Math.abs(a) <= 3) return `That is ${Math.abs(a)} day(s) ${a > 0 ? "after" : "before"} today, close enough to add real headlines.`;
    if (a > 0) return `That is ${a} days after today: there is no real news yet, so add hypothetical events if you want any.`;
    return `That is ${-a} days before today.`;
  }

  function stat(label, value, sub) {
    return h("div", { class: "stat" }, h("div", { class: "label" }, label), h("div", { class: "value" }, value), h("div", { class: "sub" }, sub));
  }

  function timeCard() {
    const st = S.status;
    const box = h("div", {});
    if (S.run.autonomy === null) S.run.autonomy = !!st.llm.ready;
    const playing = st.session.agents.length > 0;
    const advance = async (days) => {
      const argv = ["run", "--run", S.world, "--days", String(days), S.run.autonomy ? "--autonomy" : "--no-autonomy"];
      if (playing && S.run.keepIdle) argv.push("--keep-control");
      const before = st.day;
      await act(box, argv, {
        ok: `Moved ${days} day(s) forward: day ${before} → ${before + days}.`,
        rerender: (rec) => {
          const note = rec.lines.find((l) => l.startsWith("Note:"));
          const figures = rec.lines.find((l) => l.startsWith("Public figures took"));
          const off = rec.lines.find((l) => l.startsWith("This run,"));
          return h("div", {},
            figures ? h("div", {}, figures) : null,
            off ? h("div", {}, off) : null,
            note ? h("div", {}, "These days are after today's real date, so no real news exists for them. ",
              h("button", { class: "btn link", onclick: () => { S.whatif.tab = "future"; go("whatif"); } }, "Add hypothetical events")) : null);
        },
      });
    };
    return card("Move time forward", "Each day, people wake and sleep on local time, read the news they follow, talk to their circle, decide, marry, have children, change jobs and travel.",
      h("div", { class: "row" },
        h("button", { class: "btn primary act", onclick: () => advance(1) }, "+1 day"),
        h("button", { class: "btn act", onclick: () => advance(7) }, "+7 days"),
        h("button", { class: "btn act", onclick: () => advance(30) }, "+30 days"),
        h("span", { class: "muted" }, "or"),
        h("input", { type: "number", min: 1, max: 3650, value: S.run.custom, oninput: (e) => (S.run.custom = e.target.value), "aria-label": "Days" }),
        h("button", { class: "btn act", onclick: () => advance(Math.max(1, parseInt(S.run.custom, 10) || 1)) }, "Run")),
      h("div", { class: "row", style: "margin-top:10px" },
        h("label", { class: "check small", title: "Public figures nobody plays get a decision turn when something relevant happens (uses Claude; capped)" },
          h("input", { type: "checkbox", checked: S.run.autonomy, disabled: !st.llm.ready, onchange: (e) => (S.run.autonomy = e.target.checked) }),
          "Let ", term("public figure", "public figures"), " react on their own", st.llm.ready ? "" : " (needs Claude)"),
        playing ? h("label", { class: "check small", title: "Otherwise the simulator acts for your characters while time runs" },
          h("input", { type: "checkbox", checked: S.run.keepIdle, onchange: (e) => (S.run.keepIdle = e.target.checked) }),
          "Keep my characters idle instead of on ", term("autopilot")) : null),
      box);
  }

  function nextSteps(st) {
    const tiles = [];
    if (st.ingest_allowed && !st.prompts.some((p) => p.source === "ingest")) tiles.push(tile("Add today's real news", "Top headlines per country, in their own languages", () => { S.whatif.tab = "news"; go("whatif"); }));
    if (st.ingest_allowed && !(st.social && st.social.last_fetch)) tiles.push(tile("Read today's social posts", "Posts by public figures and channels, from X, Bluesky, Mastodon and Telegram", () => go("social")));
    tiles.push(tile("Add a what-if", "A product, news, a market move, in plain words", () => { S.whatif.tab = "words"; go("whatif"); }));
    if (st.day < 7) tiles.push(tile("Move time forward", "Let a week pass and see what spreads", () => $(".btn.primary.act") && $(".btn.primary.act").focus()));
    tiles.push(tile("Look through someone's eyes", "Find a person and open their view", () => go("people")));
    tiles.push(tile(st.session.agents.length ? "Continue playing" : "Play as someone", st.session.agents.length ? `You play ${st.session.agents.map((a) => a.name).join(", ")}` : "Even a president: post, message, use powers", () => go("play")));
    if (st.prompts.length) tiles.push(tile("Measure an effect", "Compare the world with and without a prompt", () => go("results")));
    return tiles;
  }

  function promptsCard(st) {
    const box = h("div", {});
    const list = st.prompts.length
      ? h("ul", { class: "timeline" }, ...st.prompts.slice().reverse().map((p) => h("li", {},
        h("div", { class: "when" }, `#${p.prompt_id}`, h("br"), `day ${p.day}`),
        h("div", { class: "what" }, h("div", {}, p.prompt || "(no text)"), h("div", { class: "muted small" }, `${p.date} · `, sourceTag(p))),
        h("div", { class: "acts" },
          p.scenario_file ? h("button", { class: "btn small", onclick: () => showFile(box, p.scenario_file, `Prompt ${p.prompt_id} as events`) }, "Events") : null,
          p.session ? null : h("button", { class: "btn small", onclick: () => { S.results.what = `prompt:${p.prompt_id}`; go("results"); } }, "Measure"),
          h("button", { class: "btn small danger act", title: "Restore the world to just before this prompt and discard everything after it", onclick: () => undo(box, p) }, "Undo from here")))))
      : h("p", { class: "muted" }, "Nothing added yet. Use What-if to add news, products or market moves; time can also just run.");
    return card("Prompts so far", h("span", {}, "Every change you make is a numbered ", term("prompt"), ". Undo returns the world to just before one; replaying is exact thanks to ", term("recordings"), "."), list, box);
  }

  function sourceTag(p) {
    const label = p.session ? `your session ${p.session}` : { ingest: "real news & posts", llm: "your words", feed: "future events" }[p.source] || (p.source.endsWith(".yaml") ? "example" : p.source);
    return h("span", { class: "tag" }, label);
  }

  async function undo(box, p) {
    if (!window.confirm(`Undo prompt #${p.prompt_id} and everything after it? The world returns to day ${p.day}, just before it. Reports already written stay on disk.`)) return;
    await act(box, ["reset", "--run", S.world, "--before-prompt", String(p.prompt_id)], { ok: `The world is back to day ${p.day}, just before prompt #${p.prompt_id}.`, rerender: true });
  }

  async function showFile(box, rel, title) {
    try {
      const { text } = await api(`/api/file?world=${encodeURIComponent(S.world)}&path=${encodeURIComponent(rel)}`);
      box.replaceChildren(card(title, null, h("pre", { class: "out" }, text)));
    } catch (e) {
      box.replaceChildren(errorNotice(e));
    }
  }

  // ------------------------------------------------------------------ What-if
  function viewWhatif() {
    const st = S.status;
    if (!st) return page("What-if", null, loading());
    const W = S.whatif;
    const tabs = [["words", "Describe in words"], ["future", "Future events"], ["examples", "Ready-made examples"], ["news", "Today's real news"]];
    const out = page("What-if", "Change the world: launch a product, spread news, move a currency or gold. Your words become concrete, checkable events that you confirm before anything changes.");
    out.append(h("div", { class: "row", role: "tablist", style: "margin-bottom:14px" }, ...tabs.map(([id, label]) =>
      h("button", { class: W.tab === id ? "btn primary" : "btn", role: "tab", "aria-selected": String(W.tab === id), onclick: () => { W.tab = id; W.preview = null; W.questions = null; render(); } }, label))));
    out.append({ words: whatifWords, future: whatifFuture, examples: whatifExamples, news: whatifNews }[W.tab](st));
    return out;
  }

  function needClaude(st) {
    return st.llm.ready ? null : h("div", { class: "notice warn" }, h("strong", {}, "This needs Claude."), "Start the page with your API key (see Start). Ready-made examples work without it.");
  }

  function whatifWords(st) {
    const W = S.whatif;
    const ta = h("textarea", { placeholder: "e.g. Launch a $199 smart ring in the US, Germany, India and Iran with a month of ads", oninput: (e) => (W.text = e.target.value) }, W.text);
    const box = h("div", {});
    return h("div", {},
      card("Describe what happens", h("span", {}, `Happens on day ${st.day} (${st.date}) unless you say “in two weeks” and so on. Claude turns it into events; you check them before they are added.`),
        needClaude(st), ta,
        suggestions([
          "Launch a $199 smart ring in the US, Germany, India and Iran",
          "Iran-US tensions intensify: alarming news in Iran and the US all week, and the rial falls 30% against the dollar over two weeks",
          "The US dollar rises 20% against every other currency over four weeks",
          "A rumour spreads in Brazil that a popular bank is about to fail",
        ], (s) => { W.text = s; ta.value = s; }),
        h("div", { class: "row" }, h("button", { class: "btn primary act", disabled: !st.llm.ready, onclick: () => previewText(box, "inject") }, "Preview events")),
        box),
      W.preview && W.preview.kind === "inject" ? previewCard(W.preview, box) : null);
  }

  function whatifFuture(st) {
    const W = S.whatif;
    const ta = h("textarea", { placeholder: "e.g. gold rises 1% a day for a week, then drops 10% in one day", oninput: (e) => (W.future = e.target.value) }, W.future);
    const box = h("div", {});
    const after = st.days_ahead_of_today >= 0 ? "Real news ends today" : "Real news exists up to today";
    return card("Feed events for days with no real news", `${after}; for later dates you can describe what you want to happen. Timing is relative to the first day without real news.`,
      needClaude(st), ta,
      suggestions(["gold rises 1% a day for a week, then drops 10% in one day", "the Nigerian naira falls 25% against the dollar over a month", "Iran-US war intensifies next week", "a major earthquake hits Indonesia in ten days"], (s) => { W.future = s; ta.value = s; }),
      h("div", { class: "row" }, h("button", { class: "btn primary act", disabled: !st.llm.ready, onclick: () => previewText(box, "feed") }, "Preview events")),
      box, W.preview && W.preview.kind === "feed" ? previewCard(W.preview, box) : null);
  }

  async function previewText(box, kind) {
    const W = S.whatif;
    let text = clean(kind === "feed" ? W.future : W.text);
    if (!text) { box.replaceChildren(h("div", { class: "notice error" }, "Write something first.")); return; }
    const answered = Object.entries(W.answers).filter(([, a]) => a.trim());
    if (answered.length) text += "\nDetails: " + answered.map(([q, a]) => `${q} ${a}`).join(" ");
    const argv = kind === "feed"
      ? ["feed", "--run", S.world, "--preview", "--json", "--", text]
      : ["inject", "--run", S.world, "--prompt", text, "--preview", "--json"];
    box.replaceChildren(h("div", { class: "notice" }, "Claude is reading your text…"));
    const rec = await run(argv);
    const marker = rec.lines.findIndex((l) => l.startsWith("The model needs more detail"));
    if (marker >= 0) {
      W.questions = rec.lines.slice(marker + 1).filter((l) => l.startsWith("- ")).map((l) => l.slice(2));
      box.replaceChildren(questionsCard(W.questions, () => previewText(box, kind)));
      return;
    }
    if (rec.status !== "ok") { box.replaceChildren(errorNotice(jobError(rec))); return; }
    const data = lastJson(rec);
    W.preview = { kind, text, day: data.day, scenario: data.scenario, moves: data.market_moves || [], odds: data.odds_effects || [], cmdline: rec.cmdline };
    W.answers = {};
    render();
  }

  function questionsCard(questions, retry) {
    const W = S.whatif;
    return card("Claude needs a little more detail", "Answer what you can, then preview again.",
      ...questions.map((q) => field(q, h("input", { value: W.answers[q] || "", oninput: (e) => (W.answers[q] = e.target.value) }))),
      h("div", { class: "row", style: "margin-top:10px" }, h("button", { class: "btn primary act", onclick: retry }, "Preview again")));
  }

  function previewCard(P, _box) {
    const st = S.status;
    const box = h("div", {});
    const stale = st.day !== P.day;
    const confirm = async () => {
      const argv = P.kind === "feed"
        ? ["feed", "--run", S.world, "--yes", "--", P.text]
        : P.kind === "scenario"
          ? ["inject", "--run", S.world, "--scenario", P.path, "--yes"]
          : ["inject", "--run", S.world, "--prompt", P.text, "--yes"];
      const rec = await act(box, argv, { ok: "Added to the world." });
      if (rec) {
        S.whatif.preview = null;
        S.flash = { kind: "ok", text: "Added. Now move time forward on the World page to see it spread, or measure its effect under Results." };
        go("world");
      }
    };
    return card("Check before adding", null,
      h("p", {}, "These events will be added to the world on day ", h("strong", {}, P.day), ":"),
      h("ul", { class: "events-list" }, ...(P.scenario.events || []).map((e) => h("li", {}, describeEvent(e)))),
      ...(P.moves || []).map(marketMoves),
      ...(P.odds || []).map(oddsEffect),
      P.scenario.horizon_days ? h("p", { class: "muted small" }, `Effects are followed for ${P.scenario.horizon_days} days when you measure them.`) : null,
      h("details", {}, h("summary", {}, "Exact events (as the simulator stores them)"), h("pre", { class: "out" }, JSON.stringify(P.scenario, null, 2))),
      stale ? h("div", { class: "notice warn" }, `The world moved to day ${st.day} since this preview; preview again so you add what you saw.`) : null,
      h("div", { class: "row", style: "margin-top:10px" },
        h("button", { class: "btn primary act", disabled: stale, onclick: confirm }, "Add to the world"),
        h("button", { class: "btn", onclick: () => { S.whatif.preview = null; render(); } }, "Discard")),
      h("div", { class: "cmd" }, P.cmdline), box);
  }

  function marketMoves(r) {
    const moved = (r.moves || []).filter((m) => m.pct !== 0);
    return h("div", { class: "notice" },
      h("strong", {}, `Markets on “${r.headline}”: `),
      moved.length
        ? moved.map((m) => `${m.label} ${m.pct > 0 ? "+" : ""}${m.pct.toFixed(2)}%${m.active ? "" : " (shown only, not applied)"}`).join("; ")
        : "no move: markets expected exactly this.",
      h("div", { class: "muted small" }, "Computed by the engine from published estimates times the surprise (data/global/markets.yaml), not by the simulated people and not by Claude. Educational use only; not a forecast."));
  }

  function oddsEffect(r) {
    return h("div", { class: r.known ? "notice" : "notice warn" },
      h("strong", {}, `“${r.headline}” bears on the watched question “${r.question}”: `),
      `direction ${r.direction > 0 ? "+" : ""}${r.direction}, so people who know the question and hear this story ${r.direction > 0 ? "raise" : "lower"} their odds of Yes.`,
      h("div", { class: "muted small" }, r.known ? "How far the odds move is set by the engine from who hears the story, not by the text." : "This is not a watched question; adding would be refused."));
  }

  function describeEvent(e) {
    const where = (list) => (list && list.length ? list.join(", ") : "every country");
    const when = (off) => (off ? ` from day +${off}` : "");
    switch (e.type) {
      case "product_launch":
        return `Product “${e.name}” at $${e.price_usd} in ${where(e.countries)}${when(e.available_from_day_offset)}; advertised ${((e.pushes || [])[0] || {}).duration_days || 0} days on ${(((e.pushes || [])[0] || {}).channel_kinds || []).join(", ")}.`;
      case "news":
        return `News: “${e.headline}” in ${where(e.countries)}${e.language ? ` (${e.language})` : ""}${when(e.start_day_offset)} for ${e.duration_days} days; tone ${e.valence > 0 ? "good" : e.valence < 0 ? "bad" : "neutral"} (${e.valence}).` + ((e.bears_on || []).length ? ` Bears on ${e.bears_on.map((b) => `${b.question} (${b.direction > 0 ? "+" : ""}${b.direction})`).join(", ")}.` : "");
      case "macro":
        return `${{ fx_local_per_usd: "Exchange rate (local currency per dollar)", gold: "Gold price", us_stocks: "US stocks (S&P 500)", sentiment: "Economic mood" }[e.variable] || e.variable} in ${e.country || "the whole world"}: ` +
          (e.steps || []).map((s) => `${s.pct_per_day > 0 ? "+" : ""}${s.pct_per_day}% a day for ${s.days} day(s)${when(s.start_day_offset)}`).join(", then ") + ".";
      case "announcement":
        return `${{ us_policy_rate: "Federal Reserve decision", us_payrolls: "US jobs report" }[e.indicator] || e.indicator}${when(e.start_day_offset)}: “${e.headline}”. Announced ${e.actual}, markets expected ${e.expected}${e.guidance_bp ? `, guidance ${e.guidance_bp > 0 ? "+" : ""}${e.guidance_bp} bp on the 2-year yield` : ""}; news in ${where(e.countries)}.`;
      case "statement":
        return `Statement by ${String(e.speaker).replace(/^[A-Z]{3}\./, "").replace(/_/g, " ")} (${String(e.speaker).slice(0, 3)})${when(e.start_day_offset)}: “${e.headline}”; tone ${e.valence > 0 ? "good" : e.valence < 0 ? "bad" : "neutral"}.`;
      case "social":
        return `Real post by ${e.author_handle ? "@" + e.author_handle : e.author_name || "a user"} on ${({ x: "X", bluesky: "Bluesky", mastodon: "Mastodon", telegram: "Telegram" })[e.platform] || e.platform} in ${where(e.countries)} (${e.language}): “${e.headline}”; tone ${e.valence > 0 ? "good" : e.valence < 0 ? "bad" : "neutral"}; social channels for ${e.duration_days} days.`;
      case "question":
        return e.action === "close" ? `Stop watching the question ${e.id}.` : `Watch the question “${e.question}”${e.price !== null && e.price !== undefined ? `, starting at the market price ${(100 * e.price).toFixed(1)}% (${e.fetch_date})` : `, starting at ${(100 * (e.prior || 0.5)).toFixed(0)}%`}${e.resolves_by ? `; resolves ${e.resolves_by}` : ""}.`;
      case "social_digest":
        return `What the fetch saw (${e.fetch_day}): ${(e.seen || []).length} posts, trending topics and figures' recent posts, kept for display and context.`;
      default:
        return `${e.type.replace(/_/g, " ")}: ${JSON.stringify(e).slice(0, 160)}`;
    }
  }

  function whatifExamples(st) {
    const info = S.info;
    const box = h("div", {});
    const W = S.whatif;
    const preview = async (ex) => {
      box.replaceChildren(h("div", { class: "notice" }, "Reading the example…"));
      const rec = await run(["inject", "--run", S.world, "--scenario", ex.path, "--preview", "--json"], { quiet: true });
      if (rec.status !== "ok") { box.replaceChildren(errorNotice(jobError(rec))); return; }
      const data = lastJson(rec);
      W.preview = { kind: "scenario", path: ex.path, day: data.day, scenario: data.scenario, moves: data.market_moves || [], odds: data.odds_effects || [], cmdline: rec.cmdline };
      render();
    };
    return h("div", {},
      card("Ready-made examples", "Written scenarios that need no Claude. Preview one to see exactly what it adds.",
        h("div", { class: "next-steps" }, ...(info ? info.examples : []).map((ex) => tile(ex.name, ex.prompt, () => preview(ex)))), box),
      W.preview && W.preview.kind === "scenario" ? previewCard(W.preview, box) : null);
  }

  function whatifNews(st) {
    const box = h("div", {});
    let per = 2;
    const ingest = () => act(box, ["ingest", "--run", S.world, "--yes", "--no-social", "--per-country", String(per)], {
      ok: "Today's headlines were added.",
      then: async (rec) => {
        const n = rec.lines.find((l) => l.startsWith("prompt: Live ingest")) || "";
        box.append(h("p", { class: "muted" }, n.replace("prompt: ", "")), outputBlock(rec, "What was added"));
      },
    });
    return card("Today's real news", "Fetches today's top headlines for each country from Google News (BBC Persian and IRNA for Iran), in each country's languages. Headlines reach only people who read that language. With Claude, each is labelled with a topic and tone; without it they still raise awareness.",
      st.ingest_allowed ? null : h("div", { class: "notice warn" }, h("strong", {}, "Not available for this world's date."), `The world is on ${st.date}, ${Math.abs(st.days_ahead_of_today)} days from today; real headlines would be out of place. Create a world starting today, or use Future events.`),
      h("div", { class: "row" },
        field("Headlines per country", select([["1", "1"], ["2", "2"], ["3", "3"], ["5", "5"]], per, (v) => (per = parseInt(v, 10)))),
        h("button", { class: "btn primary act", disabled: !st.ingest_allowed, onclick: ingest }, "Fetch and add headlines")),
      h("p", { class: "muted small" }, "Real posts from social networks are under ", h("button", { class: "btn link", onclick: () => go("social") }, "Social"), "."),
      box);
  }

  // ------------------------------------------------------------------ Social
  const PLATFORM = { x: "X", bluesky: "Bluesky", mastodon: "Mastodon", telegram: "Telegram" };
  const SOURCE_KIND = { official: ["tag accent", "official API"], unofficial: ["tag warn", "unofficial — may stop working"], public_preview: ["tag", "public preview"] };
  const fmtPeople = (x) => (x >= 1e9 ? `${(x / 1e9).toFixed(1)} billion` : x >= 1e6 ? `${(x / 1e6).toFixed(1)} million` : x >= 1e3 ? `${Math.round(x / 1e3)} thousand` : `${Math.round(x)}`);
  const tone = (v) => (v > 0.15 ? "good" : v < -0.15 ? "bad" : "neutral");
  const plural = (n, word) => `${Number(n).toLocaleString()} ${word}${Number(n) === 1 ? "" : "s"}`;

  function viewSocial() {
    const st = S.status;
    if (!st) return page("Social", null, loading());
    const out = page("Social", "Real posts by public figures, institutions and channels, what is trending, and the posts made in this world. Real posts reach people through the social channels in their language; reach comes from the author's audience and the post's engagement, never from a model.");
    out.append(h("div", { class: "grid two" }, socialSources(st), socialFetch(st)));
    out.append(socialTrending(st));
    out.append(socialFeed(st));
    return out;
  }

  function socialSources(st) {
    const soc = st.social || { configured: {} };
    const lf = soc.last_fetch;
    return card("Sources", "Networks that can be read for free. Facebook, TikTok, WhatsApp, Instagram and WeChat cannot, so China has no source.",
      h("ul", { class: "events-list" }, ...Object.entries(soc.configured).map(([name, c]) => {
        const [cls, label] = SOURCE_KIND[c.kind] || ["tag", c.kind];
        return h("li", {}, h("strong", {}, PLATFORM[name] || name), " ", h("span", { class: cls }, label), c.enabled ? "" : h("span", { class: "tag" }, "off"),
          h("span", { class: "muted small" }, ` ${c.accounts} accounts, ${c.queries} topic searches in this world`),
          c.kind === "unofficial" ? h("div", { class: "muted small" }, c.note) : null);
      })),
      lf ? h("p", { class: "small" }, `Last fetch: ${lf.fetch_day} (world day ${lf.world_day}). `,
        Object.entries(lf.stats || {}).map(([k, v]) => `${PLATFORM[k] || k}: ${v.posts} posts from ${v.queries} queries${v.failed ? `, ${v.failed} failed` : ""}`).join("; ")) : h("p", { class: "muted small" }, "Nothing fetched yet."),
      lf && lf.problems && lf.problems.length ? h("details", {}, h("summary", {}, `${lf.problems.length} source problems (skipped)`), h("pre", { class: "out" }, lf.problems.join("\n"))) : null);
  }

  function socialFetch(st) {
    const Soc = S.social;
    const box = h("div", {});
    const fetchPosts = () => {
      Soc.rows = null;
      return act(box, ["ingest", "--run", S.world, "--yes", "--no-news", "--per-country-social", String(Soc.per), ...(Soc.includeX ? [] : ["--no-x"])], {
        ok: "Today's posts were added.",
        then: async (rec) => box.append(outputBlock(rec, "What was fetched")),
      });
    };
    return card("Fetch today's posts", "The top posts per country become stories in the world, in their own language. Posts by ordinary people only count towards trending topics; they are not spread or shown.",
      st.ingest_allowed ? null : h("div", { class: "notice warn" }, h("strong", {}, "Not available for this world's date."), ` The world is on ${st.date}; real posts would be out of place.`),
      h("div", { class: "row" },
        field("Posts per country", select([["1", "1"], ["2", "2"], ["3", "3"], ["5", "5"]], String(Soc.per), (v) => (Soc.per = parseInt(v, 10)))),
        h("label", { class: "check" }, h("input", { type: "checkbox", checked: Soc.includeX, onchange: (e) => { Soc.includeX = e.target.checked; store.set("includeX", Soc.includeX); } }), "Include X (unofficial reader)")),
      h("p", { class: "muted small" }, "X is read through a free third-party service. X has taken scrapers to court and several such services have shut down, so it may stop working. About ten searches per ten minutes are allowed."),
      h("button", { class: "btn primary act", disabled: !st.ingest_allowed, onclick: fetchPosts }, "Fetch and add posts"),
      box);
  }

  function socialTrending(st) {
    const tr = (st.social && st.social.trending) || {};
    const isos = Object.keys(tr).sort();
    return card(h("span", {}, term("trending", "Trending"), " in the last fetch"), "Counts of fetched posts per topic. Context only: trending has no effect on the simulation.",
      isos.length ? h("div", { class: "grid two" }, ...isos.map((iso) => h("div", {}, h("strong", {}, iso),
        h("ul", { class: "events-list" }, ...tr[iso].map((r) => h("li", {}, `${r.topic}: ${plural(r.n, "post")}, tone ${tone(r.mean_valence)}`, r.top_handles.length ? h("span", { class: "muted small" }, ` (${r.top_handles.map((x) => "@" + x).join(", ")})`) : null)))))) : h("p", { class: "muted" }, "Nothing fetched yet."));
  }

  function socialFeed(st) {
    const Soc = S.social;
    const box = h("div", {});
    const load = () => {
      const argv = ["social-feed", "--run", S.world, "--json", "--limit", "80", "--kind", Soc.kind];
      if (Soc.country) argv.push("--country", Soc.country);
      Soc.loading = runJSON(argv).then((r) => { Soc.rows = r.rows; Soc.loading = null; if (box.isConnected) render(); })
        .catch((e) => { Soc.loading = null; box.replaceChildren(errorNotice(e)); });
    };
    if (Soc.rows === null && !Soc.loading) load();
    const reload = () => { Soc.rows = null; Soc.loading = null; render(); };
    const rows = Soc.rows;
    return card("Feed", "Newest first. Real posts link to the original; posts made in the world are [simulated].",
      h("div", { class: "row" },
        field("Country", select([["", "All countries"], ...st.countries.map((c) => [c.iso3, c.name])], Soc.country, (v) => { Soc.country = v; reload(); })),
        field("Show", select([["all", "Real and in-world"], ["real", "Real posts"], ["virtual", "Posts made in the world"]], Soc.kind, (v) => { Soc.kind = v; reload(); })),
        h("button", { class: "btn small", onclick: reload }, "Refresh")),
      rows === null ? loading("Loading the feed…") : rows.length ? h("div", {}, ...rows.map(feedRow)) : h("p", { class: "muted" }, "No posts yet: fetch real posts above, or post from Play."),
      box);
  }

  function feedRow(r) {
    if (r.kind === "real") {
      const who = r.handle ? (r.url ? h("a", { href: r.url, target: "_blank", rel: "noopener noreferrer" }, "@" + r.handle) : "@" + r.handle) : (r.name || "a user");
      const reach = r.reached === null || r.reached === undefined
        ? h("span", { class: "muted small" }, "not tracked (shown for context only)")
        : h("span", { class: "small" }, `reached ${plural(r.reached, "simulated person").replace("persons", "people")} (standing for about ${fmtPeople(r.reached_people)} real people), reposted ${plural(r.world_reposts || 0, "time")} in the world`);
      const metrics = r.platform === "telegram"
        ? `${(r.views || 0).toLocaleString()} views`
        : `${plural(r.likes || 0, "like")} · ${plural(r.reposts || 0, "repost")} on ${PLATFORM[r.platform] || r.platform}`;
      return h("div", { class: "feed-row" },
        h("div", { class: "row" }, h("span", { class: "tag" }, PLATFORM[r.platform] || r.platform), h("strong", {}, who), h("span", { class: "muted small" }, `${r.country} · ${r.date}${r.is_repost ? " · repost" : ""}`)),
        h("p", { style: "margin:4px 0", dir: "auto" }, r.text),
        h("div", { class: "muted small" }, `${r.topic}, ${tone(r.valence || 0)} news · ${metrics}`),
        reach);
    }
    return h("div", { class: "feed-row" },
      h("div", { class: "row" }, h("span", { class: "tag accent" }, "in the world"), h("strong", {}, r.name), h("span", { class: "muted small" }, `${r.country} · day ${r.day} · ${r.post_kind} to ${r.audience}`)),
      h("p", { style: "margin:4px 0", dir: "auto" }, `[simulated] ${r.text}`),
      h("div", { class: "small" }, `reached ${r.reached.toLocaleString()} simulated people (standing for about ${fmtPeople(r.reached_people)} real people), ${r.believed.toLocaleString()} convinced, `, term("repost", plural(r.reposts, "repost"))));
  }

  // ------------------------------------------------------------------ filters (People and Ask)
  function filterExpr(f) {
    const parts = [];
    if (f.country) parts.push(`country == ${quote(f.country)}`);
    if (f.sex) parts.push(`sex == ${quote(f.sex)}`);
    if (f.ageMin) parts.push(`age >= ${parseInt(f.ageMin, 10)}`);
    if (f.ageMax) parts.push(`age <= ${parseInt(f.ageMax, 10)}`);
    if (f.info) parts.push(`info_tier >= ${parseInt(f.info, 10)}`);
    if (f.status) parts.push(`status == ${quote(f.status)}`);
    if (f.area === "urban") parts.push("urban == True");
    if (f.area === "rural") parts.push("urban == False");
    if (f.figures === "only") parts.push("notable == True");
    if (f.figures === "exclude") parts.push("notable == False");
    return parts.join(" and ");
  }

  function filterForm(f, state, onchange, full) {
    const st = S.status;
    const set = (k) => (v) => { f[k] = v; state.advanced = null; onchange(); };
    const tiers = st.vocab.info_tiers;
    const expr = state.advanced !== null ? state.advanced : filterExpr(f);
    return h("div", {},
      h("div", { class: "form-grid" },
        field("Country", select([["", "Any"], ...st.countries.map((c) => [c.iso3, `${c.iso3} – ${c.name}`])], f.country, set("country"))),
        field("Sex", select([["", "Any"], ["F", "Female"], ["M", "Male"]], f.sex, set("sex"))),
        field("Age from", h("input", { type: "number", min: 0, max: 110, value: f.ageMin, onchange: (e) => set("ageMin")(e.target.value) })),
        field("Age to", h("input", { type: "number", min: 0, max: 110, value: f.ageMax, onchange: (e) => set("ageMax")(e.target.value) })),
        field(h("span", {}, term("information level", "Follows the news")), select([["", "Any"], ...tiers.map((t, i) => [String(i), `at least: ${t}`])], f.info, set("info"))),
        full ? field("Work", select([["", "Any"], ...st.vocab.status.map((s) => [s, s])], f.status, set("status"))) : null,
        full ? field("Area", select([["", "Any"], ["urban", "Urban"], ["rural", "Rural"]], f.area, set("area"))) : null,
        full ? field(h("span", {}, term("public figure", "Public figures")), select([["", "Include"], ["only", "Only them"], ["exclude", "Leave out"]], f.figures, set("figures"))) : null),
      h("details", {}, h("summary", {}, "Advanced filter (edit the expression directly)"),
        h("input", { style: "width:100%;margin-top:6px", value: expr, oninput: (e) => (state.advanced = e.target.value) }),
        h("p", { class: "muted small" }, "Fields: id, age, country, location, sex (F/M), urban, education (0-8), occupation, status, marital, income, info_tier (0-3), internet, visibility, honesty, risk, notable, household. Example: country in ('IRN','DEU') and income > 1000 and internet == True")));
  }

  // ------------------------------------------------------------------ People
  function viewPeople() {
    const st = S.status;
    if (!st) return page("People", null, loading());
    const P = S.people;
    const results = h("div", {});
    const detail = h("div", {});
    const search = async (keep) => {
      if (!keep) P.limit = 25;
      const expr = clean(P.advanced !== null ? P.advanced : filterExpr(P.form));
      results.replaceChildren(loading("Searching…"));
      try {
        const argv = ["agents", "--run", S.world, "--limit", String(P.limit), "--json"];
        if (expr) argv.push("--filter", expr);
        P.result = await runJSON(argv);
        P.result.expr = expr;
      } catch (e) {
        P.result = null;
        results.replaceChildren(errorNotice(e));
        return;
      }
      results.replaceChildren(peopleTable(P, detail, () => { P.limit += 50; search(true); }));
    };
    const out = page("People", "Everyone in this world is a full person with a family, a job, beliefs and friends. Pick anyone to see their profile, their circle and what they have heard, in their own words if you like.");
    out.append(card("Find people", null, filterForm(P.form, P, () => {}, true),
      h("div", { class: "row" }, h("button", { class: "btn primary act", onclick: () => search(false) }, "Search"), h("span", { class: "muted small" }, "Leave everything empty to list everyone."))));
    out.append(card(h("span", {}, term("public figure", "Public figures")), "Real office holders and opinion leaders. Open one to see their sourced positions.",
      h("div", { class: "suggest" }, ...st.public_figures.map((f) => h("button", { type: "button", onclick: () => openAgent(P, detail, f.agent_id) }, `${f.name} · ${f.office.split(";")[0]} (${f.country})`)))));
    out.append(h("div", { class: "grid two" }, h("div", {}, results), h("div", {}, detail)));
    if (P.result) results.replaceChildren(peopleTable(P, detail, () => { P.limit += 50; search(true); }));
    else search(false);
    if (P.selected !== null) openAgent(P, detail, P.selected, true);
    return out;
  }

  function peopleTable(P, detail, more) {
    const r = P.result;
    const tiers = S.status.vocab.info_tiers;
    return card(`${fmt(r.matches)} people match`, r.expr ? `Filter: ${r.expr}` : "Everyone (no filter).",
      h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Name"), h("th", {}, "Age"), h("th", {}, "Country"), h("th", {}, "Work"), h("th", {}, "News"))),
        h("tbody", {}, ...r.agents.map((a) => h("tr", { class: "clickable" + (P.selected === a.id ? " selected" : ""), "data-id": a.id, tabindex: 0, onclick: () => openAgent(P, detail, a.id), onkeydown: (e) => e.key === "Enter" && openAgent(P, detail, a.id) },
          h("td", {}, h("strong", {}, a.name), h("div", { class: "muted small" }, `#${a.id}${a.public_figure ? " · " + a.public_figure.office.split(";")[0] : ""}`)),
          h("td", {}, `${a.sex} ${a.age}`), h("td", {}, a.country), h("td", {}, a.occupation), h("td", {}, tiers[a.info_tier] || a.info_tier)))))),
      r.agents.length < r.matches ? h("div", { class: "row", style: "margin-top:10px" }, h("button", { class: "btn act", onclick: more }, "Show more")) : null);
  }

  async function openAgent(P, detail, id, cached) {
    P.selected = id;
    document.querySelectorAll("tr.selected").forEach((tr) => tr.classList.remove("selected"));
    const row = document.querySelector(`tr[data-id="${id}"]`);
    if (row) row.classList.add("selected");
    detail.replaceChildren(loading(`Opening #${id}…`));
    if (!cached || !P.profile || P.profile.id !== id) {
      const rec = await run(["agent", "--run", S.world, String(id)], { quiet: true });
      if (rec.status !== "ok") { detail.replaceChildren(errorNotice(jobError(rec))); return; }
      P.profile = { id, text: rec.lines.join("\n"), cmdline: rec.cmdline };
    }
    detail.replaceChildren(agentCard(P.profile));
    detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function agentCard(profile) {
    const st = S.status;
    const id = profile.id;
    const name = profile.text.split("\n")[0].replace(/\s+\(agent.*$/, "");
    const askBox = h("div", {});
    const dayBox = h("div", {});
    const playBox = h("div", {});
    let question = "";
    let day = Math.max(0, st.day - 1);
    let prose = false;
    const ask = async () => {
      if (!clean(question)) return;
      askBox.replaceChildren(loading("Asking…"));
      const rec = await run(["ask", "--run", S.world, "--scope", `agent:${id}`, ...(st.llm.ready ? [] : ["--no-llm"]), "--", clean(question)]);
      askBox.replaceChildren(rec.status === "ok" ? answersBlock(rec, "agent") : errorNotice(jobError(rec)));
    };
    const qta = h("textarea", { placeholder: "What have you heard lately?", oninput: (e) => (question = e.target.value) });
    const story = async () => {
      dayBox.replaceChildren(loading("Re-living that day…"));
      const rec = await run(["transcript", "--run", S.world, "--agent", String(id), "--day", String(day), ...(prose ? ["--llm"] : [])]);
      dayBox.replaceChildren(rec.status === "ok" ? outputBlock(rec) : errorNotice(jobError(rec)));
    };
    const play = () => {
      Object.assign(S.play, { brief: null, briefing: null, circle: null, circling: null, as: id });
      return act(playBox, ["possess", "--run", S.world, "--agent", String(id)], { ok: `You now play ${name}.`, then: async () => go("play") });
    };
    return h("div", {},
      card(name, "Their profile and what they know, as the simulator holds it.", h("pre", { class: "out" }, profile.text), h("div", { class: "cmd" }, profile.cmdline)),
      card("Ask them", st.llm.ready ? "They answer only from what they have actually heard." : "Without Claude you see their numbers instead of an answer.",
        qta,
        suggestions(["What have you heard this week?", "How do you feel about the economy?", "Would you buy the smart ring?"], (s) => { question = s; qta.value = s; }),
        h("button", { class: "btn primary act", onclick: ask }, "Ask"), askBox),
      card("Their day as a story", st.day ? "Re-lives one past day exactly and shows who they met and what they shared." : "Run at least one day first.",
        h("div", { class: "row" },
          field("Day", h("input", { type: "number", min: 0, max: Math.max(0, st.day - 1), value: day, oninput: (e) => (day = parseInt(e.target.value, 10) || 0) })),
          h("label", { class: "check small" }, h("input", { type: "checkbox", disabled: !st.llm.ready, onchange: (e) => (prose = e.target.checked) }), "Written as prose by Claude"),
          h("button", { class: "btn act", disabled: !st.day, onclick: story }, "Show the day")),
        dayBox),
      card("Play as them", "Take over this person: talk to their circle, post, decide, and for public figures use the powers of their office.",
        h("button", { class: "btn act", onclick: play }, `Play as ${name}`), playBox));
  }

  // `ask` prints "Name: answer" for one person and "- Name: answer" per person in a group. Numbers-only
  // answers (a snapshot) and the world summary are shown as they are.
  function answersBlock(rec, scope) {
    if (rec.lines.some((l) => l.startsWith("--- snapshot")) || scope === "world") return outputBlock(rec);
    const cards = [];
    const notes = [];
    rec.lines.forEach((line, i) => {
      const group = line.match(/^- ([^:]{2,80}): (.+)$/);
      const single = scope === "agent" && i === 0 ? line.match(/^([^:]{2,80}): (.+)$/) : null;
      const m = group || single;
      if (m) cards.push(h("div", { class: "answer" }, h("span", { class: "who" }, m[1] + ": "), m[2]));
      else if (line.trim()) notes.push(line);
    });
    return h("div", {}, notes.length ? h("p", { class: "muted small" }, notes.join(" ")) : null, ...cards, h("div", { class: "cmd" }, rec.cmdline));
  }

  // ------------------------------------------------------------------ Ask
  function viewAsk() {
    const st = S.status;
    if (!st) return page("Ask", null, loading());
    const A = S.ask;
    const box = h("div", {});
    const send = async () => {
      if (!clean(A.question)) { box.replaceChildren(h("div", { class: "notice error" }, "Write a question first.")); return; }
      let scope = "world";
      if (A.mode === "agent") {
        if (!/^\d+$/.test(String(A.agent).trim())) { box.replaceChildren(h("div", { class: "notice error" }, "Enter a person's number (find it under People).")); return; }
        scope = `agent:${String(A.agent).trim()}`;
      } else if (A.mode === "group") {
        const expr = (A.advanced !== null ? A.advanced : filterExpr(A.form)).trim();
        scope = `group:${expr || "alive"}`;
      }
      const argv = ["ask", "--run", S.world, "--scope", scope, "--sample", String(A.sample)];
      if (A.noLlm || !st.llm.ready) argv.push("--no-llm");
      argv.push("--", clean(A.question));
      box.replaceChildren(loading("Interviewing…"));
      const rec = await run(argv);
      A.result = rec;
      A.resultScope = A.mode;
      box.replaceChildren(rec.status === "ok" ? answersBlock(rec, A.mode) : errorNotice(jobError(rec)));
    };
    const sampleLabel = h("span", { class: "muted" }, ` ${A.sample}`);
    const modes = [["agent", "One person"], ["group", "A group"], ["world", "Everyone"]];
    const out = page("Ask", "Interview people about anything. Each answer comes only from what that person has actually heard, from whom, and how they feel; people who never heard of something say so. Asking changes nothing in the world.");
    out.append(card("Who do you want to ask?", null,
      h("div", { class: "row", role: "radiogroup" }, ...modes.map(([id, label]) => h("label", { class: "check" }, h("input", { type: "radio", name: "ask-mode", checked: A.mode === id, onchange: () => { A.mode = id; render(); } }), label))),
      A.mode === "agent" ? h("div", { class: "row", style: "margin-top:10px" }, field("Person number", h("input", { type: "number", min: 0, value: A.agent, oninput: (e) => (A.agent = e.target.value) }), "Find numbers under People."), h("button", { class: "btn small", onclick: () => go("people") }, "Find someone")) : null,
      A.mode === "group" ? h("div", { style: "margin-top:10px" }, filterForm(A.form, A, () => {}, false),
        h("div", { class: "row" }, field("How many people to interview", h("input", { type: "range", min: 1, max: 20, value: A.sample, oninput: (e) => { A.sample = e.target.value; sampleLabel.textContent = ` ${A.sample}`; } })), sampleLabel)) : null,
      A.mode === "world" ? h("p", { class: "muted", style: "margin-top:10px" }, "For everyone you get the overall mood and awareness numbers. To see how a prompt changed the whole world, use Results.") : null));
    const ta = h("textarea", { placeholder: "Would you buy the smart ring? Why?", oninput: (e) => (A.question = e.target.value) }, A.question);
    out.append(card("Question", st.llm.ready ? null : "Without Claude you get each person's numbers instead of a spoken answer.", ta,
      suggestions(["What have you heard this week?", "Would you buy the smart ring?", "How worried are you about prices?", "Who do you trust most for news?"], (s) => { A.question = s; ta.value = s; }),
      h("div", { class: "row" },
        h("button", { class: "btn primary act", onclick: send }, "Ask"),
        h("label", { class: "check small" }, h("input", { type: "checkbox", checked: A.noLlm || !st.llm.ready, disabled: !st.llm.ready, onchange: (e) => (A.noLlm = e.target.checked) }), "Numbers only (no Claude)")),
      box));
    if (A.result) box.replaceChildren(A.result.status === "ok" ? answersBlock(A.result, A.resultScope) : errorNotice(jobError(A.result)));
    return out;
  }

  // ------------------------------------------------------------------ Play
  function viewPlay() {
    const st = S.status;
    if (!st) return page("Play", null, loading());
    const me = st.session.agents;
    const out = page("Play", "Become someone in this world. You act once a day: message people in your circle, post publicly, decide about products, and if you play a public figure, use the powers of the office. Whether people believe you depends on how much they trust you, not on how loudly you write.");
    if (!me.length) {
      out.append(playStart(st));
    } else {
      out.append(playDesk(st, me));
    }
    out.append(sessionsCard(st));
    return out;
  }

  function playStart(st) {
    const box = h("div", {});
    let anyone = "";
    const possess = (ref, label) => {
      Object.assign(S.play, { brief: null, briefing: null, circle: null, circling: null });
      return act(box, ["possess", "--run", S.world, "--agent", String(ref)], { ok: `You now play ${label}. Act below, then finish the day.`, rerender: true });
    };
    return h("div", {},
      box,
      card(h("span", {}, "Play a ", term("public figure")), "Real office holders; what you write for them is labelled [simulated].",
        h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, ...st.public_figures.map((f) => h("tr", {},
          h("td", {}, h("strong", {}, f.name), h("div", { class: "muted small" }, f.office)),
          h("td", {}, f.country),
          h("td", { class: "num" }, h("button", { class: "btn small act", onclick: () => possess(`role:${f.key}`, f.name) }, "Play")))))))),
      card("Play anyone", "Any person in the world, by number. Find one under People and press “Play as them”.",
        h("div", { class: "row" }, h("input", { type: "number", min: 0, placeholder: "person number", oninput: (e) => (anyone = e.target.value) }),
          h("button", { class: "btn act", onclick: () => /^\d+$/.test(anyone) && possess(anyone, `#${anyone}`) }, "Play"),
          h("button", { class: "btn link", onclick: () => go("people") }, "Browse people"))),
      createPersonCard(st));
  }

  function createPersonCard(st) {
    const box = h("div", {});
    const f = { country: st.countries[0].iso3, sex: "", age: "", first: "", last: "", region: "", education: "", income: "", languages: "", visibility: "", alignment: "", friends: 5, acquaintances: 5, describe: "", role: "", office: "", possess: true };
    const set = (k) => (e) => (f[k] = e.target ? (e.target.type === "checkbox" ? e.target.checked : e.target.value) : e);
    const submit = async () => {
      const argv = ["create-agent", "--run", S.world, "--country", f.country];
      const opt = (flag, v) => { if (clean(v) !== "") argv.push(flag, clean(v)); };
      opt("--sex", f.sex); opt("--age", f.age); opt("--first-name", f.first); opt("--last-name", f.last); opt("--region", f.region);
      opt("--education", f.education); opt("--income", f.income); opt("--languages", f.languages); opt("--visibility", f.visibility);
      opt("--friends", f.friends); opt("--acquaintances", f.acquaintances);
      // A fixed list of values; one "--alignment=-0.5" token because clean() would strip a leading minus.
      if (LEANS.some(([v]) => v !== "" && v === f.alignment)) argv.push(`--alignment=${f.alignment}`);
      if (clean(f.describe)) argv.push("--describe", clean(f.describe));
      if (f.role) { argv.push("--notable-role", f.role); opt("--office", f.office); }
      if (f.possess) argv.push("--possess");
      Object.assign(S.play, { brief: null, briefing: null, circle: null, circling: null });
      await act(box, argv, { ok: f.possess ? "Person created; you now play them." : "Person created.", rerender: (rec) => h("pre", { class: "out" }, rec.lines.join("\n")) });
    };
    return card("Create a new person", "Add yourself or anyone you imagine. Unset traits are taken from a real resident of that country. New people do not change population statistics.",
      h("div", { class: "form-grid" },
        field("Country", select(st.countries.map((c) => [c.iso3, c.name]), f.country, (v) => (f.country = v))),
        field("Sex", select([["", "Any"], ["F", "Female"], ["M", "Male"]], "", (v) => (f.sex = v))),
        field("Age", h("input", { type: "number", min: 0, max: 110, oninput: set("age") })),
        field("First name", h("input", { oninput: set("first") })),
        field("Last name", h("input", { oninput: set("last") })),
        field("Area", select([["", "Any"], ["urban", "Urban"], ["rural", "Rural"]], "", (v) => (f.region = v))),
        field("Education (0-8)", h("input", { type: "number", min: 0, max: 8, oninput: set("education") }), "ISCED: 3 secondary, 6 bachelor, 8 doctorate"),
        field("Monthly income", h("input", { type: "number", min: 0, oninput: set("income") }), "In local currency"),
        field("Languages", h("input", { placeholder: "fa,en", oninput: set("languages") })),
        field(h("span", {}, term("visibility", "Visibility")), h("input", { type: "number", min: 0, max: 9, step: 0.5, oninput: set("visibility") }), "1 family … 8 head of state"),
        field(h("span", {}, term("political lean", "Political lean")), select(LEANS, "", (v) => (f.alignment = v)), "Each country names its camps on people's cards"),
        field("Friends", h("input", { type: "number", min: 0, max: 50, value: f.friends, oninput: set("friends") })),
        field("Acquaintances", h("input", { type: "number", min: 0, max: 50, value: f.acquaintances, oninput: set("acquaintances") }))),
      field("Describe them (optional, uses Claude)", h("textarea", { placeholder: "a nurse in Tabriz who follows the news closely", oninput: set("describe") })),
      h("details", { style: "margin:10px 0" }, h("summary", {}, "Make them a public figure"),
        h("div", { class: "form-grid", style: "margin-top:8px" },
          field("Role", select([["", "Not a public figure"], ...st.vocab.roles.map((r) => [r.key, r.label])], "", (v) => (f.role = v))),
          field("Office or title", h("input", { placeholder: "Mayor of Tabriz", oninput: set("office") })))),
      h("div", { class: "row" },
        h("button", { class: "btn primary act", onclick: submit }, "Create"),
        h("label", { class: "check small" }, h("input", { type: "checkbox", checked: true, onchange: set("possess") }), "Play them right away")),
      box);
  }

  function playDesk(st, me) {
    const P = S.play;
    if (!P.as || !me.some((a) => a.id === P.as)) P.as = me[0].id;
    const actor = me.find((a) => a.id === P.as);
    const asArgs = me.length > 1 ? ["--as", String(actor.id)] : [];
    const desk = h("div", {});
    const briefBox = h("div", {});
    const circleBox = h("div", {});
    const dayBox = h("div", {});

    // Briefing and circle are fetched once per day and shared by every redraw (the promise is kept in state).
    const showBrief = () => briefBox.replaceChildren(P.brief.status === "ok"
      ? h("div", {}, h("pre", { class: "out" }, P.brief.lines.join("\n")), h("div", { class: "cmd" }, P.brief.cmdline))
      : errorNotice(jobError(P.brief)));
    const loadBrief = (force) => {
      if (force) { P.brief = null; P.briefing = null; }
      if (P.brief) { showBrief(); return; }
      briefBox.replaceChildren(loading("Reading your briefing…"));
      const argv = ["brief", "--run", S.world, ...asArgs, P.replies && st.llm.ready ? "--replies" : "--no-replies"];
      P.briefing = P.briefing || run(argv, { quiet: true }).then((rec) => { P.brief = rec; return rec; });
      P.briefing.then(() => briefBox.isConnected && showBrief());
    };
    const loadCircle = () => {
      if (P.circle) {
        circleBox.replaceChildren(circleTable(P.circle, (id) => { P.target = String(id); P.tab = "message"; render(); }));
        return;
      }
      circleBox.replaceChildren(loading("Loading who listens to you…"));
      P.circling = P.circling || runJSON(["circle", "--run", S.world, ...asArgs, "--json"]).then((c) => (P.circle = c));
      P.circling.then(() => { if (circleBox.isConnected) render(); }) // redraw so the message form lists them
        .catch((e) => { P.circling = null; circleBox.replaceChildren(errorNotice(e)); });
    };
    const fresh = () => { P.brief = null; P.briefing = null; P.circle = null; P.circling = null; };
    const nextDay = () => { fresh(); return act(dayBox, ["run", "--run", S.world, "--days", "1", "--keep-control", S.run.autonomy ? "--autonomy" : "--no-autonomy"], {
      ok: "A day passed. Your briefing shows what reached you and how people took your messages.", rerender: true }); };
    const autopilot = () => { fresh(); return act(dayBox, ["run", "--run", S.world, "--days", String(Math.max(1, parseInt(P.autopilotDays, 10) || 1))], {
      ok: "Time passed with your characters on autopilot. Your next action takes them back.", rerender: true }); };
    const release = () => { fresh(); return act(dayBox, ["release", "--run", S.world], { ok: "You stopped playing; they act on their own again.", rerender: true }); };

    desk.append(card(null, null,
      h("div", { class: "row" },
        h("div", {}, h("h2", { style: "margin:0" }, actor.name), h("div", { class: "muted small" }, [actor.office || actor.occupation, actor.country, `#${actor.id}`].filter(Boolean).join(" · "))),
        h("span", { class: actor.controlled_now ? "tag accent" : "tag warn", title: GLOSSARY.autopilot }, actor.controlled_now ? "you are in control" : "on autopilot until you act"),
        h("span", { class: "spacer" }),
        me.length > 1 ? field("Acting as", select(me.map((a) => [String(a.id), a.name]), String(P.as), (v) => { P.as = parseInt(v, 10); fresh(); render(); })) : null),
      h("div", { class: "row", style: "margin-top:12px" },
        h("strong", {}, `Day ${st.day} · ${st.date}`),
        h("button", { class: "btn primary act", onclick: nextDay, title: "Your queued actions happen today; then the world moves one day" }, "Finish today → next day"),
        h("span", { class: "muted" }, "or"),
        h("button", { class: "btn act", onclick: autopilot }, "Autopilot for"),
        h("input", { type: "number", min: 1, max: 365, value: P.autopilotDays, oninput: (e) => (P.autopilotDays = e.target.value), "aria-label": "Days of autopilot" }),
        h("span", { class: "muted" }, "days"),
        h("span", { class: "spacer" }),
        h("button", { class: "btn danger act", onclick: release }, "Stop playing")),
      dayBox));

    desk.append(h("div", { class: "grid two" },
      card("Your briefing", "What reached you yesterday and how people took your messages.",
        h("label", { class: "check small" }, h("input", { type: "checkbox", checked: P.replies && st.llm.ready, disabled: !st.llm.ready, onchange: (e) => { P.replies = e.target.checked; loadBrief(true); } }), "In-character replies from people you messaged (Claude)"),
        briefBox, h("button", { class: "btn small act", style: "margin-top:8px", onclick: () => loadBrief(true) }, "Refresh")),
      actionsCard(st, actor, asArgs)));
    desk.append(card(h("span", {}, "Your ", term("circle")), "People who listen to you, most trusting first. Message one directly.", circleBox));
    desk.append(card("Play someone else as well", "You can play several people at once; pick who acts with “Acting as”.",
      h("div", { class: "row" }, h("button", { class: "btn", onclick: () => go("people") }, "Choose from People"))));

    loadBrief(false);
    loadCircle();
    return desk;
  }

  function circleTable(c, onMessage) {
    if (!c.listeners.length) return h("p", { class: "muted" }, "Nobody listens to this person personally. Public figures reach people through posts and the media.");
    return h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "Name"), h("th", {}, "Tie"), h("th", { class: "num" }, h("span", {}, term("trust", "Trust in you"))), h("th", {}, ""))),
      h("tbody", {}, ...c.listeners.slice(0, 40).map((m) => h("tr", {},
        h("td", {}, m.name, h("span", { class: "muted small" }, ` #${m.id}`)), h("td", {}, m.tie), h("td", { class: "num" }, m.trust.toFixed(2)),
        h("td", { class: "num" }, h("button", { class: "btn small", onclick: () => onMessage(m.id) }, "Message")))))));
  }

  function actionsCard(st, actor, asArgs) {
    const P = S.play;
    const box = h("div", {});
    const tabs = [["message", "Message"], ["post", "Post"], ["decide", "Decide"]];
    if (actor.policies.length) tabs.push(["office", "Use your office"]);
    if (!tabs.some(([id]) => id === P.tab)) P.tab = "message";
    const hours = Array.from({ length: 24 }, (_, i) => [String(i), `${String(i).padStart(2, "0")}:00 your time`]);
    let hour = "10";
    let topic = "";
    let stance = "";
    const claimOpts = () => {
      const a = [];
      if (topic) a.push("--topic", topic);
      if (stance !== "") a.push("--stance", stance);
      return a;
    };
    const claimFields = () => h("details", { style: "margin-top:8px" }, h("summary", {}, st.llm.ready ? "Set the topic and tone yourself (skips Claude)" : "Topic and tone (needed without Claude)"),
      h("div", { class: "form-grid", style: "margin-top:8px" },
        field("Topic", select([["", st.llm.ready ? "Let Claude decide" : "world"], ...st.topics.map((t) => [t, t])], topic, (v) => (topic = v))),
        field("Tone", select([["", st.llm.ready ? "Let Claude decide" : "neutral"], ["-1", "very bad news"], ["-0.5", "bad news"], ["0", "neutral"], ["0.5", "good news"], ["1", "very good news"]], stance, (v) => (stance = v)))));
    const queued = (rec) => box.replaceChildren(h("div", { class: "notice ok" }, h("strong", {}, "Queued for today."), "It happens when you finish the day. ", h("div", { class: "cmd" }, rec.cmdline)));
    let body;
    if (P.tab === "message") {
      let text = "";
      const targets = (P.circle ? P.circle.listeners : []).slice(0, 60);
      const send = async () => {
        if (!P.target || !clean(text)) { box.replaceChildren(h("div", { class: "notice error" }, "Choose who and write a message.")); return; }
        const rec = await act(box, ["say", "--run", S.world, ...asArgs, "--to", P.target, "--hour", hour, ...claimOpts(), "--", clean(text)]);
        if (rec) queued(rec);
      };
      body = h("div", {},
        field("To", select([["", targets.length ? "Choose someone in your circle" : "Nobody in your circle"], ...targets.map((m) => [String(m.id), `${m.name} (${m.tie}, trust ${m.trust.toFixed(2)})`])], P.target, (v) => (P.target = v))),
        field("Message", h("textarea", { placeholder: "I heard prices will rise next month", oninput: (e) => (text = e.target.value) })),
        h("div", { class: "row" }, field("When", select(hours, hour, (v) => (hour = v))), h("span", { class: "muted small" }, "If they are asleep, it reaches them when they wake.")),
        claimFields(),
        h("div", { class: "row", style: "margin-top:10px" }, h("button", { class: "btn primary act", onclick: send }, "Send")));
    } else if (P.tab === "post") {
      let text = "";
      let audience = actor.audiences.includes("national") ? "national" : actor.audiences[actor.audiences.length - 1];
      const send = async () => {
        if (!clean(text)) { box.replaceChildren(h("div", { class: "notice error" }, "Write something to post.")); return; }
        const rec = await act(box, ["post", "--run", S.world, ...asArgs, "--audience", audience, "--hour", hour, ...claimOpts(), "--", clean(text)]);
        if (rec) queued(rec);
      };
      const labels = { circle: "Your circle", community: "Your local community", national: "The whole country (media)", global: "The world (foreign media)" };
      body = h("div", {},
        h("fieldset", {}, h("legend", {}, "Who can see it"), ...st.vocab.audiences.map((a) => h("label", { class: "check", style: "margin-right:14px", title: actor.audiences.includes(a) ? "" : "Your visibility does not reach this far" },
          h("input", { type: "radio", name: "aud", disabled: !actor.audiences.includes(a), checked: a === audience, onchange: () => (audience = a) }), labels[a]))),
        field("Post", h("textarea", { placeholder: "We will announce new trade measures this week", oninput: (e) => (text = e.target.value) })),
        h("div", { class: "row" }, field("When", select(hours, hour, (v) => (hour = v)))),
        claimFields(),
        h("p", { class: "muted small" }, "How far a post spreads depends on who you are, never on how strongly it is worded."),
        actor.reach ? h("p", { class: "small" }, "Eligible audience (who could see it at most, not who will): ",
          ...actor.audiences.map((a, i) => `${i ? "; " : ""}${labels[a].toLowerCase()}: ${actor.reach[a].people.toLocaleString()} simulated people, standing for about ${fmtPeople(actor.reach[a].represents)} real people`)) : null,
        h("div", { class: "row" }, h("button", { class: "btn primary act", onclick: send }, "Post"),
          h("button", { class: "btn link", onclick: () => { S.social.kind = "virtual"; S.social.rows = null; go("social"); } }, "See how my posts did")));
    } else if (P.tab === "decide") {
      let product = st.products[0] || "";
      let choice = "adopt";
      const send = async () => {
        const rec = await act(box, ["decide", "--run", S.world, ...asArgs, "--", product, choice]);
        if (rec) queued(rec);
      };
      body = st.products.length
        ? h("div", {},
          field("Product", select(st.products.map((p) => [p, p]), product, (v) => (product = v))),
          h("div", { class: "row", style: "margin:10px 0" }, ...[["adopt", "Buy it"], ["reject", "Refuse it"], ["delay", "Wait"]].map(([v, l]) => h("label", { class: "check" }, h("input", { type: "radio", name: "choice", checked: v === choice, onchange: () => (choice = v) }), l))),
          h("button", { class: "btn primary act", onclick: send }, "Decide"))
        : h("p", { class: "muted" }, "No products exist yet. Launch one under What-if.");
    } else {
      body = officeForm(st, actor, asArgs, box, queued, hours);
    }
    return card("Act today", "Actions are queued for today and happen when you finish the day.",
      h("div", { class: "row", role: "tablist", style: "margin-bottom:10px" }, ...tabs.map(([id, label]) => h("button", { class: P.tab === id ? "btn small primary" : "btn small", onclick: () => { P.tab = id; render(); } }, label))),
      body, box);
  }

  function officeForm(st, actor, asArgs, box, queued, hours) {
    let kind = actor.policies[0].kind;
    let hour = "10";
    const f = { value: "", target: "", text: "", audience: "national", product: st.products[0] || "", name: "", price: "", quality: "0.6", countries: "" };
    const form = h("div", {});
    const draw = () => {
      const p = actor.policies.find((x) => x.kind === kind);
      const kids = [];
      if (p.note) kids.push(h("p", { class: "muted small" }, p.note));
      if (p.effect === "news") {
        kids.push(field("Statement", h("textarea", { oninput: (e) => (f.text = e.target.value) })));
        kids.push(field("Audience", select([["national", "National"], ["global", "Global"]], f.audience, (v) => (f.audience = v))));
      }
      if (p.param && p.effect !== "product") {
        const mid = ((p.param.min + p.param.max) / 2).toString();
        if (f.value === "") f.value = mid;
        kids.push(field(`${p.param.name.replace(/_/g, " ")} (${p.param.min} to ${p.param.max})`, h("input", { type: "number", min: p.param.min, max: p.param.max, step: "any", value: f.value, oninput: (e) => (f.value = e.target.value) })));
      }
      if (p.scope === "target_country") {
        const others = st.countries.filter((c) => c.iso3 !== actor.country);
        if (!f.target) f.target = others[0] ? others[0].iso3 : "";
        kids.push(field("Target country", select(others.map((c) => [c.iso3, c.name]), f.target, (v) => (f.target = v))));
      }
      if (p.effect === "product_price") kids.push(field("Product (one you launched)", select(st.products.map((x) => [x, x]), f.product, (v) => (f.product = v))));
      if (p.effect === "product") {
        kids.push(h("div", { class: "form-grid" },
          field("Product name", h("input", { oninput: (e) => (f.name = e.target.value) })),
          field("Price in dollars", h("input", { type: "number", min: 0, oninput: (e) => (f.price = e.target.value) })),
          field("Real quality (0-1)", h("input", { type: "number", min: 0, max: 1, step: 0.1, value: f.quality, oninput: (e) => (f.quality = e.target.value) })),
          field("Countries", h("input", { placeholder: actor.country, oninput: (e) => (f.countries = e.target.value) }), "Comma-separated, e.g. USA,DEU")));
      }
      kids.push(h("div", { class: "row" }, field("When", select(hours, hour, (v) => (hour = v)))));
      form.replaceChildren(...kids);
    };
    const send = async () => {
      const p = actor.policies.find((x) => x.kind === kind);
      const params = [];
      const opts = ["--hour", hour];
      if (p.effect === "news") {
        if (!clean(f.text)) { box.replaceChildren(h("div", { class: "notice error" }, "Write the statement.")); return; }
        opts.push("--text", clean(f.text));
        params.push(`audience=${f.audience}`);
      }
      if (p.param && p.effect !== "product") params.push(`${p.param.name}=${f.value}`);
      if (p.scope === "target_country") params.push(`target=${f.target}`);
      if (p.effect === "product_price") params.push(`product=${f.product}`);
      if (p.effect === "product") {
        params.push(`name=${f.name}`, `price_usd=${f.price}`, `quality=${f.quality}`);
        if (f.countries.trim()) params.push(`countries=${f.countries.replace(/\s+/g, "")}`);
      }
      const rec = await act(box, ["act", "--run", S.world, ...asArgs, ...opts, "--", kind, ...params]);
      if (rec) queued(rec);
    };
    draw();
    return h("div", {},
      field("Power", select(actor.policies.map((p) => [p.kind, p.kind.replace(/_/g, " ")]), kind, (v) => { kind = v; f.value = ""; draw(); })),
      h("div", { style: "margin-top:10px" }, form),
      h("p", { class: "muted small" }, "Effect sizes come from data/roles.yaml (placeholders, not measured values). Powers your office does not have are refused."),
      h("button", { class: "btn primary act", onclick: send }, "Use this power"));
  }

  function sessionsCard(st) {
    const box = h("div", {});
    if (!st.sessions.length) return card(h("span", {}, "Your ", term("session", "sessions")), "Once you play someone, your turns are saved here and can be replayed after an undo.", h("p", { class: "muted" }, "No sessions yet."));
    let force = false;
    let toDay = "";
    return card(h("span", {}, "Your ", term("session", "sessions")), "Every turn you play is saved. After undoing (World → Undo from here), replay re-applies your turns at their original days and reproduces the same history.",
      h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Session"), h("th", { class: "num" }, "Turns"), h("th", { class: "num" }, "Undone"), h("th", {}, ""))),
        h("tbody", {}, ...st.sessions.map((s) => h("tr", {},
          h("td", {}, `#${s.id}`, st.session.current === s.id ? h("span", { class: "tag accent", style: "margin-left:6px" }, "current") : null),
          h("td", { class: "num" }, s.turns), h("td", { class: "num" }, s.undone),
          h("td", { class: "num" },
            h("button", { class: "btn small", onclick: async () => box.replaceChildren(outputBlock(await run(["session", "show", "--run", S.world, "--session", String(s.id)]), `Session ${s.id}`)) }, "Show turns"),
            " ",
            h("button", { class: "btn small act", disabled: !s.undone, title: s.undone ? "" : "Nothing to replay", onclick: () => {
              const argv = ["session", "replay", "--run", S.world, "--session", String(s.id)];
              if (force) argv.push("--force");
              if (toDay) argv.push("--to-day", toDay);
              act(box, argv, { ok: "Replayed.", then: async (rec) => box.append(outputBlock(rec)) });
            } }, "Replay"),
            " ",
            h("button", { class: "btn small", onclick: () => { S.results.what = `session:${s.id}`; go("results"); } }, "Measure"))))))),
      h("details", {}, h("summary", {}, "Replay options"),
        h("div", { class: "row", style: "margin-top:8px" },
          field("Continue to day", h("input", { type: "number", min: 0, oninput: (e) => (toDay = e.target.value) })),
          h("label", { class: "check small" }, h("input", { type: "checkbox", onchange: (e) => (force = e.target.checked) }), "Replay even if the world differs"))),
      box);
  }

  // ------------------------------------------------------------------ Results
  function viewResults() {
    const st = S.status;
    if (!st) return page("Results", null, loading());
    const R = S.results;
    const box = h("div", {});
    const viewer = h("div", {});
    const prompts = st.prompts.filter((p) => !p.session);
    const options = [
      ...prompts.map((p) => [`prompt:${p.prompt_id}`, `Prompt #${p.prompt_id} (day ${p.day}): ${String(p.prompt).slice(0, 70)}`]),
      ...st.sessions.map((s) => [`session:${s.id}`, `Your session #${s.id} versus autopilot`]),
    ];
    if (!R.what && options.length) R.what = options[options.length - 1][0];
    const measure = async () => {
      if (!R.what) return;
      const [kind, id] = R.what.split(":");
      const argv = ["experiment", "--run", S.world, `--${kind}`, id, "--seeds", String(R.seeds)];
      if (R.horizon) argv.push("--horizon", String(R.horizon));
      const rec = await act(box, argv, { ok: "Experiment finished; writing the report…" });
      if (!rec) return;
      const rep = await act(box, ["report", "--run", S.world, `--${kind}`, id], { ok: "Report ready." });
      if (rep) openReport(viewer, kind === "session" ? `reports/session_${pad4(id)}/report.md` : `reports/${pad4(id)}/report.md`);
    };
    const out = page("Results", h("span", {}, "Find out what a prompt actually changed. An ", term("experiment"), " replays the world from just before it, with and without it, on several ", term("seed", "seeds"), ", and compares the two (the ", term("baseline"), ") so the difference is the prompt's effect, not chance."));
    out.append(card("Measure an effect", "About a minute for 4 seeds × 120 days on a full world.",
      options.length
        ? h("div", {},
          h("div", { class: "form-grid" },
            field("What to measure", select(options, R.what, (v) => (R.what = v))),
            field("Seeds", select([["2", "2 (quick)"], ["4", "4"], ["6", "6"], ["8", "8 (tighter ranges)"]], String(R.seeds), (v) => (R.seeds = parseInt(v, 10)))),
            field("Days to follow", h("input", { type: "number", min: 7, max: 730, placeholder: "the prompt's own", value: R.horizon, oninput: (e) => (R.horizon = e.target.value) }))),
          h("button", { class: "btn primary act", onclick: measure }, "Run experiment and write report"))
        : h("p", { class: "muted" }, "Add a prompt under What-if first; then you can measure it here."),
      box));
    out.append(card("Reports", "Each report compares the world with and without a prompt: awareness, adoption, mood, who changed, and the caveats.",
      st.reports.length
        ? h("div", { class: "suggest" }, ...st.reports.map((r) => h("button", { type: "button", onclick: () => openReport(viewer, r) }, r.replace("reports/", "").replace("/report.md", ""))))
        : h("p", { class: "muted" }, "No reports yet."),
      experimentsWithoutReports(st, viewer)));
    out.append(viewer);
    if (R.reportPath) openReport(viewer, R.reportPath);
    return out;
  }

  function experimentsWithoutReports(st, viewer) {
    const pending = st.experiments.filter((e) => !st.reports.includes(`reports/${e}/report.md`));
    if (!pending.length) return null;
    const box = h("div", {});
    return h("div", { style: "margin-top:10px" }, h("p", { class: "muted small" }, "Experiments without a report:"),
      h("div", { class: "row" }, ...pending.map((e) => h("button", { class: "btn small act", onclick: async () => {
        const argv = e.startsWith("session_") ? ["report", "--run", S.world, "--session", String(parseInt(e.slice(8), 10))] : ["report", "--run", S.world, "--prompt", String(parseInt(e, 10))];
        const rec = await act(box, argv, { ok: "Report ready." });
        if (rec) openReport(viewer, `reports/${e}/report.md`);
      } }, `Write report ${e}`))), box);
  }

  async function openReport(viewer, rel) {
    S.results.reportPath = rel;
    viewer.replaceChildren(loading("Opening the report…"));
    try {
      const { text } = await api(`/api/file?world=${encodeURIComponent(S.world)}&path=${encodeURIComponent(rel)}`);
      viewer.replaceChildren(card(null, null, h("div", { class: "row" }, h("span", { class: "muted small" }, rel), h("span", { class: "spacer" }), h("button", { class: "btn small", onclick: () => { S.results.reportPath = null; viewer.replaceChildren(); } }, "Close")), markdown(text)));
      viewer.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      viewer.replaceChildren(errorNotice(e));
    }
  }

  // A small Markdown renderer for the reports Jahan writes (headings, tables, lists, bold, code).
  function markdown(src) {
    const root = h("div", { class: "md" });
    const inline = (text) => {
      const frag = document.createDocumentFragment();
      const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
      let last = 0;
      let m;
      while ((m = re.exec(text))) {
        if (m.index > last) frag.append(text.slice(last, m.index));
        const t = m[0];
        if (t.startsWith("**")) frag.append(h("strong", {}, t.slice(2, -2)));
        else if (t.startsWith("`")) frag.append(h("code", {}, t.slice(1, -1)));
        else frag.append(h("em", {}, t.slice(1, -1)));
        last = m.index + t.length;
      }
      if (last < text.length) frag.append(text.slice(last));
      return frag;
    };
    const lines = src.replace(/\r/g, "").split("\n");
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (!line.trim()) continue;
      const hm = line.match(/^(#{1,4})\s+(.*)$/);
      if (hm) { root.append(h(`h${Math.min(4, hm[1].length)}`, {}, inline(hm[2]))); continue; }
      if (line.trim().startsWith("|")) {
        const rows = [];
        while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(lines[i++].trim());
        i--;
        const cells = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        const body = rows.filter((r) => !/^\|[\s:|-]+\|$/.test(r));
        const [head, ...rest] = body;
        root.append(h("div", { class: "table-wrap" }, h("table", {},
          h("thead", {}, h("tr", {}, ...cells(head).map((c) => h("th", {}, inline(c))))),
          h("tbody", {}, ...rest.map((r) => h("tr", {}, ...cells(r).map((c) => h("td", { class: /^[-+]?[\d.,]+%?/.test(c) ? "num" : null }, inline(c)))))))));
        continue;
      }
      if (/^\s*[-*]\s+/.test(line)) {
        const ul = h("ul", {});
        while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) ul.append(h("li", {}, inline(lines[i++].replace(/^\s*[-*]\s+/, ""))));
        i--;
        root.append(ul);
        continue;
      }
      const para = [line];
      while (i + 1 < lines.length && lines[i + 1].trim() && !/^(#|\||\s*[-*]\s)/.test(lines[i + 1])) para.push(lines[++i]);
      root.append(h("p", {}, inline(para.join(" "))));
    }
    return root;
  }

  // ------------------------------------------------------------------ Activity
  function viewActivity() {
    const out = page("Activity", "Every command this page ran, newest first. Copy any of them to run it yourself in a terminal (prefix with uv run).");
    if (!S.jobs.length) out.append(h("p", { class: "muted" }, "Nothing has run yet in this browser tab."));
    out.append(...S.jobs.map((j) => card(null, null,
      h("div", { class: "row" },
        h("span", { class: { ok: "tag accent", failed: "tag warn", cancelled: "tag", running: "tag" }[j.status] || "tag" }, j.status),
        h("code", { style: "word-break:break-all" }, j.cmdline),
        h("span", { class: "spacer" }),
        j.ended ? h("span", { class: "muted small" }, `${((j.ended - j.started) / 1000).toFixed(1)}s`) : null,
        h("button", { class: "btn small", onclick: (e) => { navigator.clipboard && navigator.clipboard.writeText("uv run " + j.cmdline); e.target.textContent = "Copied"; } }, "Copy")),
      h("details", {}, h("summary", {}, `${j.lines.length} line(s) of output`), h("pre", { class: "out" }, j.lines.join("\n"))))));
    return out;
  }

  // ------------------------------------------------------------------ Guide and tools
  function viewGuide() {
    const out = page("Guide & tools", "How Jahan works in a few minutes, what the words mean, and the checks and developer tools.");
    out.append(card("How it works", null,
      h("ol", {},
        h("li", {}, h("strong", {}, "Create a world. "), "Thousands of people in eight countries, built from UN, World Bank and ILO statistics: households, jobs, languages, friendships, media habits, and real public figures."),
        h("li", {}, h("strong", {}, "Move time. "), "Each day people sleep on local time, read the news they follow in languages they speak, talk to their circle, share good and bad information, and trust each other more or less as advice works out. They marry, have children, change jobs, travel and die."),
        h("li", {}, h("strong", {}, "Add what-ifs. "), "Today's real headlines, a product launch, a currency or gold move, or anything you describe. Every change is a numbered prompt you can undo."),
        h("li", {}, h("strong", {}, "Look and ask. "), "Open anyone's view of the world, or interview a person or a group. Answers come only from what they actually heard."),
        h("li", {}, h("strong", {}, "Play. "), "Become anyone, even a president, and act day by day."),
        h("li", {}, h("strong", {}, "Measure. "), "Experiments compare the world with and without a prompt across several random seeds."))),
      h("p", { class: "muted small" }, "Claude is used only to turn words into structured events, label headlines, voice people and public figures, and write prose. It never changes the world directly; the simulator computes every effect, so the same inputs always give the same history."));
    out.append(card("Words used here", null, h("dl", {}, ...Object.entries(GLOSSARY).flatMap(([k, v]) => [h("dt", {}, h("strong", {}, k)), h("dd", { style: "margin:0 0 8px" }, v)]))));
    out.append(card("Checks", null, healthButtons(), validateTool()));
    out.append(card("Draft a real person's profile", "Fetches an encyclopedia summary and news about a real public figure and drafts a roster entry with a source for every stance. The draft is written for you to verify; the simulator never loads it until you copy it into data/notables/.", draftTool()));
    out.append(card("Developer tools", null, devTools()));
    out.append(card("Appearance", null, h("div", { class: "row" }, ...[["", "Match my system"], ["light", "Light"], ["dark", "Dark"]].map(([v, l]) =>
      h("button", { class: "btn small", onclick: () => { setTheme(v); } }, l)))));
    return out;
  }

  function validateTool() {
    const box = h("div", {});
    let strict = false;
    return h("div", { style: "margin-top:12px" },
      h("div", { class: "row" },
        h("button", { class: "btn act", onclick: async () => box.replaceChildren(outputBlock(await run(["validate-data", "--countries", (S.info ? S.info.countries : ["USA"]).join(","), ...(strict ? ["--strict"] : [])]), "Data check")) }, "Check the data files"),
        h("label", { class: "check small", title: "Also fail on provisional or placeholder sources" }, h("input", { type: "checkbox", onchange: (e) => (strict = e.target.checked) }), "Strict (sources must be final)")),
      box);
  }

  function draftTool() {
    const box = h("div", {});
    const f = { country: "USA", name: "", role: "", office: "" };
    const roles = S.status ? S.status.vocab.roles : [];
    const submit = async () => {
      if (!clean(f.name) || !clean(f.role) || !clean(f.office)) { box.replaceChildren(h("div", { class: "notice error" }, "Fill in the name, role and office.")); return; }
      const rec = await run(["notables-draft", "--country", f.country, "--name", clean(f.name), "--role", clean(f.role), "--office", clean(f.office)]);
      box.replaceChildren(rec.status === "ok" ? outputBlock(rec, "Draft written") : errorNotice(jobError(rec)));
    };
    return h("div", {},
      h("div", { class: "form-grid" },
        field("Country", select((S.info ? S.info.countries : ["USA"]).map((c) => [c, c]), f.country, (v) => (f.country = v))),
        field("Name", h("input", { placeholder: "Full public name", oninput: (e) => (f.name = e.target.value) })),
        field("Role", roles.length ? select([["", "Choose"], ...roles.map((r) => [r.key, r.label])], "", (v) => (f.role = v)) : h("input", { placeholder: "business_leader", oninput: (e) => (f.role = e.target.value) })),
        field("Office", h("input", { placeholder: "Chief executive, Example Corp", oninput: (e) => (f.office = e.target.value) }))),
      h("button", { class: "btn act", disabled: S.info && !(S.info.llm.api_key_set && S.info.llm.sdk_installed), onclick: submit }, "Draft (uses Claude)"),
      box);
  }

  function devTools() {
    const box = h("div", {});
    let agents = 10000;
    let days = 30;
    return h("div", {},
      h("div", { class: "row" },
        h("button", { class: "btn act", disabled: !S.world, title: S.world ? "" : "Open a world first", onclick: async () => box.replaceChildren(outputBlock(await run(["events-hash", "--run", S.world]), "Fingerprint of this world's event log")) }, "Event-log fingerprint"),
        h("span", { class: "muted small" }, "Identical histories have identical fingerprints.")),
      h("div", { class: "row", style: "margin-top:10px" },
        field("Speed test: people", h("input", { type: "number", min: 100, max: 50000, value: agents, oninput: (e) => (agents = e.target.value) })),
        field("days", h("input", { type: "number", min: 1, max: 365, value: days, oninput: (e) => (days = e.target.value) })),
        h("button", { class: "btn act", onclick: async () => box.replaceChildren(outputBlock(await run(["bench", "--agents", String(agents), "--days", String(days)]), "Speed test")) }, "Run speed test")),
      box);
  }

  function setTheme(v) {
    if (v) document.documentElement.setAttribute("data-theme", v);
    else document.documentElement.removeAttribute("data-theme");
    store.set("theme", v);
  }

  // ------------------------------------------------------------------ Watch
  const SVG_NS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs, ...kids) {
    const el = document.createElementNS(SVG_NS, tag); // h() makes HTML elements, which cannot draw
    for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) el.setAttribute(k, String(v));
    kids.forEach((c) => c && el.append(c));
    return el;
  }

  function sparkline(values, base, down) {
    const W = 140, H = 32, P = 3;
    if (values.length < 2) return h("span", { class: "muted small" }, "after the next day");
    const all = base === null || base === undefined ? values : [...values, base];
    let lo = Math.min(...all), hi = Math.max(...all);
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    const x = (i) => P + (i * (W - 2 * P)) / (values.length - 1);
    const y = (v) => H - P - ((v - lo) * (H - 2 * P)) / (hi - lo);
    const pts = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    return svg("svg", { class: down ? "spark down" : "spark", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${values.length} days` },
      base === null || base === undefined ? null : svg("line", { x1: P, x2: W - P, y1: y(base).toFixed(1), y2: y(base).toFixed(1) }),
      svg("polyline", { points: pts }));
  }

  function watchValue(v, unit) {
    if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
    if (unit === "probability") return `${(100 * v).toFixed(1)}%`;
    if (unit === "mood") return (v >= 0 ? "+" : "") + v.toFixed(3);
    if (unit === "rate") return Number(v.toPrecision(5)).toLocaleString();
    return v.toFixed(2);
  }

  function watchChange(s) {
    let text, sign;
    if (s.unit === "probability") { sign = s.since_watched_abs; text = `${sign >= 0 ? "+" : ""}${(100 * sign).toFixed(1)} points`; }
    else if (s.unit === "mood") { sign = s.since_watched_abs; text = (sign >= 0 ? "+" : "") + sign.toFixed(3); }
    else { sign = s.since_watched_pct || 0; text = `${sign >= 0 ? "+" : ""}${sign.toFixed(2)}%`; }
    return h("span", { class: Math.abs(sign) < 1e-9 ? "" : sign > 0 ? "up" : "down" }, text);
  }

  const pctCell = (p) => (p === null || p === undefined ? h("span", { class: "muted" }, "n/a") : h("span", { class: Math.abs(p) < 1e-9 ? "" : p > 0 ? "up" : "down" }, `${p >= 0 ? "+" : ""}${p.toFixed(2)}%`));

  function viewWatch() {
    const st = S.status;
    if (!st) return page("Watch", null, loading());
    const W = S.watch;
    const box = h("div", {});
    const key = `${S.world}|${st.day}|${st.prompts.length}|${(st.watch || []).join(",")}|${W.days}`;
    if (W.key !== key && !W.loading) {
      // after time moved forward, also ask why nothing moved since the day the page last showed
      if (W.seenWorld !== S.world || W.seenDay === undefined || st.day < W.seenDay) W.fromDay = null;
      else if (st.day > W.seenDay) W.fromDay = W.seenDay;
      const argv = ["watch", "show", "--run", S.world, "--json", "--days", String(W.days)];
      if (W.fromDay !== null && W.fromDay !== undefined) argv.push("--since-day", String(W.fromDay));
      W.loading = runJSON(argv)
        .then((d) => { W.data = d; W.key = key; W.loading = null; W.seenWorld = S.world; W.seenDay = d.day; if (S.view === "watch") render(); })
        .catch((e) => { W.loading = null; W.key = key; box.replaceChildren(errorNotice(e)); });
    }
    const d = W.data && W.data.day !== undefined && W.key && W.key.startsWith(S.world + "|") ? W.data : null;
    const toggle = (id, on) => act(box, ["watch", on ? "add" : "remove", "--run", S.world, id], { ok: on ? "Watching it from today." : "Stopped watching it.", rerender: true });
    const out = page("Watch", h("span", {}, "Pick what to follow, move time forward, and see how it moved, and why: each market move is split into its parts (news reactions, saving and cashing out, announcements and what-ifs). Everything here is simulated, not a forecast."));
    for (const n of watchNotices(st, box)) out.append(n);
    out.append(timeCard());
    const rows = d ? d.series : [];
    const ex = d && d.explanation;
    if (ex && ex.quiet) out.append(h("div", { class: "notice warn" },
      h("strong", {}, `Nothing you watch moved between day ${ex.from_day} and day ${ex.to_day} (markets by less than 0.05%). Why:`),
      h("ul", { class: "events-list small" }, ...ex.sentences.map((s) => h("li", {}, s)))));
    out.append(card("Watching", d ? `Day ${d.day} (${d.date}). The line shows the last ${W.days} days; the dashed line is the level when you started watching.` : null,
      !d ? loading(W.loading ? "Reading the history…" : "") : rows.length
        ? h("div", { class: "table-wrap" }, h("table", {},
          h("thead", {}, h("tr", {}, h("th", {}, "Series"), h("th", { class: "num" }, "Now"), h("th", { class: "num" }, "Since watched"),
            h("th", {}, "Why it moved"), h("th", {}, `Last ${W.days} days`), h("th", {}, ""))),
          h("tbody", {}, ...rows.map((s) => h("tr", {},
            h("td", {}, h("strong", {}, s.label), s.illustrative ? h("span", { class: "tag", title: s.note || "" }, "illustrative") : null,
              h("div", { class: "muted small" }, `since day ${s.since_day} (${s.since_date})${s.rebased ? ", restarted after an undo" : ""}`), s.note ? h("div", { class: "muted small" }, s.note) : null),
            h("td", { class: "num" }, watchValue(s.now, s.unit)),
            h("td", { class: "num" }, watchChange(s)),
            h("td", {}, s.components && s.parts.length ? h("div", { class: "chips parts" }, ...s.parts.map((p) => h("span", { class: "chip small" }, `${p.label} `,
                s.unit === "probability" ? ptsCell(s.components[`${p.name}_pts`]) : pctCell(s.components[`${p.name}_pct`]))))
              : h("span", { class: "muted" }, "—")),
            h("td", {}, sparkline(s.history.map((r) => r.value), s.since_value, (s.since_watched_abs || 0) < 0)),
            h("td", {}, h("button", { class: "btn small act", onclick: () => toggle(s.id, false) }, "Stop")))))))
        : h("p", { class: "muted" }, "Nothing watched yet: tick something below, then move time forward."),
      box));
    const cat = d ? d.catalog : [];
    const groups = [["market", "Markets"], ["mood", "Moods"], ["odds", "Event odds"]];
    const qs = rows.filter((s) => s.question);
    if (qs.length) out.append(card(h("span", {}, "Event odds: simulated and real"), "The simulated odds are the average belief of simulated traders. The real price is shown for comparison only and never moves the simulation.",
      ...qs.map(questionRow)));
    out.append(oddsQuestionsCard(st, box));
    out.append(card("Add to the watch list", "Markets move on announcements, scripted what-ifs and, for US stocks, people's trading. Use What-if to add news or a market move and watch the effect.",
      !d ? loading() : h("div", {}, ...groups.filter(([k]) => cat.some((c) => c.kind === k)).map(([k, label]) => h("div", {},
        h("h4", {}, label),
        h("div", { class: "watch-cat" }, ...cat.filter((c) => c.kind === k).map((c) => h("label", { class: "check", title: c.note || "" },
          h("input", { type: "checkbox", checked: c.watched, onchange: (e) => toggle(c.id, e.target.checked) }), c.label))))))));
    if (d) out.append(h("p", { class: "muted small" }, d.disclaimer));
    return out;
  }

  function watchNotices(st, box) {
    const out = [];
    if (st.markets && !st.markets.everyday) out.push(h("div", { class: "notice warn" },
      h("strong", {}, "This world was made before everyday activity existed"),
      st.markets.people_trading
        ? "Its people trade US stocks only on the day news reaches them; moods move only with news, and there are no other countries' markets. "
        : "Its US stocks move only on announcements and what-ifs, and nobody in it buys or sells. ",
      "Turning everyday activity on changes that from today: people act on news as they look at their accounts, each country gets its own market, job losses and cash-outs matter, and moods move with life and conversation. It is recorded as a prompt, so Undo (Results) takes the world back to just before it; days already simulated stay as they were. ",
      h("button", { class: "btn small act", onclick: () => act(box, ["economy", "on", "--run", S.world, "--yes"], { ok: "Everyday activity is on from today.", rerender: true }) }, "Turn on everyday activity")));
    if (!st.llm.ready) out.push(h("div", { class: "notice" },
      h("strong", {}, "Claude is not set up, so some things are off"),
      "Public figures do not act on their own, new headlines are not rated against your questions, and what-ifs in plain words do not work. Start the browser with ",
      h("code", {}, "jahan ui --key-file <file>"), " or see ", h("code", {}, "jahan doctor"), "."));
    return out;
  }

  const prob = (p) => (p === null || p === undefined || Number.isNaN(p) ? "n/a" : `${(100 * p).toFixed(1)}%`);

  function ptsCell(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return h("span", { class: "muted" }, "n/a");
    const pts = 100 * v;
    return h("span", { class: Math.abs(pts) < 0.05 ? "" : pts > 0 ? "up" : "down" }, `${pts >= 0 ? "+" : ""}${pts.toFixed(1)} pts`);
  }

  function kindPicker(q, box) {
    const set = (k) => act(box, ["watch", "set", "--run", S.world, `odds:${q.id}`, "--kind", k], { ok: "Time effect changed (a prompt you can undo).", rerender: true });
    return h("div", { class: "small" }, "Time effect: ",
      select([["by_date", "happens by the date (odds fall on quiet days)"], ["holds_through", "holds through the date (odds rise on quiet days)"], ["other", "none"]], q.kind, set),
      h("span", { class: "muted" }, ` (${q.kind_by === "user" ? "your choice" : q.kind_by === "seed" ? "set in the saved question" : q.kind_by === "rule" ? "read from the wording" : "added before time effects existed"}${q.kind_why ? ": " + q.kind_why : ""}; heuristic: a constant chance per day inside each trader's belief)`),
      q.days_left !== null && q.days_left !== undefined ? h("span", { class: "muted" }, ` · ${q.days_left} day(s) left`) : null);
  }

  function questionRow(s) {
    const q = s.question;
    const qb = h("div", {});
    const sim = q.simulated;
    const real = q.real;
    const pastReal = real && S.status && real.fetch_date < S.status.date;
    return h("div", { class: "qrow" },
      h("div", {}, h("strong", {}, q.question), " ", q.source && q.source.url ? h("a", { href: q.source.url, target: "_blank", rel: "noopener noreferrer", class: "small" }, "market page") : null),
      h("div", { class: "small" },
        q.status === "open" ? `Simulated ${prob(sim ? sim.odds : null)}` : `Closed on day ${q.closed_day} at ${prob(q.final_odds)}`,
        sim ? ` (${sim.n_traders.toLocaleString()} simulated traders, standing for about ${fmtPeople(sim.represents)} people)` : "",
        real ? ` · real market ${prob(real.price)} on ${real.fetch_date}` : " · no real price",
        real && real.resolved ? ` · resolved ${real.resolved} on the market` : "",
        q.resolves_by ? ` · resolves ${q.resolves_by}` : ""),
      pastReal ? h("div", { class: "muted small" }, `Real prices end at ${real.fetch_date}; the simulation runs on without them.`) : null,
      q.status === "open" && q.resolves_by ? kindPicker(q, qb) : null,
      qb,
      q.movers.length ? h("ul", { class: "events-list small" }, ...q.movers.map((m) => h("li", {},
        h("span", { class: m.moved > 0 ? "up" : "down" }, `${m.moved > 0 ? "+" : ""}${(100 * m.moved).toFixed(1)} points `),
        `"${m.label}" (day ${m.day}, ${m.by === "scripted" ? "your what-if" : m.by === "llm" ? "rated by Claude" : m.by}${m.why && m.why !== "scripted" ? ": " + m.why : ""})`)))
        : h("div", { class: "muted small" }, "No story has moved it yet."));
  }

  const slugify = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "question";

  function oddsQuestionsCard(st, box) {
    const W = S.watch;
    if (W.seedsFor !== S.world) {
      W.seedsFor = S.world;
      W.seeds = null;
      runJSON(["watch", "catalog", "--run", S.world, "--json"]).then((r) => { W.seeds = r.seeds || []; if (S.view === "watch") render(); })
        .catch(() => { W.seeds = []; });
    }
    const watched = new Set(st.questions || []);
    const add = (argv, ok) => act(box, ["watch", "add", "--run", S.world, ...argv], { ok, rerender: true });
    const seedList = W.seeds === null ? loading() : W.seeds.length ? h("div", {},
      h("p", { class: "muted small" }, `Read from Polymarket on ${W.seeds[0].fetch_date}. Watching one starts its simulated traders at that price.`),
      h("label", { class: "check small" }, h("input", { type: "checkbox", checked: W.force, onchange: (e) => (W.force = e.target.checked) }), "Start from these prices even though the world's date is different"),
      h("ul", { class: "events-list" }, ...W.seeds.map((q) => h("li", {},
        h("span", { class: "num" }, prob(q.yes_price)), " ", q.question, q.resolves_by ? h("span", { class: "muted small" }, ` (resolves ${q.resolves_by})`) : null, " ",
        watched.has(q.id) ? h("span", { class: "tag" }, "watching")
          : h("button", { class: "btn small act", onclick: () => add([q.series, ...(W.force ? ["--force"] : [])], "Watching it: simulated traders start near the market price.") }, "Watch")))))
      : h("p", { class: "muted" }, "No offline questions in data/global/questions.yaml.");
    const C = W.custom;
    const custom = h("div", { class: "form-grid" },
      field("Your own yes/no question", h("input", { value: C.question, placeholder: "Will the US and Iran sign a deal by December?", oninput: (e) => (C.question = e.target.value) })),
      field("Starting probability", select([["0.1", "10%"], ["0.25", "25%"], ["0.5", "50%"], ["0.75", "75%"], ["0.9", "90%"]], C.prior, (v) => (C.prior = v))),
      field("Resolves by (optional)", h("input", { type: "date", value: C.resolves, onchange: (e) => (C.resolves = e.target.value) })),
      field("Traders in (optional)", h("input", { value: C.countries, placeholder: "USA,IRN", oninput: (e) => (C.countries = e.target.value) }), "Default: every country"));
    const addCustom = () => {
      const text = clean(C.question);
      if (!text) { box.replaceChildren(h("div", { class: "notice error" }, "Write the question first.")); return; }
      const argv = [`odds:custom:${slugify(text)}`, "--question", text, "--prior", C.prior];
      if (C.resolves) argv.push("--resolves-by", C.resolves);
      if (C.countries.trim()) argv.push("--countries", C.countries.replace(/\s+/g, ""));
      add(argv, "Watching your question.");
    };
    const liveBox = h("div", {});
    const findLive = async () => {
      const argv = ["watch", "catalog", "--run", S.world, "--polymarket", "--json"];
      if (W.liveQuery.trim()) argv.push("--query", clean(W.liveQuery));
      else if (W.liveTag) argv.push("--tag", W.liveTag);
      liveBox.replaceChildren(loading("Reading Polymarket…"));
      try {
        W.live = (await runJSON(argv)).polymarket || [];
        render();
      } catch (e) {
        liveBox.replaceChildren(errorNotice(e));
      }
    };
    const liveList = W.live === null ? null : W.live.length ? h("ul", { class: "events-list" }, ...W.live.map((q) => h("li", {},
      h("span", { class: "num" }, prob(q.yes_price)), " ", q.question,
      h("span", { class: "muted small" }, ` (${q.event_title ? q.event_title + "; " : ""}${q.resolves_by ? "resolves " + q.resolves_by : "no date"})`), " ",
      h("a", { href: q.url, target: "_blank", rel: "noopener noreferrer", class: "small" }, "page"), " ",
      watched.has(q.slug) ? h("span", { class: "tag" }, "watching")
        : h("button", { class: "btn small act", onclick: () => add([q.series, ...(W.force ? ["--force"] : [])], "Watching it: simulated traders start at today's market price.") }, "Watch"))))
      : h("p", { class: "muted" }, "Nothing open matched.");
    const live = h("div", {},
      h("div", { class: "row" },
        h("input", { value: W.liveQuery, placeholder: "Search, e.g. Iran", oninput: (e) => (W.liveQuery = e.target.value), "aria-label": "Search Polymarket" }),
        h("span", { class: "muted small" }, "or the busiest in"),
        select([["", "all topics"], ["geopolitics", "geopolitics"], ["politics", "politics"], ["economy", "economy"]], W.liveTag, (v) => { W.liveTag = v; W.liveQuery = ""; }),
        h("button", { class: "btn act", onclick: findLive }, "Find on Polymarket"),
        st.questions && st.questions.length ? h("button", { class: "btn act", title: "Read today's prices of the watched Polymarket questions (for comparison only)", onclick: () => act(box, ["watch", "fetch", "--run", S.world], { ok: "Real prices updated.", rerender: true }) }, "Refresh real prices") : null),
      h("p", { class: "muted small" }, "Reads Polymarket's public data (no account; responses are recorded in the world). Prices are for comparison and as a starting point; check Polymarket's terms of use before reusing them."),
      liveList, liveBox);
    return card(h("span", {}, "Event ", term("question", "questions")), "Pick a real prediction-market question, or ask your own. Stories in What-if can bear on it (the preview says how); with Claude set up, new headlines are rated against it as time moves.",
      h("h4", {}, "Live from Polymarket"), live,
      h("h4", {}, "Saved questions (work offline)"), seedList, h("h4", {}, "Your own question"), custom, h("button", { class: "btn act", onclick: addCustom }, "Watch my question"));
  }

  const RENDER = { start: viewStart, world: viewWorld, whatif: viewWhatif, social: viewSocial, people: viewPeople, ask: viewAsk, play: viewPlay, results: viewResults, watch: viewWatch, activity: viewActivity, guide: viewGuide };

  // ------------------------------------------------------------------ boot
  async function boot() {
    setTheme(store.get("theme") || "");
    S.token = readToken();
    $("#console-toggle").addEventListener("click", () => {
      const open = !$("#console").classList.contains("open");
      setConsole(open);
      store.set("consoleOpen", open);
    });
    $("#menu-toggle").addEventListener("click", () => {
      const sb = $("#sidebar");
      sb.classList.toggle("open");
      $("#menu-toggle").setAttribute("aria-expanded", String(sb.classList.contains("open")));
    });
    $("#world-select").addEventListener("change", (e) => openWorld(e.target.value));
    // Pasting a new link (after restarting `jahan ui`) into this tab only changes the #fragment, which does not
    // reload the page: pick up the new token and reload.
    window.addEventListener("hashchange", () => {
      const t = readToken();
      if (t && t !== S.token) location.reload();
    });
    if (!S.token) {
      $("#main").replaceChildren(page("Open Jahan from its link", "This page needs the private link that `jahan ui` printed in its terminal (it ends with #t=…). Start it with: uv run jahan ui"));
      return;
    }
    try {
      await refreshInfo();
    } catch (e) {
      $("#main").replaceChildren(page("Jahan is not reachable", e.status === 403 ? "This link belongs to an earlier session. Use the link printed the last time you started `jahan ui`." : `The local server did not answer (${e.message}). Is \`jahan ui\` still running?`));
      return;
    }
    const saved = store.get("world");
    const view = store.get("view") || "start";
    renderChrome();
    if (saved && S.info.worlds.some((w) => w.path === saved)) await openWorld(saved, view === "start" ? "world" : view);
    else go("start");
  }

  boot();
})();
