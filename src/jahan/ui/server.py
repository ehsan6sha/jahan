"""`jahan ui`: a local web page whose buttons run `jahan` commands.

The page is only a remote control.  Every click becomes an ordinary `jahan ...` command, run as a separate
process in the project folder exactly as if typed in a terminal; its output streams back to the page,
which also shows the command line so people can learn the CLI.  Nothing is computed in the browser or in
this server, and the server never imports the simulator.

Safety (the commands can spend the user's API key):
* the server listens on 127.0.0.1 only and checks the Host header (no DNS rebinding);
* a random token, handed to the page in the URL fragment (never sent to any server or kept in history),
  must accompany every API call, and state-changing calls must come from the page's own origin;
* only known commands run, without a shell, and path options must stay inside the runs, examples,
  configs and drafts folders;
* one command at a time per world, because commands write the same files; a job can be cancelled, which
  kills its whole process tree (experiments start worker processes).
"""

from __future__ import annotations

import html
import http.server
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

WORLD_COMMANDS = {
    "gen", "run", "inject", "feed", "ingest", "ask", "transcript", "experiment", "report", "reset", "agents",
    "agent", "status", "create-agent", "possess", "release", "say", "post", "decide", "act", "brief", "circle",
    "events-hash", "session", "social-feed",
}  # fmt: skip
GLOBAL_COMMANDS = {"doctor", "validate-data", "bench", "notables-draft"}
SESSION_SUBCOMMANDS = {"list", "show", "replay"}
PATH_OPTIONS = {"--run": "runs", "-r": "runs", "--scenario": "scenarios", "--config": "configs", "--out": "drafts"}
SHORT_OPTIONS = {"-r", "-y", "-f"}  # the only single-dash options the CLI has; clusters like "-yr" are refused
NUMBER = re.compile(r"-\d+(\.\d+)?")  # negative values such as --stance -0.5
MAX_ARGS = 80
MAX_ARG_CHARS = 20_000
MAX_BODY = 1_000_000
KEEP_JOBS = 300
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
}
READABLE_SUFFIXES = {".md", ".yaml", ".yml", ".txt", ".json"}


class ArgvError(ValueError):
    """A command the page may not run."""


class Busy(RuntimeError):
    def __init__(self, job_id: int) -> None:
        super().__init__(f"this world is busy with job {job_id}")
        self.job_id = job_id


@dataclass
class Settings:
    project: Path  # where commands run (the repository: data/, configs/, examples/)
    runs: Path  # folder holding worlds
    token: str
    port: int = 0
    env: dict[str, str] = field(default_factory=dict)

    def roots(self, kind: str) -> list[Path]:
        return {
            "runs": [self.runs],
            "scenarios": [self.project / "examples" / "scenarios", self.runs],
            "configs": [self.project / "configs"],
            "drafts": [self.project / "data" / "notables" / "drafts"],
        }[kind]


def _inside(value: str, roots: list[Path], project: Path) -> Path:
    p = Path(value)
    p = (p if p.is_absolute() else project / p).resolve()
    for r in roots:
        r = r.resolve()
        if p != r and p.is_relative_to(r):
            return p
    raise ArgvError(f"{value!r} is outside the folders the interface may use")


def check_argv(argv: Any, s: Settings) -> tuple[list[str], str | None]:
    """Validate a command from the page.  Returns the argv to run (paths made relative) and the world it uses."""
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise ArgvError("a command is a non-empty list of strings")
    if len(argv) > MAX_ARGS or any(len(a) > MAX_ARG_CHARS for a in argv):
        raise ArgvError("command too long")
    cmd = argv[0]
    if cmd not in WORLD_COMMANDS | GLOBAL_COMMANDS:
        raise ArgvError(f"unknown command {cmd!r}")
    if cmd == "session" and (len(argv) < 2 or argv[1] not in SESSION_SUBCOMMANDS):
        raise ArgvError("session needs list, show or replay")
    out = list(argv)
    world: str | None = None
    seen: set[str] = set()
    i = 1
    while i < len(out):
        a = out[i]
        if a == "--":
            # The CLI's end-of-options marker, but it can also be consumed as an option's value ("--prompt --"),
            # so the tokens after it are checked like any others; the page strips leading dashes from text.
            i += 1
            continue
        if a.startswith("-") and not a.startswith("--") and a not in SHORT_OPTIONS and not NUMBER.fullmatch(a):
            # The CLI would read "-rPATH" or "-yr PATH" as a path option this scanner cannot see.
            raise ArgvError(f"write {a!r} as separate long options")
        if a.startswith("-") and "=" in a and a.split("=", 1)[0] in PATH_OPTIONS:
            raise ArgvError("give path options as two arguments")
        if a in PATH_OPTIONS:
            kind = PATH_OPTIONS[a]
            if kind in seen:
                raise ArgvError(f"{a} given twice")
            seen.add(kind)
            if i + 1 >= len(out):
                raise ArgvError(f"{a} needs a value")
            p = _inside(out[i + 1], s.roots(PATH_OPTIONS[a]), s.project)
            out[i + 1] = _display_path(p, s.project)
            if PATH_OPTIONS[a] == "runs":
                world = str(p)
            i += 2
            continue
        i += 1
    if cmd in WORLD_COMMANDS and world is None:
        raise ArgvError(f"{cmd} needs --run <world>")
    return out, world


def _display_path(p: Path, project: Path) -> str:
    try:
        return str(p.relative_to(project.resolve())).replace("\\", "/")
    except ValueError:
        return str(p)


def cmdline(argv: list[str]) -> str:
    """The command as someone would type it (for display and copying)."""

    def q(a: str) -> str:
        return a if a and all(ch.isalnum() or ch in "-_./:=,@" for ch in a) else '"' + a.replace('"', '\\"') + '"'

    return "jahan " + " ".join(q(a) for a in argv)


@dataclass
class Job:
    id: int
    argv: list[str]
    world: str | None
    lines: list[str] = field(default_factory=list)
    status: str = "running"  # running | ok | failed | cancelled (only once the process has stopped)
    cancel_requested: bool = False
    exit_code: int | None = None
    started: float = field(default_factory=time.time)
    ended: float | None = None
    proc: subprocess.Popen[str] | None = None

    def view(self, since: int = 0) -> dict[str, Any]:
        new = self.lines[since:]  # one slice, so "next" never skips a line appended meanwhile
        return {
            "id": self.id,
            "argv": self.argv,
            "cmdline": cmdline(self.argv),
            "world": self.world,
            "status": self.status,
            "cancelling": self.cancel_requested and self.status == "running",
            "exit_code": self.exit_code,
            "started": self.started,
            "ended": self.ended,
            "lines": new,
            "next": since + len(new),
        }


def child_env(s: Settings) -> dict[str, str]:
    env = dict(os.environ)
    for var in ("FORCE_COLOR", "PY_COLORS", "GITHUB_ACTIONS"):
        env.pop(var, None)
    env.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "NO_COLOR": "1", "COLUMNS": "160"})
    env.update(s.env)
    return env


def kill_tree(proc: subprocess.Popen[str]) -> None:
    """Stop a command and everything it started (experiments run worker processes)."""
    if sys.platform == "win32":
        if proc.poll() is None:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False)
        return

    # The command leads its own process group (start_new_session), whose id is its pid; signalling the group
    # also reaches workers that outlive it.
    def signal_group(sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    signal_group(signal.SIGTERM)
    force = threading.Timer(3.0, signal_group, args=(signal.SIGKILL,))
    force.daemon = True  # never keeps the server from exiting
    force.start()


class Jobs:
    def __init__(self, s: Settings) -> None:
        self.s = s
        self.lock = threading.Lock()
        self.jobs: dict[int, Job] = {}
        self.busy: dict[str, int] = {}
        self.next_id = 1

    def start(self, argv: Any) -> Job:
        args, world = check_argv(argv, self.s)
        if world is not None:
            world = os.path.normcase(world)  # one lock per folder, whatever the letter case on Windows
        with self.lock:
            if world is not None and world in self.busy:
                raise Busy(self.busy[world])
            job = Job(self.next_id, args, world)
            self.next_id += 1
            self.jobs[job.id] = job
            if world is not None:
                self.busy[world] = job.id
            for old in sorted(self.jobs)[:-KEEP_JOBS]:
                if self.jobs[old].status != "running":
                    del self.jobs[old]
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":  # its own group, so Cancel can stop it and its workers together
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            job.proc = subprocess.Popen(
                [sys.executable, "-m", "jahan.cli", *args],
                cwd=self.s.project,
                stdin=subprocess.DEVNULL,  # a stray confirmation fails at once instead of waiting forever
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=child_env(self.s),
                **kwargs,
            )
        except OSError as e:
            job.lines.append(f"could not start the command: {e}")
            self._finish(job, 1)
            return job
        with self.lock:
            cancelled_meanwhile = job.cancel_requested
        if cancelled_meanwhile:  # Cancel arrived while the process was being created
            kill_tree(job.proc)
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()
        return job

    def _pump(self, job: Job) -> None:
        assert job.proc is not None and job.proc.stdout is not None
        for line in job.proc.stdout:  # drained continuously, so a chatty command never blocks on a full pipe
            job.lines.append(line.rstrip("\r\n"))
        self._finish(job, job.proc.wait())

    def _finish(self, job: Job, code: int) -> None:
        with self.lock:
            job.exit_code = code
            job.ended = time.time()
            # "cancelled" only now that the process has stopped and the world is free again
            job.status = "cancelled" if job.cancel_requested else ("ok" if code == 0 else "failed")
            if job.world is not None and self.busy.get(job.world) == job.id:
                del self.busy[job.world]

    def cancel(self, job_id: int) -> Job:
        with self.lock:
            job = self.jobs[job_id]
            if job.status != "running" or job.cancel_requested:
                return job
            job.cancel_requested = True
            job.lines.append("(cancelled from the page)")
            proc = job.proc
        if proc is not None:  # otherwise start() kills it as soon as it exists
            kill_tree(proc)
        return job

    def snapshot(self) -> list[Job]:
        with self.lock:
            return sorted(self.jobs.values(), key=lambda j: -j.id)

    def shutdown(self) -> None:
        for job in self.snapshot():
            if job.status == "running":
                self.cancel(job.id)


def list_worlds(s: Settings) -> list[dict[str, Any]]:
    """Worlds in the runs folder, newest first, read from small metadata files (no simulation code)."""
    out: list[dict[str, Any]] = []
    if not s.runs.exists():
        return out
    for d in s.runs.iterdir():
        head = d / "snapshots" / "head.json"
        if not head.exists():
            continue
        try:
            meta = json.loads(head.read_text(encoding="utf-8"))
            prompts = (
                json.loads((d / "prompts.json").read_text(encoding="utf-8")) if (d / "prompts.json").exists() else []
            )
        except (OSError, ValueError):
            continue
        out.append(
            {
                "name": d.name,
                "path": _display_path(d.resolve(), s.project),
                "day": meta.get("day"),
                "start_date": (meta.get("config") or {}).get("start_date") or _find_key(meta, "start_date"),
                "prompts": len(prompts),
                "modified": head.stat().st_mtime,
            }
        )
    return sorted(out, key=lambda w: -w["modified"])


def _find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def list_examples(s: Settings) -> list[dict[str, Any]]:
    out = []
    for p in sorted((s.project / "examples" / "scenarios").glob("*.yaml")):
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        out.append(
            {
                "path": _display_path(p, s.project),
                "name": p.stem.replace("_", " "),
                "prompt": doc.get("prompt", ""),
                "horizon_days": doc.get("horizon_days"),
            }
        )
    return out


def info(s: Settings) -> dict[str, Any]:
    import importlib.util

    env = child_env(s)
    return {
        "project": str(s.project),
        "runs": _display_path(s.runs.resolve(), s.project),
        "worlds": list_worlds(s),
        "examples": list_examples(s),
        "configs": [_display_path(p, s.project) for p in sorted((s.project / "configs").glob("*.yaml"))],
        "countries": ["USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"],
        "llm": {
            "api_key_set": bool(env.get("ANTHROPIC_API_KEY")),
            "sdk_installed": importlib.util.find_spec("anthropic") is not None,
        },
        "today": time.strftime("%Y-%m-%d", time.gmtime()),
    }


def read_world_file(s: Settings, world: str, rel: str) -> str:
    base = _inside(world, [s.runs], s.project)
    target = (base / rel).resolve()
    if not target.is_relative_to(base) or target.suffix.lower() not in READABLE_SUFFIXES or not target.is_file():
        raise ArgvError("that file is not available")
    if target.stat().st_size > 5_000_000:
        raise ArgvError("file too large to show")
    return target.read_text(encoding="utf-8", errors="replace")


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "jahan-ui"
    settings: Settings
    jobs: Jobs

    def log_message(self, format: str, *args: Any) -> None:  # quiet: the terminal shows only the URL
        pass

    # ---------------------------------------------------------------- checks
    def _host_ok(self) -> bool:
        port = self.settings.port
        return self.headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _origin_ok(self, required: bool) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return not required
        port = self.settings.port
        return origin in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    def _token_ok(self) -> bool:
        return secrets.compare_digest(self.headers.get("X-Jahan-Token", ""), self.settings.token)

    # ---------------------------------------------------------------- replies
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, code: int, message: str, **extra: Any) -> None:
        self._json(code, {"error": message, **extra})

    # ---------------------------------------------------------------- routes
    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok():
            return self._error(403, "wrong host")
        url = urlparse(self.path)
        if url.path in STATIC:
            name, ctype = STATIC[url.path]
            body = resources.files("jahan.ui").joinpath("static", name).read_bytes()
            return self._send(200, body, ctype)
        if not url.path.startswith("/api/"):
            return self._error(404, "not found")
        if not (self._token_ok() and self._origin_ok(required=False)):
            return self._error(403, "open the page from the link `jahan ui` printed")
        q = parse_qs(url.query)
        parts = url.path.strip("/").split("/")
        try:
            if parts == ["api", "info"]:
                return self._json(200, info(self.settings))
            if parts == ["api", "jobs"]:
                return self._json(200, [j.view(len(j.lines)) for j in self.jobs.snapshot()])
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                job = self.jobs.jobs.get(int(parts[2]))
                if job is None:
                    return self._error(404, "no such job")
                return self._json(200, job.view(int(q.get("since", ["0"])[0])))
            if parts == ["api", "file"]:
                text = read_world_file(self.settings, q.get("world", [""])[0], q.get("path", [""])[0])
                return self._json(200, {"text": text})
        except (ArgvError, ValueError) as e:
            return self._error(400, str(e))
        return self._error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            self.close_connection = True
            return self._error(413, "request too large")
        raw = self.rfile.read(length)  # read before answering, even a refusal, so the connection ends cleanly
        if not self._host_ok():
            return self._error(403, "wrong host")
        if not (self._token_ok() and self._origin_ok(required=True)):
            return self._error(403, "open the page from the link `jahan ui` printed")
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return self._error(400, "bad JSON")
        if not isinstance(body, dict):
            return self._error(400, "bad JSON")
        parts = urlparse(self.path).path.strip("/").split("/")
        try:
            if parts == ["api", "jobs"]:
                job = self.jobs.start(body.get("argv"))
                return self._json(200, job.view())
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                return self._json(200, self.jobs.cancel(int(parts[2])).view())
        except Busy as e:
            return self._error(409, str(e), job=e.job_id)
        except (ArgvError, ValueError, KeyError) as e:
            return self._error(400, str(e))
        return self._error(404, "not found")


class _Server(http.server.ThreadingHTTPServer):
    """Binds its port exclusively.  The standard server sets SO_REUSEADDR, which on Windows lets a second
    process listen on the same port (a second `jahan ui`, or another program receiving the page's requests
    and token)."""

    daemon_threads = True
    allow_reuse_address = sys.platform != "win32"

    def server_bind(self) -> None:
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def make_server(s: Settings, ports: range) -> tuple[http.server.ThreadingHTTPServer, Jobs]:
    jobs = Jobs(s)
    handler = type("BoundHandler", (Handler,), {"settings": s, "jobs": jobs})
    last: OSError | None = None
    for port in ports:
        try:
            httpd = _Server(("127.0.0.1", port), handler)
        except OSError as e:  # port in use: try the next one
            last = e
            continue
        s.port = httpd.server_address[1]
        return httpd, jobs
    raise OSError(f"no free port in {ports.start}-{ports.stop - 1}: {last}")


def redirect_file(url: str) -> Path:
    """A private page that forwards to ``url``.  Opening it keeps the token off the browser's command line,
    where other users of the computer could read it (the approach Jupyter takes)."""
    fd, name = tempfile.mkstemp(prefix="jahan-open-", suffix=".html")  # readable by this user only
    link = html.escape(url, quote=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(
            f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={link}">'
            f'<title>Jahan</title><p><a href="{link}">Open Jahan</a></p>'
        )
    return Path(name)


def serve(project: Path, runs: Path, port: int, open_browser: bool, key_file: Path | None) -> None:
    env: dict[str, str] = {}
    if key_file is not None:
        key = key_file.expanduser().read_text(encoding="utf-8").strip()
        if not key:
            raise SystemExit(f"{key_file} is empty")
        env["ANTHROPIC_API_KEY"] = key  # only for the commands this server starts; never sent to the page
    s = Settings(project=project.resolve(), runs=runs.resolve(), token=secrets.token_urlsafe(24), env=env)
    s.runs.mkdir(parents=True, exist_ok=True)
    httpd, jobs = make_server(s, range(port, port + 20) if port else range(0, 1))
    url = f"http://127.0.0.1:{s.port}/#t={s.token}"
    print(f"Jahan is open at {url}", flush=True)
    print("This link carries a private key for this session; keep it to yourself. Press Ctrl+C to stop.", flush=True)
    page: Path | None = None
    if open_browser:
        page = redirect_file(url)
        webbrowser.open(page.as_uri())
        tidy = threading.Timer(120.0, lambda: page.unlink(missing_ok=True) if page else None)
        tidy.daemon = True  # Ctrl+C must not wait for it
        tidy.start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping; cancelling running commands.")
    finally:
        jobs.shutdown()
        httpd.server_close()
        if page is not None:
            page.unlink(missing_ok=True)
