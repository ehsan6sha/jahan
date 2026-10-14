"""`jahan ui`: the command allow-list, path confinement, request checks, and jobs run through HTTP."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from jahan.ui.server import ArgvError, Settings, check_argv, cmdline, make_server
from tests.conftest import ROOT

TOKEN = "test-token"


def _settings(tmp_path) -> Settings:
    return Settings(project=ROOT, runs=tmp_path / "runs", token=TOKEN, env={"JAHAN_OFFLINE": "1"})


def test_allow_list_and_paths(tmp_path):
    s = _settings(tmp_path)
    world = tmp_path / "runs" / "w"
    argv, w = check_argv(["run", "--run", str(world), "--days", "3"], s)
    assert w == str(world.resolve()) and argv[0] == "run"
    _, w = check_argv(
        ["inject", "--run", str(world), "--scenario", "examples/scenarios/gold_spike.yaml", "--preview"], s
    )
    assert w == str(world.resolve())
    assert check_argv(["doctor"], s)[1] is None  # no world needed
    for bad in (
        ["rm", "-rf", "/"],  # not a jahan command
        ["python", "-c", "print(1)"],
        ["run", "--days", "3"],  # a world command without a world
        ["run", "--run", str(tmp_path / "elsewhere")],  # outside the runs folder
        ["run", "--run", str(tmp_path / "runs" / ".." / "x")],  # escapes with ..
        ["run", "--run", str(tmp_path / "runs")],  # the runs folder itself is not a world
        ["run", f"--run={world}"],  # path options must be separate arguments
        ["inject", "--run", str(world), "--scenario", str(tmp_path / "evil.yaml")],
        ["session", "delete", "--run", str(world)],
        ["run", "--run", str(world), f"-r{tmp_path / 'elsewhere'}"],  # a glued short option the CLI would honour
        ["inject", "--run", str(world), "-yr", str(tmp_path / "elsewhere")],  # a cluster hiding -r
        ["run", "--run", str(world), "--run", str(tmp_path / "runs" / "other")],  # the CLI keeps the last one
        "run --run x",  # not a list
        [],
    ):
        with pytest.raises(ArgvError):
            check_argv(bad, s)
    # negative numbers and the real short options are fine
    check_argv(["post", "--run", str(world), "--stance", "-0.5", "-y", "--", "prices are rising"], s)
    # "--" can be swallowed as an option's value, after which the CLI would honour a second --run: so the
    # tokens after "--" are checked too
    for bad in (
        ["inject", "--run", str(world), "--prompt", "--", "--run", str(tmp_path / "elsewhere"), "--yes"],
        ["say", "--run", str(world), "--to", "3", "--", "--run", "../../etc"],
        ["say", "--run", str(world), "--to", "3", "--", "-rX"],
    ):
        with pytest.raises(ArgvError):
            check_argv(bad, s)
    argv, _ = check_argv(["say", "--run", str(world), "--to", "3", "--", "see you at 10"], s)
    assert argv[-1] == "see you at 10"
    assert cmdline(["say", "--run", "runs/w", "--", "hello there"]) == 'jahan say --run runs/w -- "hello there"'


def test_browser_is_opened_through_a_private_redirect_file():
    from jahan.ui.server import redirect_file

    page = redirect_file("http://127.0.0.1:8765/#t=abc&x")
    try:
        text = page.read_text(encoding="utf-8")
        assert 'url=http://127.0.0.1:8765/#t=abc&amp;x"' in text
        if hasattr(page.stat(), "st_mode") and __import__("os").name == "posix":
            assert page.stat().st_mode & 0o077 == 0  # nobody else can read the token
    finally:
        page.unlink()


def test_every_path_option_of_an_allowed_command_is_confined():
    """A new path-like option on any command the page may run must be added to PATH_OPTIONS (or this fails)."""
    import typer.main  # class names, not imports: recent Typer bundles its own copy of click

    from jahan.cli import app
    from jahan.ui.server import GLOBAL_COMMANDS, PATH_OPTIONS, SESSION_SUBCOMMANDS, WORLD_COMMANDS

    def walk(cmd, name):
        if hasattr(cmd, "commands"):
            for sub, c in cmd.commands.items():
                yield from walk(c, f"{name} {sub}".strip())
        else:
            yield name, cmd

    allowed = (WORLD_COMMANDS | GLOBAL_COMMANDS) - {"session"} | {f"session {s}" for s in SESSION_SUBCOMMANDS}
    seen = set()
    for name, cmd in walk(typer.main.get_command(app), ""):
        if name not in allowed:
            continue
        seen.add(name)
        for p in cmd.params:
            if type(p.type).__name__.endswith("Path"):
                assert "Option" in type(p).__name__, f"{name}: positional path {p.name}"
                assert set(p.opts) <= set(PATH_OPTIONS), f"{name}: {p.opts} is not confined"
    assert seen == allowed  # every allowed command exists


@pytest.fixture()
def server(tmp_path):
    s = _settings(tmp_path)
    httpd, jobs = make_server(s, range(0, 1))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield s, jobs
    jobs.shutdown()
    httpd.shutdown()
    httpd.server_close()


def _req(s, path, body=None, token=TOKEN, origin="self", host=None):
    url = f"http://127.0.0.1:{s.port}{path}"
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Jahan-Token"] = token
    if origin == "self":
        headers["Origin"] = f"http://127.0.0.1:{s.port}"
    elif origin:
        headers["Origin"] = origin
    if host:
        headers["Host"] = host
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _wait(s, job_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        code, body = _req(s, f"/api/jobs/{job_id}")
        view = json.loads(body)
        if view["status"] != "running":
            return view
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_port_is_not_shared(server, tmp_path):
    """A second server (another `jahan ui` or any program) cannot listen on the same port."""
    s, _ = server
    with pytest.raises(OSError, match="no free port"):
        make_server(_settings(tmp_path), range(s.port, s.port + 1))


def test_requests_need_token_host_and_origin(server):
    s, _ = server
    code, body = _req(s, "/", token=None, origin=None)
    assert code == 200 and b"<title>Jahan</title>" in body  # the page itself holds no secret
    assert _req(s, "/api/info", token=None, origin=None)[0] == 403
    assert _req(s, "/api/info", token="wrong", origin=None)[0] == 403
    assert _req(s, "/api/info", host="evil.example:80", origin=None)[0] == 403  # DNS rebinding
    code, body = _req(s, "/api/info", origin=None)
    assert code == 200 and "worlds" in json.loads(body)
    assert _req(s, "/api/jobs", {"argv": ["doctor"]}, origin=None)[0] == 403  # writes need the page's origin
    assert _req(s, "/api/jobs", {"argv": ["doctor"]}, origin="https://evil.example")[0] == 403
    assert _req(s, "/api/jobs", {"argv": ["rm", "-rf", "/"]})[0] == 400


def test_jobs_run_commands_one_at_a_time_per_world(server, tmp_path):
    s, _ = server
    world = str(tmp_path / "runs" / "w")
    code, body = _req(s, "/api/jobs", {"argv": ["gen", "--run", world, "--agents", "300", "--countries", "USA"]})
    assert code == 200
    view = _wait(s, json.loads(body)["id"])
    assert view["status"] == "ok" and any("Generated" in line for line in view["lines"])
    code, body = _req(s, "/api/jobs", {"argv": ["run", "--run", world, "--days", "200", "--no-autonomy"]})
    long_job = json.loads(body)["id"]
    code, body = _req(s, "/api/jobs", {"argv": ["status", "--run", world, "--json"]})
    assert code == 409 and json.loads(body)["job"] == long_job  # the world is busy
    code, body = _req(s, f"/api/jobs/{long_job}/cancel", {})
    assert code == 200
    assert _wait(s, long_job)["status"] == "cancelled"
    code, body = _req(s, "/api/jobs", {"argv": ["status", "--run", world, "--json"]})
    view = _wait(s, json.loads(body)["id"])
    st = json.loads(view["lines"][-1])
    assert view["status"] == "ok" and st["agents_alive"] > 0
    code, body = _req(s, "/api/info", origin=None)
    assert [w["name"] for w in json.loads(body)["worlds"]] == ["w"]
    code, _ = _req(s, f"/api/file?world={world}&path=../../secret.txt", origin=None)
    assert code == 400
