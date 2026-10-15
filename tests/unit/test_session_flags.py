"""A session remembers separately whether public figures acted and whether stories were rated (replay must match)."""

from jahan.actors import session
from jahan.store.rundir import RunDir


def test_autonomy_and_odds_are_remembered_separately(tmp_path):
    rd = RunDir(tmp_path)
    sid = session.new_session(rd)
    assert not session.autonomy_used(rd, sid) and not session.odds_used(rd, sid)
    session.note_odds(rd, sid, True)  # e.g. `run --no-autonomy` with watched questions
    assert session.odds_used(rd, sid) and not session.autonomy_used(rd, sid)
    other = session.new_session(rd)
    session.note_autonomy(rd, other, True)
    data = session.load(rd)
    del data["sessions"][str(other)]["autonomy"]
    data["sessions"][str(other)]["autonomy"] = True  # a session recorded before the odds flag existed
    session.save(rd, data)
    assert session.odds_used(rd, other)  # it rated stories exactly when figures acted
