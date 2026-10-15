"""Event kinds are unique; headline updates are told apart from reposts, also in logs written before the fix."""

import pyarrow.parquet as pq

from jahan.model.enums import EventKind, is_item_update
from jahan.scenario.report import _created_items
from jahan.store.eventlog import EventLog


def test_event_kinds_have_no_aliases():
    values = [m.value for m in EventKind.__members__.values()]
    assert len(values) == len(set(values)), "two EventKind names share a number"
    assert EventKind.ITEM_UPDATED != EventKind.SOCIAL_SHARED


def test_is_item_update_reads_new_and_legacy_rows():
    assert is_item_update(EventKind.ITEM_UPDATED, "Rial falls 9.0% against the US dollar")
    assert is_item_update(28, "Rial falls 9.0% against the US dollar")  # legacy update row
    assert not is_item_update(EventKind.SOCIAL_SHARED, "")  # a repost carries no detail
    assert not is_item_update(EventKind.ITEM_CREATED, "x")


def test_reposts_do_not_blank_report_labels(tmp_path):
    log = EventLog()
    log.add(0, EventKind.ITEM_CREATED, -1, item=7, detail="Rial falls 2.1% against the US dollar")
    log.add(1, EventKind.SOCIAL_SHARED, 42, -5, 7, 0.1, 0)  # a repost of the same item
    log.add(2, EventKind.ITEM_UPDATED, -1, item=7, detail="Rial falls 6.3% against the US dollar")
    log.add(3, EventKind.SOCIAL_SHARED, 43, -5, 7, 0.1, 0)
    path = tmp_path / "events.parquet"
    pq.write_table(log.table(), path)
    assert _created_items(path) == [(7, "Rial falls 6.3% against the US dollar")]
