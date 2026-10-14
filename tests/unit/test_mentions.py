"""Public figures are named in the news by aliases in several scripts, and react to top headlines."""

import shutil

import yaml

from jahan import paths
from jahan.actors import autonomy
from jahan.config import RunConfig
from jahan.gen.world_gen import generate_world
from jahan.model.mentions import mentions, script_of
from jahan.model.notables import NATIVE_SCRIPT, load_roster, validate_rosters
from jahan.scenario.inject import apply_scenario
from jahan.scenario.spec import NewsEvent, Scenario
from jahan.store.eventlog import EventLog

M1 = ["USA", "BRA", "DEU", "NGA", "IRN", "IND", "CHN", "IDN"]
TRUMP = ["Trump", "Donald Trump", "@POTUS", "@realDonaldTrump", "ترامپ", "特朗普"]
XI = ["Xi Jinping", "President Xi", "习近平", "习主席"]


def test_latin_aliases_match_whole_words_without_accents_or_case():
    assert mentions("Trump's tariffs hit Brazil", TRUMP)
    assert mentions("TRUMP SIGNS ORDER", TRUMP)
    assert not mentions("The rise of Trumpism", ["Trump"])
    assert not mentions("A post by @realDonaldTrump", ["Trump"])  # the handle is its own alias
    assert mentions("Statement from @POTUS tonight", TRUMP)
    assert mentions("Galipolo holds rates", ["Galípolo"]) and mentions("Galípolo holds rates", ["Galipolo"])
    assert not mentions("Mexico and Xi'an announce a festival", XI)
    assert mentions("President Xi meets Lula", XI)


def test_arabic_script_normalisation():
    assert mentions("ترامپ تهدید کرد", TRUMP)
    # Arabic yeh and kaf in the text, Persian forms in the alias
    assert mentions("رئيس جمهور پزشكيان گفت", ["پزشکیان"])
    # a zero-width non-joiner in the alias, a plain space in the text
    assert mentions("سخنان خامنه ای", ["خامنه‌ای"])
    assert script_of("خامنه‌ای") == "arabic"


def test_cjk_and_devanagari_substrings():
    assert mentions("习近平会见外宾", XI)
    assert mentions("प्रधानमंत्री नरेंद्र मोदी ने कहा", ["नरेंद्र मोदी"])
    assert script_of("नरेंद्र मोदी") == "devanagari" and script_of("习近平") == "cjk"


def test_every_figure_has_aliases_and_a_native_script_one():
    for iso in M1:
        for e in load_roster(paths.data_dir(), iso).entries:
            assert e.aliases, e.key
    for iso in ("IRN", "CHN", "IND"):
        for e in load_roster(paths.data_dir(), iso).entries:
            lang = {"IRN": "fa", "CHN": "zh", "IND": "hi"}[iso]
            assert any(script_of(a) == NATIVE_SCRIPT[lang] for a in e.aliases), e.key


def test_validator_requires_a_native_script_alias(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(paths.data_dir(), data)
    path = data / "notables" / "IRN.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["entries"][0]["aliases"] = ["Mojtaba Khamenei"]
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    problems = validate_rosters(data, ["IRN"])
    assert any("IRN.supreme_leader" in p and "arabic script" in p for p in problems)


def world(countries=("USA", "IRN"), n=1500, seed=3):
    return generate_world(RunConfig(n_agents=n, countries=list(countries), seed=seed, start_date="2026-10-13"))


def news(w, day, **kw):
    w.day = day
    apply_scenario(w, Scenario(prompt="t", events=[NewsEvent(**kw)]), EventLog(), day)
    w.day = day + 1


def chosen_keys(w):
    return {w.notables[c["col"]]["key"]: c for c in autonomy.triggers(w)}


def test_a_persian_headline_naming_trump_triggers_him():
    w = world()
    news(
        w,
        3,
        headline="ترامپ درباره تحریم‌ها صحبت کرد",
        topic="world",
        valence=-0.5,
        salience=0.4,
        countries=["IRN"],
        language="fa",
        origin="web",
    )
    got = chosen_keys(w)
    assert "USA.head_of_state" in got
    assert any("you were named" in r[2] for r in got["USA.head_of_state"]["reasons"])


def test_a_top_economy_headline_triggers_economic_figures_once():
    w = world()
    news(
        w,
        3,
        headline="Inflation jumps to 5%",
        topic="economy",
        valence=-0.6,
        salience=0.9,
        countries=["USA"],
        origin="web",
    )
    got = chosen_keys(w)
    assert {"USA.head_of_state", "USA.finance_minister", "USA.central_bank_governor"} <= set(got)
    assert not any(k.startswith("IRN.") for k in got)  # pushed only in the US
    assert any("top headline" in r[2] for r in got["USA.finance_minister"]["reasons"])
    # used once: the next day, the same item does not trigger again
    w.day += 1
    assert not any("Inflation" in r[2] for c in autonomy.triggers(w) for r in c["reasons"])


def test_low_salience_or_foreign_headlines_do_not_trigger():
    w = world()
    news(
        w, 3, headline="Minor budget note", topic="economy", valence=0.1, salience=0.5, countries=["USA"], origin="web"
    )
    assert not any("top headline" in r[2] for c in autonomy.triggers(w) for r in c["reasons"])
    w2 = world()
    news(
        w2,
        3,
        headline="Inflation jumps to 5%",
        topic="economy",
        valence=-0.6,
        salience=0.9,
        countries=["IRN"],
        origin="web",
    )
    assert not any(k.startswith("USA.") for k in chosen_keys(w2))


def test_caps_hold_under_a_flood_of_headlines():
    w = world()
    w.day = 3
    events = [
        NewsEvent(
            headline=f"Story {i}", topic="economy", valence=-0.5, salience=0.9, countries=["USA", "IRN"], origin="web"
        )
        for i in range(30)
    ]
    apply_scenario(w, Scenario(prompt="t", events=events), EventLog(), 3)
    w.day = 4
    chosen = autonomy.triggers(w)
    assert len(chosen) <= autonomy.TURNS_PER_WEEK
    assert len({c["col"] for c in chosen}) == len(chosen)


def test_a_social_mention_of_a_handle_counts_as_naming():
    w = world()
    news(
        w,
        3,
        headline="A post about the economy",
        topic="tech",
        valence=0.0,
        salience=0.3,
        countries=["USA"],
        origin="web",
    )
    item = w.items.n - 1
    w.macro.setdefault("social", {}).setdefault("items", {})[str(item)] = {"mentions": ["@POTUS"]}
    got = chosen_keys(w)
    assert "USA.head_of_state" in got
    assert any("you were named" in r[2] for r in got["USA.head_of_state"]["reasons"])
