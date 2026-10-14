# Jahan

*Jahan* (Persian for "world") is an open-source simulator of human society. It is not a game.
It runs thousands of agents whose demographic, economic, linguistic and informational profiles are drawn
from published statistics. Agents live in households, work and study together, keep friends and family
abroad, sleep on their local clock, read the media their habits allow, talk, share good and bad
information, trust each other more or less depending on how past advice turned out, marry, have
children, divorce, change jobs and die. You inject a scenario ("launch product X", "the dollar rises 20%",
"Iran–US tensions intensify") and watch its effects emerge, then rewind the world to any point before a
prompt.

> **Status:** Milestone 1 (Gate 0, 1a, 1b, 1c) is built. Eight countries (USA, Brazil, Germany, Nigeria,
> Iran, India, China, Indonesia), 10,000 agents, sourced demography and labour data, life events, diaspora
> ties, short trips abroad, products, news and macro scenarios, counterfactual experiments, reports, time
> travel. Notable public figures are in (real names, verified office holders as of 2026-10-13, sourced
> stances, parasocial trust, statements whose reach follows visibility). The LLM layer is in: free-text
> prompts, fed future events, persona interviews, readable transcripts and live news ingest. People can
> create agents or take control of any agent (including a head of state), talk, post, decide and use the
> powers of an office, turn by turn; public figures nobody controls decide on salient days. Everything is
> recorded for exact replay. A browser interface (`jahan ui`) covers every command. Not yet built:
> multiplayer (Milestone 2). See `docs/plan.md`.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

### In your browser

```
uv sync --extra llm                                    # the extra is only needed for Claude features
uv run jahan ui                                        # opens the interface in your browser
uv run jahan ui --key-file $HOME\.anthropic_key        # same, with your Anthropic key read from a file
```

The page is a remote control for the commands below. Every button runs a `jahan` command on your own
computer and shows its output and the exact command, so you can learn and script them later. Nothing is
simulated in the browser. You can:

- create and open worlds, and move time forward;
- add what-ifs in plain words, future events, ready-made examples, or today's real news, after checking a
  preview of the exact events;
- browse and filter people, open anyone's view of the world, ask them questions, and replay one of their
  days as a story;
- play anyone, including public figures: message your circle, post, decide, use the powers of an office,
  then finish the day;
- measure effects with experiments and read the reports;
- undo to before any prompt and replay your sessions.

The server listens only on `127.0.0.1`. It prints a private link containing a session key, and other
websites cannot use it to run commands. It runs one command per world at a time; a running command can be
cancelled.

### In a terminal

```
uv sync
uv run jahan gen        --run runs/demo --config configs/m1.yaml
uv run jahan run        --run runs/demo --days 30
uv run jahan inject     --run runs/demo --scenario examples/scenarios/dollar_rises.yaml --yes
uv run jahan run        --run runs/demo --days 120
uv run jahan experiment --run runs/demo --prompt 1 --seeds 4 --horizon 120
uv run jahan report     --run runs/demo --prompt 1          # runs/demo/reports/0001/report.md
uv run jahan agents     --run runs/demo --filter "country == 'IRN' and age > 30 and info_tier >= 2" --limit 5
uv run jahan agent      --run runs/demo 42                 # one person's profile, circle, schedule and views
uv run jahan reset      --run runs/demo --before-prompt 1  # time travel
```

Example scenarios in `examples/scenarios/`: a product launch, a 20% dollar rise during a worldwide launch,
a gold spike (+1%/day for a week, then −10%), Iran–US tension with a falling rial, and a false product
rumour.

### With an LLM

The LLM never changes the world directly. It turns your text into a structured scenario that you confirm,
labels headlines, and voices agents from what they actually know. Install the extra, set your own
Anthropic API key, and start a fresh world (news can only be ingested while the world date is within three
days of today, so `ingest` comes right after `gen`):

```
uv sync --extra llm
$env:ANTHROPIC_API_KEY = "sk-ant-..."            # PowerShell; use `export ANTHROPIC_API_KEY=...` on Linux/macOS
uv run jahan doctor --check-models              # one tiny request per model; each line should end in OK
uv run jahan gen        --run runs/live --config configs/m1.yaml
uv run jahan ingest     --run runs/live         # today's real headlines per country and language
uv run jahan run        --run runs/live --days 7
uv run jahan inject     --run runs/live --prompt 'Launch a $199 smart ring in the US, Germany, India and Iran'
uv run jahan run        --run runs/live --days 30
uv run jahan feed       --run runs/live "gold rises 1% a day for a week, then drops 10% in one day"
uv run jahan ask        --run runs/live --scope "group:country == 'IRN' and info_tier >= 2" "Would you buy the smart ring?"
uv run jahan transcript --run runs/live --agent 42 --day 35 --llm
```

- **Models and cost.** Scenario and feed parsing use `claude-sonnet-5-5`; headline labels, replies,
  interviews, transcripts and public figures' decisions use `claude-haiku-5-5`. Override them with
  `JAHAN_MODEL_STRONG` and `JAHAN_MODEL_FAST`. Each command prints its token use and an approximate cost at
  the list prices published on 2026-10-06 (Haiku 5.5: $0.10 / $0.50, Sonnet 5.5: $2 / $10 per million input
  / output tokens); check current prices. Every command is capped at 2M input and 400K output tokens;
  `JAHAN_MAX_USD` adds a dollar cap per command. Changing models changes which recordings match, so a re-run
  after a reset then calls the model again.
- `ingest` fetches RSS feeds listed in `data/feeds/feeds.yaml` (Google News per country and language;
  BBC Persian and IRNA for Iran, where Google News has no edition). The top stories per country become
  news items that reach only channels in their language. Without a key, headlines are labelled neutral:
  they raise awareness but do not move mood. It refuses when the world date is more than three days from
  the real date.
- When `run` crosses today's real date, it offers to take hypothetical events for the days with no real
  news (only in an interactive terminal; scripts get a one-line note). `feed` does the same at any time.
- Every web fetch and model answer is stored under `runs/<name>/recordings/`. `--provider replay` and
  `ingest --replay` use only those recordings and fail if one is missing, so resets and re-runs never
  depend on the network.
- Text written for a real public figure is always labelled `[simulated]`. It is not a real quote.

### Playing a person

You can take control of any agent, or create a new one, and act through it. The world waits for you at
each day boundary.

```
uv run jahan possess  --run runs/demo --agent role:USA.head_of_state   # or an agent id
uv run jahan post     --run runs/demo --audience national "We will announce new trade measures this week"
uv run jahan act      --run runs/demo tariff rate_pct=25               # only powers the office has
uv run jahan play     --run runs/demo --days 14                        # daily briefing, then your commands
uv run jahan release  --run runs/demo
uv run jahan experiment --run runs/demo --session 1 --seeds 4          # your session versus autopilot
uv run jahan report   --run runs/demo --session 1
uv run jahan create-agent --run runs/demo --country IRN --sex F --age 29 --describe "a nurse in Tabriz" --possess
```

`play` also reads commands from a file, for scripted sessions. For example, after possessing the US
president:

```
Get-Content examples/sessions/president_trade.txt | uv run jahan play --run runs/demo --days 7   # PowerShell
uv run jahan play --run runs/demo --days 7 < examples/sessions/president_trade.txt            # bash
```

- **What you control.** While you are at the controls, an agent stops sharing, buying, marrying, changing
  jobs and travelling on its own. It still hears what its circle and the media tell it.
- **Autopilot.** When the world moves without you, the engine drives your agents. This covers `run`, and
  any `play` day that times out or that you skip with `auto`. Your next command takes them back.
  `run --keep-control` keeps them idle instead. Autopilot days are recorded, so replays reproduce them.
- **Messages.** `say` (or `say <id> <text>` inside `play`) reaches one person who listens to you. They get
  it when they are awake, otherwise the next day. Whether they believe it depends on how much they trust
  you, their own view and their openness. The next briefing shows how they took it. With an LLM, they also
  reply in character, and the reply is told the outcome so the words match the numbers.
- **Posts.** Reach depends on who you are, not on how loudly you write. Anyone reaches their circle.
  Better-known people reach their local community, meaning friends of friends. Only public figures whose
  role allows it reach national or global media. The LLM only turns your words into a topic, a product
  and one of five stance levels; the engine sets salience from your visibility.
- **Powers of an office.** `act` checks `data/roles.yaml`, so a president cannot set interest rates and a
  central bank governor can. Effect sizes come from that file. They are invented placeholders and are
  labelled as such.
- **Limits.** An agent can send at most 20 messages and 3 posts a day.
- **Time travel.** Every turn is a prompt, so `reset --before-prompt N` undoes it. `session replay`
  re-applies the undone turns at their original days and reproduces the identical event log. Before each
  turn it checks two things. The recorded actions file must be unedited. The world must be in exactly the
  state the turn was played on. If either check fails, replay stops.
- **Public figures on their own.** With an LLM, each figure nobody controls gets a decision turn when
  something salient happens. Triggers are a relevant story spreading in their country, market news, being
  named in the news, a top headline on one of their role's topics in their country, or a scheduled meeting
  such as a central bank's. "Named" uses each figure's aliases from the roster: surnames, titles, handles
  and spellings in Persian, Arabic, Chinese and Hindi script. Common surnames (Modi, Lula, Hemmati) also
  match other people with that name. Each figure gets at most two turns a week, and the whole world at most
  twenty. Under a flood of headlines, scheduled meetings can lose their turn to more salient stories. The model picks from the role's permitted actions and the
  engine validates them. Experiments do not re-run these decisions inside their branches, and reports say
  so.
- **Drafting a real person.** `notables-draft` fetches an encyclopedia summary and news about a real person
  and drafts a roster entry. Every stance must cite one of the fetched URLs. The draft is written for you
  to verify and is never loaded by the simulator until you copy it into `data/notables/`.

**Isolating one shock.** An experiment compares "with prompt N" against "without prompt N", starting from
the snapshot taken just before N. To measure the effect of a dollar rise on an existing launch, inject the
launch as prompt 1, run some days, inject the dollar rise as prompt 2, and run the experiment on prompt 2.

## How it works

- **Deterministic numeric core.** All state lives in numpy column arrays. Every random draw is a pure
  function of (seed, stream, day, agent or edge id), so runs are reproducible across processes, phases are
  vectorised, and baseline and treated branches share identical draws. A year at 10,000 agents takes
  about 20 seconds on a 4-core laptop.
- **Time.** One tick is one day. Each agent has hourly awake, work and free masks in local time, so people
  asleep when news breaks see it later, and contacts (including with relatives abroad) need overlapping
  free hours.
- **People.** Age, sex, fertility and mortality come from UN World Population Prospects; urbanisation,
  internet use and exchange rates from the World Bank; education, employment and occupation from ILOSTAT
  labour force surveys. Households, marriages, children, divorces, job moves and deaths follow those rates.
- **Trust.** Each tie carries a Beta posterior over a prior built from tie type, reputation, homophily and
  an attractiveness halo that fades as evidence accumulates. Bad outcomes weigh 2.5 times good ones.
- **Political camps.** Each country has one political axis: +1 is the camp of the head of government on
  the roster date, -1 the main opposition (`politics:` in each country file). Adults are spread over the
  camps in sourced shares: US party identification (Gallup), the German Sonntagsfrage, and election results
  for Brazil, India, Indonesia and Nigeria, where non-voters count as neutral. Iran and China use
  placeholder thirds. Sourced group differences move people along the axis; so far these are Indian
  language communities (a stand-in for region) and a near-zero religion effect in Brazil. Children keep a
  parent's camp about 70% of the time. Trust in a public figure starts from their approval within the
  person's camp. Trump's comes from Gallup's party breakdown, and Merz's from the share of CDU/CSU
  supporters satisfied with him. Figures without a breakdown borrow the US gap between camps, shifted to
  their own approval. Outside the US the gap is halved on the log-odds scale, because the US is unusually
  polarised. In every case the population average equals the figure's overall approval. A
  leader's statement still reaches opponents, who mostly stay unconvinced. In a test, two weeks after an
  upbeat presidential statement, the people who had heard it held an average view of 0.76 among supporters
  and 0.58 among opponents (0.5 is undecided).
- **Information.** Media channels reach agents by information habits, language and internet access.
  Agents share what they believe; dishonest agents exaggerate; well-informed receivers pull claims toward
  the truth; acceptance depends on trust and a bounded-confidence window. Buzz fades with novelty.
- **Mood and markets.** Exchange-rate and gold paths move local prices of dollar-priced goods and create
  headlines; each agent's economic mood follows the news they know and shifts purchase decisions.
- **Time travel.** Every prompt writes a snapshot first. `reset` restores it and truncates the logs;
  re-running produces the identical timeline.

## Data honesty

Every distribution cites an entry in `data/registry.yaml` with a confidence level. Demography (UN WPP 2024),
World Bank indicators, ILOSTAT labour tables and UN DESA migrant stock are downloaded and converted by
scripts in `scripts/data/`; other blocks (ethnicity, religion, languages, household structure, incomes,
media habits, names) are `provisional` or `low` and say what source should replace them. Personality and
values are placeholders. Reports list these caveats. `uv run jahan validate-data --strict` fails until they
are replaced. Sourced country data is the most valuable contribution you can make; see `CONTRIBUTING.md`.

Known limitations: political camps are one axis per country, so Germany's AfD, Greens and Left share one
opposition camp; outside the US and Germany the gap between camps is borrowed from US polling and halved,
which is a guess, not a measurement; in Nigeria, India and Indonesia people who did not vote count as
neutral, although many of them are disaffected rather than moderate; Iran and China
have placeholder camps and flat trust within the country; diaspora agents take their residence country's
camps; backlash (opponents moving away from a leader's claim) is not modelled; without polygyny,
countries with large spousal age gaps leave some older women unmarried; behavioural coefficients are uncalibrated priors tuned to plausible ranges. Vital-event counts in
small country samples (600–2,000 agents) are noisy; the mechanisms reproduce their input rates over long or
large runs (checked: births, deaths, marriages, divorces). Media reach was tuned on product launches; for
ingested top national headlines (without LLM labels) it gives about 1–8% awareness within three days,
typically about 5%, which is probably too low for major stories and is not yet calibrated against
news-attention surveys.

Several numbers for player actions and public-figure autonomy are heuristics, not sourced values:

- the community reach of a post;
- the daily message and post limits;
- the awareness level that triggers a public figure;
- the headline salience (0.7) that triggers one;
- the weekly turn caps;
- every policy elasticity in `roles.yaml`;
- the camp band (a lean beyond ±1/3 counts as a camp), the weight of value similarity in trust (1.5 on
  the log-odds scale) and the discount for foreign figures (0.8);
- the US gap between camps borrowed for figures without a breakdown, and halving it outside the US.

Worlds created before political camps existed (snapshot schema 5) cannot be opened; create a new world.

Each distinct message or post occupies one tracked-item slot for about a week. A very talkative session
can fill the `max_slots` pool (64 by default). Further claims are then skipped and logged as such.

## Rebuilding the data

```
uv run python -I -X utf8 scripts/data/build_wpp.py --raw data/raw/wpp2024 --out data/generated/wpp2024 --year 2026 --countries USA,BRA,DEU,NGA,IRN,IND,CHN,IDN
uv run python -I -X utf8 scripts/data/build_wdi.py --out data/generated/wdi --countries USA,BRA,DEU,NGA,IRN,IND,CHN,IDN
uv run python -I -X utf8 scripts/data/build_ilo.py download --raw data/raw/ilostat --countries USA,BRA,DEU,NGA,IRN,IND,CHN,IDN
uv run python -I -X utf8 scripts/data/build_ilo.py build --raw data/raw/ilostat --out data/generated/ilo --countries USA,BRA,DEU,NGA,IRN,IND,CHN,IDN
uv run python -I -X utf8 scripts/data/build_migration.py --xlsx data/raw/undesa_ims2024/undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx --out data/generated/migration/stock_2024.yaml
```

The UN WPP CSVs (about 265 MB) and the migrant stock workbook are downloaded by hand into `data/raw/`
(git-ignored); the URLs are recorded in the generated files and in the registry.

## License

Apache-2.0.
