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
> recorded for exact replay. A watch list follows US stocks (moved partly by simulated holders' trading),
> currencies, moods and the odds of real prediction-market questions as time moves. A browser interface
> (`jahan ui`) covers every command. Not yet built:
> multiplayer (Milestone 2). See `docs/plan.md`.

## Disclaimer

Jahan is open-source software provided "as is", without warranty of any kind, express or implied. It is
a solely educational and testing project, meant for learning, research and experiments.

- **Not advice.** Nothing it produces is financial, investment, trading, legal, medical or any other
  professional advice.
- **Not a forecast.** Simulated people, opinions, news reach and market moves are not predictions of real
  ones. No accuracy is promised, including for the [benchmark](#benchmark-backtests-against-real-events-2023-2026).
- **Your own risk.** You use it at your own risk. The developers and contributors accept no liability for
  any loss or damage arising from its use. See sections 7 and 8 of the [LICENSE](LICENSE).

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
- watch US stocks, US tech stocks, gold, currencies and each country's mood move as time passes (the
  Watch page), with the part moved by people's trading shown separately;
- watch the odds of real prediction-market questions (searched on Polymarket, or saved ones that work
  offline) or your own, held by simulated traders, next to the real price;
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
uv run jahan watch add  --run runs/demo market:us_stocks market:us_tech_stocks mood:USA
uv run jahan run        --run runs/demo --days 7
uv run jahan watch show --run runs/demo                   # how each watched series moved, and why
uv run jahan watch catalog --run runs/demo               # what can be watched, with saved questions
uv run jahan watch add  --run runs/demo odds:seed:us-x-iran-ceasefire-continues-through-october-31-20260917
uv run jahan inject     --run runs/demo --scenario examples/scenarios/iran_hormuz_odds.yaml --yes
uv run jahan backtest                                      # the benchmark below (offline, about 4 min)
```

Example scenarios in `examples/scenarios/`: a product launch, a 20% dollar rise during a worldwide launch,
a gold spike (+1%/day for a week, then −10%), Iran–US tension with a falling rial, a false product
rumour, and the Fed's 16 September 2026 rate hike (for a world dated the day before: `gen --start-date
2026-09-15 --allow-later-roster`).

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
uv run jahan ingest     --run runs/live         # today's real headlines and social posts per country and language
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
- **Social media.** `ingest` also reads posts by public figures, institutions and channels, and topic
  searches, from networks that can be read for free (accounts in `data/feeds/social.yaml`, checked on
  2026-10-14):
  - X, through an unofficial reader (unifapi.com). X has taken scrapers to court and several free readers
    have shut down, so this source may stop working. It is on by default; `--no-x` skips it. The free tier
    allows about ten searches per ten minutes, so a second ingest within ten minutes may get nothing from X.
  - Bluesky and Mastodon, through their official public APIs.
  - Telegram public channel previews (for example BBC Persian).

  Facebook, TikTok, WhatsApp, Instagram and WeChat cannot be read for free, so China has no source. The top
  posts per country (`--per-country-social`, default 2) become items that reach people through the social
  channels in the post's language. A post by a public figure on the roster counts as that figure's words, so
  people weigh it by their trust in them. Reach comes from the author's followers (a figure's roster
  visibility) and the post's likes, reposts and views, never from a model. Each platform's push strength is its
  share of the country's internet users (DataReportal 2026). In their feeds, people see more posts on topics
  they care about and more posts by figures they trust (heuristic weights). Posts by ordinary people (fewer
  than 100,000 followers) only count towards trending topics: their words are not spread, stored in the
  world or shown. Trending topics are context for figures' decisions and for you; they do not change the
  simulation. Posts in a language nobody in that country reads in this world are dropped, for example
  Modi's English posts, because India's social channels here are Hindi and regional. `--no-news` and
  `--no-social` fetch one kind only. `jahan social-feed --run runs/live` lists the real posts and the posts
  made in the world, with how far they got. Public figures' real posts are shown to them when they decide
  what to do and in interviews.
- **Reposting.** Ordinary people who are online, convinced by a story and on its side may repost it, once
  per story, on a day within a week of hearing it; extroverts and people interested in the topic do it more.
  A repost does not create a new story. It adds a short, weak push on the social channels of the reposter's
  country, in their languages, sized by their audience (10^visibility people) as a share of everyone online
  there. Pushes never exceed full strength, rise by at most 0.5 a day, and reposts never keep a story
  tracked for more than 30 days. Products are not reposted. A handful of reposts barely moves national
  awareness: in a test, reposts by 14 of 5,000 people raised awareness of a top story from 9.5% to 9.7%.
  Posts you make in your community can reach social media only this way. All repost coefficients are
  heuristics (`repost_*` in `SimParams`).
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
  and one of five stance levels; the engine sets salience from your visibility. Briefings show how far each
  post got: simulated people reached, the real people they stand for, and how many you convinced.
  `social-feed --kind virtual` lists every post made in the world, including public figures' statements.
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
  An *announcement* (a Federal Reserve rate decision or a US jobs report) moves a US stock index, the
  dollar against the euro and, for jobs reports, gold by published event-study estimates times the
  surprise, which is the announced value minus what markets expected the day before
  (`data/global/markets.yaml`, each estimate cited in the registry). A fully expected decision moves
  nothing. The estimate for gold after rate decisions is shown but not applied, because studies disagree
  on its sign. Other currencies and gold move only when a prompt scripts them.
- **People's trading in US stocks.** The US stock index has a second part, moved by the simulated people.
  - **Who holds stock.** US adults hold stock with the shares the Federal Reserve's Survey of Consumer
    Finances 2022 reports by household income: 34% of the bottom half, 78% of the next 40% and 95% of the
    top tenth. Each holder's stake is that group's median holding: $12,600, $53,200 or $608,000. Jahan ranks
    adults by household income where the survey ranks families. The draw reproduces the overall share
    (56% of simulated adults, against Gallup's 62%), but not the gap by education: 61% of simulated
    college graduates hold stock against Gallup's 84%, and 55% of those with high school or less against
    42%. Residents of other countries hold nothing, for lack of a sourced share.
  - **How they trade.** On weekdays, holders with internet trade on the news they learned that day: how
    much better or worse it makes the economy look (weekend news is traded on Monday). Net buying moves
    the index by five times the money, the multiplier Gabaix and Koijen (2021) estimate for flows into the
    whole market; here it is applied to the simulated households' flow, a simplification. How much of
    their holdings people move per unit of news, kappa, is the one fitted number (0.058; see the
    benchmark). The simulated holders stand in for all US household equity; institutions are not
    modelled, and there is no random daily noise, so quiet days do not move at all.
  - **What it can and cannot do.** When a story stops being tracked, people buy back what they sold on
    it, so this part fades with the news and cannot make lasting trends. Its sizes are illustrations of the
    mechanism, not calibrated price impact: one multiplier for every flow, no opposing traders and no
    noise. In a test, the example Iran–US
    tension story took the people's part down 6.8% over four weeks, and it was back above its start
    two months later. A recession headline took it down 5.0%. Real US–Iran escalation days in 2026 moved the
    S&P 500 by less than 1% each.
  - **Turning it off.** On the 2026 events it made the forecasts worse (see the benchmark). To run without
    it, put `params: {market_people: false}` in the run config before `jahan gen`; announcements and
    scripted moves then move US stocks alone.
  - **US tech stocks (Nasdaq-100)** are derived, not modelled on their own: each day they move by 1.299
    times the US stock index's move, the slope of the Nasdaq-100's daily changes on the S&P 500's from
    January 2023 to October 2026 (FRED; R² 0.89; only the coefficient is stored).
- **Event odds.** A watched yes/no question ("US x Iran ceasefire continues through October 31?") is held
  as a belief by simulated traders: about one in ten internet-using adults, more of them young men (a
  low-confidence US survey figure, borrowed for every country, and a heuristic skew). They start with
  beliefs spread around the real market price, so their average equals it, and the simulated odds are
  that average: with log utility a prediction market's price equals its traders' mean belief (Wolfers and
  Zitzewitz 2006). Others can hear of the question by word of mouth, but only traders count.
  - **What moves the odds.** News moves them when traders hear it. A story can bear on a question, either
    written into a what-if (`bears_on`, shown in the preview) or rated by Claude. Claude rates each new
    story against each watched question as time moves: a direction from -1 to +1, and 0 for most stories.
    It never sets a probability, and its ratings are recorded for exact replay. A trader who hears a story
    rated +1 multiplies their odds of Yes by about 2.7 (a heuristic). So a story moves the odds as far as
    the simulation spreads it.
  - **Questions to watch.** `jahan watch catalog --polymarket` searches Polymarket's public data, and
    `watch add odds:<market>` starts from today's price. Ten saved questions (prices read on 15 October
    2026) work offline, or you can ask your own. `watch fetch` reads today's real prices for comparison;
    they never move the simulation. Check Polymarket's terms of use before reusing its prices; Jahan stores
    only the question, its page, the dates and the prices it read.
  - **What it is not.** The odds are a belief average, not a market. There is no order book, no liquidity
    and no drift toward the answer as the deadline nears. Nothing pulls them back toward the real price,
    and Jahan does not know how a question resolves; it closes the question on its date.
- **Time travel.** Every prompt writes a snapshot first. `reset` restores it and truncates the logs;
  re-running produces the identical timeline.

## Benchmark: backtests against real events, 2023-2026

> **Read this first.** Most market moves in this benchmark come from published event-study estimates
> multiplied by the surprise (see "Mood and markets" above). Those estimates are imposed from the studies,
> not learned by the simulation, so those rows test the estimates as Jahan applies them. For US stocks the
> simulated holders' trading adds a second part ("People's trading" above); its one number was fitted on
> the 2023-2025 events and is tested on 2026, and on 2026 it made the forecasts worse. The simulation also
> adds how far the news spreads and how public figures react. Educational and testing use only; not
> financial advice; not a forecast; no accuracy is promised.

This is a descriptive replay of published estimates on real events, not a blind test of forecasting skill.
It covers every Fed rate decision and every US jobs report from January 2023 to October 2026: 30 decisions
and 37 jobs reports, 67 announcements in all.

**What the backtests found**

- **Overall: no better than "no change" for stocks and gold, a little better for the dollar.** Average
  miss, measured in typical daily moves (one standard deviation of daily changes):

  | Market | Jahan | "No change" |
  |---|---|---|
  | US stocks | 1.07 | 1.06 |
  | Gold | 0.84 | 0.85 |
  | Dollar | 1.15 | 1.23 |

- **People's trading: slightly better in the years it was fitted on, worse on the year held out.**
  Average miss for US stocks, in typical daily moves:

  | Events | Estimates only | With people's trading | "No change" |
  |---|---|---|---|
  | 2023-2025, used to fit kappa (58) | 1.07 | 1.02 | 1.06 |
  | 2026, held out (8) | 1.03 | 1.11 | 1.07 |

  On the 2026 events the people's part made the forecast worse, mostly on the two jobs reports. Eight
  events are far too few to judge it either way. On 35 of the 66 days, mostly jobs reports, the people's
  part pointed the other way from the published estimate: holders buy on good economic news, while the
  estimate says good job news is bad for stocks because it means higher rates.
- **Direction: 25 right out of 41 (61%), not clearly better than a coin toss.** Only forecasts of at least
  a quarter of a typical move count. By market: stocks 8 of 15, gold 3 of 4, dollar 14 of 22. A fair coin
  does at least this well about one time in nine (one-sided p = 0.11). The real chance is higher still,
  because calls on the same day are not independent.
- **Major misses: 26 out of 196 market forecasts (13%).** The limits were fixed before the first run.
- **Fed decisions were almost always expected,** so Jahan forecasts almost no move on those days. Only one
  of the 30 was a real surprise: the 50 bp cut on 18 Sep 2024, when markets gave a 50 bp cut about 63%.
  Jahan got the dollar exactly (-0.29%, as in reality). It forecast stocks +0.36% against a real -0.29%,
  within a day's normal noise. The big Fed-day moves came from forward guidance, such as the hawkish
  projections that sent stocks down 2.95% on 18 Dec 2024, and from other news.
- **Jobs reports and stocks: the published sign did not hold.** The estimate comes from 1994-2001
  expansions, when good job news was bad for stocks because it meant higher rates. In 2023-2024, stocks
  often rose on strong reports instead (May, June and October 2023; February and April 2024). The paper
  itself shows the sign flips with the business cycle. Here it was right 8 times out of 15.
- **News reach: plausible.** Two days after the 16 Sep 2026 hike, 14.8% of simulated US adults had heard of
  it (14.0-15.9% across 5 seeds). That is inside the range implied by surveys of US consumers around Fed
  press conferences (Lamla and Vinogradov 2019). It is a plausibility check on one announcement, not a
  validation of the social layer: the survey's 40% includes people who had already heard Fed news, while
  Jahan starts from zero and tracks only this headline.
- **Public figures: right after one sourced fix.** In the first run, simulated Trump stayed silent, unlike
  the real one. After one sourced stance was added, his reaction matched the real one in direction. The
  figures' answers depend heavily on the documented stances they are given; the fix changed that data, not
  the engine. Details below.

**Method.** Run `uv run jahan backtest` (offline and deterministic, about 4 minutes).
- `--years 2026 --out FILE.md` writes the tables below.
- `--full-out docs/backtests.md` writes every event: [docs/backtests.md](docs/backtests.md).

- **The world.** For each event, Jahan builds a world dated the day before the release, using the m1
  config (10,000 people, eight countries). It injects only what was known before the release: the
  decision or the jobs number, and what markets expected. Then it runs three days.
- **What the world lacks.** It holds people, political camps and public figures, but no news from before
  the event. The figures' roster is dated 13 Oct 2026, after the events, so their approval ratings and
  some of their stances include later information. The market rows do not use the roster.
- **The data.** Real moves and every source URL are in `data/backtests/events_2023_2025.yaml` and
  `events_2026.yaml`.
  - **S&P 500 and 2-year yield changes** come from FRED. `scripts/check_market_actuals.py` re-checks all
    142 of them, and all match.
  - **Gold** comes from exchange-rates.org's yearly pages for 2023-2025 and mostly from the same site for
    2026. The pages do not state their pricing time, and on some days they differ noticeably from spot
    prices taken after the New York close, so the gold rows are the least reliable.
  - **The dollar index (DXY)** comes from Yahoo Finance; three 2023-2025 days where Yahoo repeats the
    previous close are left out.
  - **Vendor data.** Only daily percent changes are quoted, each with its page. Check the vendors' terms
    before reusing them.
  - **Fed expectations.** These are CME FedWatch probabilities as reported in the news the day before (or
    the morning of) each decision. Four 2023-2024 decisions had only phrases such as "near-certain" and are
    treated as fully expected; most of 2026's are assumed or secondary (marked in the file).
  - **Jobs consensus.** This is the Dow Jones survey, as reported by CNBC.
  - **Excluded or confounded days.** The 7 Apr 2023 jobs report came out on Good Friday, when US stocks
    were closed. The 4 Apr 2025 report fell in the middle of the tariff crash (S&P 500 -5.97%).- **The estimates** (`data/global/markets.yaml`, each with the paper's own number and a registry entry):

  | Announcement | Market | Published estimate | Window | Source |
  |---|---|---|---|---|
  | Fed decision surprise | US stocks | -3.961% per percentage point | daily | Gurkaynak, Sack and Swanson 2005 |
  | Fed decision surprise | dollar index | +0.19% per 6 bp | daily | Roache and Rossi 2009 |
  | Fed decision surprise | gold | -0.31% per 6 bp (not applied) | daily | Roache and Rossi 2009 |
  | Fed forward guidance | US stocks; dollar | -0.14%; +0.21% per 4.61 bp of the 2-year yield | 30 minutes | Swanson 2021 |
  | Jobs report surprise | US stocks | -0.220% per 77.5k jobs | 5 minutes | Andersen, Bollerslev, Diebold and Vega 2007 |
  | Jobs report surprise | gold; dollar index | -0.18%; +0.13% per 77.5k jobs | daily | Roache and Rossi 2009 |

  The samples run from 1990 to 2019, so every event here (2023-2026) is outside them; nothing was fitted to
  these events.
- **Fitting people's trading.** `uv run jahan backtest --fit-people` chooses kappa under a rule written
  in `data/global/investors.yaml` before the first fit:
  - **The loss.** It minimises the mean absolute miss of the US stock forecast (estimates plus people), in
    typical daily moves, over the 58 forecast events of 2023-2025 with a real stock move.
  - **The grid.** It searches kappa from 0 to 2 in steps of 0.001, at 10,000 people. The 2026 events never
    enter the choice.
  - **Stability.** A scripted 3% fall must not spiral and must be at least half undone within 40 days.
  - **The record.** The full record, with how big two what-if stories get, is in the data file.
  - **The first version failed.** In the first version people traded as their mood drifted toward the
    news. Mood moves a tenth of the way each day, so on a release day the fit saw a tenth of the reaction
    and chose a kappa ten times larger (0.581). The example Iran–US tension story then took the index down
    47% over four weeks.
  - **The change.** People now trade on the news they learn that day. The same rule chose kappa = 0.058,
    with the same release-day results and much smaller multi-day moves. The 2026 results had been seen
    before this change, so they are no longer a clean out-of-sample test; treat them as exploratory until
    events after October 2026 are scored. The change was prompted by the size of the what-if moves, not
    by the 2026 numbers, which are the same under both versions. Both fits are in the file.
  - **The fit sees only release days.** Most of the people's reaction comes on later days, as the news
    spreads, so the stability test, not the fit, governs those.
- **Scoring.** The limits were fixed before the run:
  - A major miss is one larger than 2 typical daily moves, or a wrong direction on a real move larger than
    one.
  - A typical daily move (2023 to October 2026) is 0.92% for the S&P 500 (from FRED), and about 1.28% for
    gold and 0.41% for DXY (vendor data, approximate). The gold figure comes from front-month futures,
    whose contract rolls may overstate it, while the real gold moves are spot prices. The gold verdicts
    stay the same for a typical gold move between 1.24% and 1.58%; with a smaller one, more gold rows would
    count as major misses.
  - A direction only counts when the forecast is at least a quarter of a typical move.

The 2026 market data were seen before these limits were written; the 2023-2025 events were added
afterwards with the same limits and unchanged estimates. The limits are generic and no estimate was fitted
to any of these events.

<!-- generated by `jahan backtest` (m1 config, 10,000 people, 5 seeds for news reach); edit data/backtests, not this table -->

**Summary.** Average miss in typical daily moves, Jahan's published estimates / "no change"; major misses out of forecasts; direction right out of the forecasts large enough to have one (a quarter of a typical move or more).

| Events | Count | US stocks | Gold | Dollar | Major misses | Direction right |
|---|---|---|---|---|---|---|
| Fed decisions 2023 | 8 | 0.88 / 0.93 | 0.72 / 0.72 | 1.63 / 1.56 | 4 of 22 | 0 of 2 |
| Fed decisions 2024 | 8 | 1.31 / 1.26 | 0.97 / 0.97 | 1.10 / 1.15 | 2 of 24 | 1 of 2 |
| Fed decisions 2025 | 8 | 0.38 / 0.39 | 0.48 / 0.48 | 0.87 / 0.92 | 1 of 24 | 0 of 0 |
| Fed decisions 2026 | 6 | 0.88 / 0.83 | 1.64 / 1.64 | 1.01 / 1.15 | 3 of 18 | 1 of 2 |
| **Fed decisions 2023-2026** | 30 | 0.86 / 0.86 | 0.91 / 0.91 | 1.13 / 1.17 | 10 of 88 | 2 of 6 |
| Jobs reports 2023 | 12 | 1.16 / 1.15 | 0.77 / 0.82 | 1.54 / 1.62 | 7 of 34 | 6 of 12 |
| Jobs reports 2024 | 12 | 0.99 / 0.91 | 0.69 / 0.65 | 0.87 / 1.00 | 5 of 36 | 9 of 14 |
| Jobs reports 2025 | 11 | 1.56 / 1.59 | 0.66 / 0.69 | 1.14 / 1.22 | 2 of 32 | 4 of 5 |
| Jobs reports 2026 | 2 | 1.48 / 1.77 | 2.07 / 2.24 | 0.87 / 1.26 | 2 of 6 | 4 of 4 |
| **Jobs reports 2023-2026** | 37 | 1.24 / 1.24 | 0.78 / 0.80 | 1.16 / 1.27 | 16 of 108 | 23 of 35 |
| **All** | 67 | 1.07 / 1.06 | 0.84 / 0.85 | 1.15 / 1.23 | 26 of 196 | 25 of 41 |

**US stocks with people's trading.** kappa = 0.058, fitted on 2023-2025; 2026 is held out and was never used to choose it. Average miss in typical daily moves; "people" is the simulated holders' trading alone, "hybrid" adds it to the published estimates; major misses and directions are for the hybrid.

| Events | Count | Estimates only | People only | Hybrid | No change | Major misses | Direction right |
|---|---|---|---|---|---|---|---|
| Fed decisions 2023 | 8 | 0.88 | 0.93 | 0.88 | 0.93 | 1 of 8 | 3 of 4 |
| Fed decisions 2024 | 8 | 1.31 | 1.29 | 1.34 | 1.26 | 1 of 8 | 1 of 3 |
| Fed decisions 2025 | 8 | 0.38 | 0.43 | 0.41 | 0.39 | 0 of 8 | 1 of 2 |
| Fed decisions 2026 | 6 | 0.88 | 0.78 | 0.83 | 0.83 | 1 of 6 | 1 of 2 |
| **Fed decisions 2023-2026** | 30 | 0.86 | 0.86 | 0.87 | 0.86 | 3 of 30 | 6 of 11 |
| Jobs reports 2023 | 11 | 1.16 | 1.02 | 1.00 | 1.15 | 2 of 11 | 6 of 7 |
| Jobs reports 2024 | 12 | 0.99 | 0.83 | 0.86 | 0.91 | 1 of 12 | 5 of 6 |
| Jobs reports 2025 | 11 | 1.56 | 1.57 | 1.52 | 1.59 | 2 of 11 | 6 of 9 |
| Jobs reports 2026 | 2 | 1.48 | 2.24 | 1.95 | 1.77 | 1 of 2 | 0 of 0 |
| **Jobs reports 2023-2026** | 36 | 1.24 | 1.19 | 1.16 | 1.24 | 6 of 36 | 17 of 22 |
| **All** | 66 | 1.07 | 1.04 | 1.03 | 1.06 | 9 of 66 | 23 of 33 |

**Market moves on the day, 2026** (percent, close to close; Jahan / real). Misses are in typical daily moves
(one daily s.d.: US stocks 0.92%, gold 1.28%, dollar 0.41%). A major miss is more than 2 s.d., or the wrong direction on a move over 1 s.d.

| Date | Event | Surprise | US stocks | Gold | Dollar | Miss (s.d.): stocks, gold, dollar | Major miss | US stocks with people / real |
|---|---|---|---|---|---|---|---|---|
| 2026-01-28 | Fed holds at 3.50-3.75% | +0.7 bp | -0.03% / -0.01% | 0.00% / +4.44% | +0.02% / +0.24% | 0.0, 3.5, 0.5 | Gold | -0.03% / -0.01% |
| 2026-03-18 | Fed holds at 3.50-3.75% | +0.0 bp | 0.00% / -1.36% | 0.00% / -3.63% | 0.00% / +0.51% | 1.5, 2.8, 1.2 | Gold | 0.00% / -1.36% |
| 2026-04-29 | Fed holds at 3.50-3.75% | +0.0 bp | 0.00% / -0.04% | 0.00% / -1.02% | 0.00% / +0.30% | 0.0, 0.8, 0.7 | none | 0.00% / -0.04% |
| 2026-06-05 | May jobs +172k vs about 85k expected | +87k jobs | -0.25% / -2.64% | -0.20% / -3.36% | +0.15% / +0.66% | 2.6, 2.5, 1.3 | US stocks, Gold | +0.19% / -2.64% |
| 2026-06-17 | Fed holds at 3.50-3.75% (Warsh's first meeting; median 2026 dot 3.4 to 3.8) | +0.0 bp | 0.00% / -1.21% | 0.00% / -1.50% | 0.00% / +0.55% | 1.3, 1.2, 1.3 | none | 0.00% / -1.21% |
| 2026-07-29 | Fed holds at 3.50-3.75% (three officials wanted a hike) | -8.3 bp | +0.33% / -1.52% | 0.00% / +1.32% | -0.26% / -0.57% | 2.0, 1.0, 0.7 | US stocks | +0.33% / -1.52% |
| 2026-08-07 | July jobs -23k vs about +80k expected | -103k jobs | +0.29% / +0.62% | +0.24% / +2.37% | -0.17% / -0.37% | 0.4, 1.7, 0.5 | none | -0.14% / +0.62% |
| 2026-09-16 | Fed raises 0.25 pt to 3.75-4.00% (first hike since 2023) | +2.0 bp | -0.08% / -0.45% | 0.00% / -0.71% | +0.06% / +0.66% | 0.4, 0.6, 1.5 | none | -0.35% / -0.45% |

Summary over these events. US stocks: Jahan misses by 1.03 s.d. on average, "no change" by 1.07; direction called 3 times, right 2. Gold: Jahan misses by 1.75 s.d. on average, "no change" by 1.79; direction called 0 times, right 0. Dollar: Jahan misses by 0.97 s.d. on average, "no change" by 1.18; direction called 3 times, right 3.

**After the fact, not a forecast.** The day's 2-year Treasury yield move, minus the part the decision itself explains, used as the forward-guidance surprise; and the gold estimate Jahan does not apply (its sign is disputed).

| Date | Event | 2-year yield | US stocks with hindsight / real | Dollar with hindsight / real | Gold, unused estimate / real |
|---|---|---|---|---|---|
| 2026-01-28 | Fed holds at 3.50-3.75% | +3 bp | -0.11% / -0.01% | +0.14% / +0.24% | -0.04% / +4.44% |
| 2026-03-18 | Fed holds at 3.50-3.75% | +8 bp | -0.24% / -1.36% | +0.36% / +0.51% | 0.00% / -3.63% |
| 2026-04-29 | Fed holds at 3.50-3.75% | +8 bp | -0.24% / -0.04% | +0.36% / +0.30% | 0.00% / -1.02% |
| 2026-06-17 | Fed holds at 3.50-3.75% (Warsh's first meeting; median 2026 dot 3.4 to 3.8) | +15 bp | -0.46% / -1.21% | +0.68% / +0.55% | 0.00% / -1.50% |
| 2026-07-29 | Fed holds at 3.50-3.75% (three officials wanted a hike) | -4 bp | +0.33% / -1.52% | -0.27% / -0.57% | +0.43% / +1.32% |
| 2026-08-28 | Warsh's hawkish Jackson Hole speech (guidance only) | +14 bp | -0.43% / -0.25% | +0.64% / +0.54% | 0.00% / -3.32% |
| 2026-09-16 | Fed raises 0.25 pt to 3.75-4.00% (first hike since 2023) | +7 bp | -0.26% / -0.45% | +0.34% / +0.66% | -0.10% / -0.71% |

**Not modelled.** No published estimate exists, so Jahan moves nothing unless a prompt scripts the move.

| Date | Event | US stocks | Gold | Dollar |
|---|---|---|---|---|
| 2026-01-20 | Trump threatens tariffs on 8 European countries over Greenland | -2.06% | +3.55% | -0.75% |
| 2026-02-20 | Supreme Court strikes down the IEEPA tariffs | +0.69% | +2.16% | -0.13% |
| 2026-03-02 | First trading day after US and Israeli strikes on Iran | +0.04% | +1.07% | +0.79% |
| 2026-04-08 | Two-week US-Iran ceasefire; oil falls 13% | +2.51% | -0.43% | -0.51% |
| 2026-07-07 | US strikes on Iran; Trump says the truce is over | -0.45% | -1.53% | +0.29% |
| 2026-09-01 | Renewed US-Iran strikes; tankers hit | -0.71% | -2.69% | +0.24% |
| 2026-09-28 | Trump rejects Iran's Hormuz plan; Brent near $108 | -0.77% | -4.01% | +0.23% |

**News reach.** Share of US adults who know the headline "The Federal Reserve raises interest rates by a quarter point to 3.75-4.00%, its first hike since 2023": 4.9% by the end of day 0, 9.9% by the end of day 1, 14.8% by the end of day 2 (day 0 is the release day; mean of 5 seeds, day 2 range 14.0-15.9%). Survey band 10-41%: inside.

In surveys around 12 Fed press conferences (2015-2018), about 30% of US consumers had heard some news about the Fed in the week before and about 40% in the first two days after; the announcement itself added about 10 points. The share who heard this decision should lie between the added 10 points and the 41% total. Jahan tracks only this headline, with no earlier Fed news.

**Why the major misses happened**

In 2023-2025 (full list in [docs/backtests.md](docs/backtests.md)):

- **Stocks after jobs reports** rose on several strong reports, against the estimate's sign (above).
- **The dollar** moved more than the estimates allow on many Fed and jobs days: its typical daily move is
  small (0.41%), so a miss of 0.8 points already counts as major.
- **Other news:** the hawkish projections of 18 Dec 2024 (stocks -2.95%), and the tariff crash on 4 Apr
  2025 (stocks -5.97% on a jobs-report day).

In 2026:

- **28 Jan, gold +4.4%.** Gold was in the middle of a rally that had nothing to do with the decision.
  Jahan applies no effect of Fed decisions on gold, because published estimates disagree on its sign.
- **18 Mar.** The Iran war and oil prices moved gold (-3.6%) and stocks (-1.4%).
- **29 Jul, US stocks.** The Fed held when markets gave about a one-in-three chance of a hike, so the
  decision alone looked like good news and Jahan forecast +0.33%. In fact three officials dissented in
  favour of a hike, long-term yields jumped (the 30-year Treasury yield rose 11 bp to 5.20% on FRED, which
  market reports called its highest since 2007) and the S&P 500 fell 1.52% ([CNN](https://www.cnn.com/2026/07/29/business/live-news/federal-reserve-interest-rate-07-29-26),
  [Yahoo Finance](https://finance.yahoo.com/markets/live/stock-market-today-wednesday-july-29-dow-sp-500-nasdaq-082009165.html)).
  The surprise was in the vote and the guidance, which a forecast made the day before cannot know.
- **5 Jun jobs report.** Jahan got the direction right but only about a tenth of the size. The stock
  estimate comes from 5-minute windows in 1994-2001. In 2026, strong hiring meant higher rates for longer
  (the 2-year yield rose 12 bp), and it was the year's worst day for US stocks. No published whole-day
  estimate was found to replace it.

**What was changed, and what was not**

- **Market estimates: unchanged.** Changing them to fit these days would be fitting the test. Two
  findings are left for a later, separate decision:
  - **Jobs reports and stocks.** The stock response to jobs reports kept its 1994-2001 sign but was right
    only 8 times out of 15 in 2023-2026.
  - **Gold.** It moved opposite to the 2-year yield on 26 of 29 Fed days. That supports the sign of the gold
    estimate Jahan does not apply.
  - **How to decide fairly.** Either change should be judged only on events after October 2026, with the
    rule written down before they happen.
- **Public figures: one sourced fix.** In the first live run, the four US figures (Trump, Bessent, Warsh,
  Musk) each got a turn on 17 Sep and all chose to do nothing. The real Trump criticised the hike on Truth
  Social on 16 Sep, said rates should be 1% or lower, and did not attack Warsh by name.
  - **Cause.** Trump's roster entry had no documented position on interest rates, and figures act only on
    documented positions.
  - **Fix.** Two stances were added, each sourced from before the decision: Trump's 4 Sep threat to halt
    trade unless the Fed cut rates, and Warsh's 28 Aug Jackson Hole remarks that inflation was still too
    high.
  - **Second run.** "[simulated] The Federal Reserve should be cutting interest rates, not raising them,
    and I expect it to bring rates down," said simulated Trump. It reached 71 simulated people, who
    stand for about 21 million Americans. Simulated Warsh said inflation "is still too high" and called the
    increase "a measured step". The real Warsh said inflation "is too high and has been for too long".
    Bessent and Musk stayed silent.
  - **Caveats.** This is not a blind test: the stances were added after the first run showed the gap. It
    is one run of a language model (Claude Haiku 5.5 with its default sampling settings), whose answers
    vary. Figures decide at the start of the
    next day, so the reaction comes a day late. Both runs together cost about $0.0013.
- **A bug found by the backtest.** Daily metrics crashed in small worlds when the only people who knew a
  story were public figures, who carry no population weight. Fixed.

**What the after-the-fact table shows.** It takes the day's 2-year yield move as the forward-guidance
surprise, over 31 Fed days (the 30 decisions and the 2026 Jackson Hole speech; table for every year in
[docs/backtests.md](docs/backtests.md)). That move is known only after the release, so this is an analysis
of what the forecast missed, not a forecast:

- **The dollar** goes the right way on 25 of 28, within about a third of a point on 22.
- **US stocks** go the right way on 21 of 30, but the moves are too small.
- **Gold** moved opposite to the 2-year yield on 26 of 29. That favours the sign Jahan does not apply (Roache
  and Rossi 2009); switching it on now would be fitting the test.
- **Gold on Iran escalations.** From July on, gold fell on every escalation, as oil fed fears of higher
  rates, though it rose on the first trading day of the war in March. A fixed "gold is a safe haven" rule
  would have had the wrong sign most of the time.

To add events, put them in a `data/backtests/events_*.yaml` file with their sources; `jahan backtest` reads
every such file.

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
- the US gap between camps borrowed for figures without a breakdown, and halving it outside the US;
- the social feed weights (interest, trust in the author), each platform's default share where no
  figure is published, post reach from followers and engagement, and every repost coefficient;
- the salience of an announcement's headline (0.9), the 1% move that makes US stocks news, and the tone
  and weight of stock headlines in people's mood;
- in people's trading: that only holders with internet trade, the 10% cap on a day's move, trading on
  weekdays only (no market holidays), and trading on the day news is learned;
- in event odds: how far a story moves a trader (a story rated +1 multiplies their odds of Yes by about
  2.7), the spread of traders' starting beliefs (0.5 on the log-odds scale), the age and sex skew of who
  trades, the optional partisan lean, and the limit of 8 open questions (each holds a tracked-item slot).

Market reactions use estimates from 1990-2019 samples, some measured over 5 or 30 minutes rather than a
whole day; they apply the same sign every time, although about a third of Fed announcements move rates and
stocks in the same direction (Jarociński and Karadi 2020). A Fed chair's own decisions in the simulation
(`set_interest_rate`, by a player or by autonomy) still use the placeholder mood effect in `roles.yaml`;
they move US stocks only through people's trading on that mood shock and on the news of it, not through
the published estimates. People's trading covers US stocks only; its one
fitted number rests on release days, and how far it moves the index over weeks is a consequence of the
model, not a measurement.

Event odds ignore how much a trader believed the story that moved them, and how stories are rated is
Claude's judgement (each rating, with a short reason, is logged and shown on the Watch page). Word of
mouth lets odds drift slowly in a quiet world: about 1.5 points over two months in a test.

Event logs written before 15 October 2026 used one number (28) for both reposts and headline updates;
reports still read them, and new logs use 30 for updates.

Worlds created before political camps existed (snapshot schema 5) cannot be opened; create a new world.

Each distinct story, message or post occupies one tracked-item slot: news for about a week after it stops
being broadcast, a public figure's statement for two months. When the `max_slots` pool (128 by default)
is full, the oldest item that is no longer being broadcast gives up its slot, and what people believed
about it is forgotten (logged as `ITEM_EVICTED`). Products keep their slots. A new item is skipped only
when every tracked item is still on air. At 10,000 agents each slot costs about 0.4 MB of memory.

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

Apache-2.0; the full text is in [LICENSE](LICENSE). Sections 7 and 8 disclaim all warranties and limit
liability; see also the [Disclaimer](#disclaimer).
