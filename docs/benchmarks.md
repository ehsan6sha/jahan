# Everyday activity benchmarks

Simulated, not a forecast; educational and testing use only. Each run follows rules committed before it:
`data/backtests/benchmarks.yaml` and `data/global/investors.yaml`.

## Version 4: other ways of rating the news, on days nobody had looked at (2026-11-08)

**Why.** After B9, the diagnosis of its results on saved data, one test at a time:

| Tested | Result | Conclusion |
|---|---|---|
| Force the right direction on the fit days, keep the rated size | miss 0.795 falls to 0.621, s rises from 1.43 to 3.94 | direction is the main loss |
| Direction on rated days | fit 30 of 50, seen 5 of 8, hold-out 4 of 7 | close to chance on ordinary news |
| Drop the most influential fit days | s = 1.42, 0.89, 0.89 and 0.00 after dropping 1, 2, 3 and 5 | five famous days carried the fit |
| Does the rater remember? (asked how the S&P 500 closed on each date) | 2023-2025: direction right 84%, 100% on days of 1% or more, correlation of claimed and real move 0.96. From July 2026: never claims to remember. But on the fit days its ratings agree with what it recalls only 54% of the time | it remembers those years, yet its ratings do not replay that memory |
| Reaction shape (powers 1 to 3, thresholds 0 to 0.5), chosen on 2023-2025 only | best 0.756 against 0.797 for a straight line; none beats "no change" from July 2026 | not the cause |
| Recent market state (prior 1, 5 and 20-day moves, drawdown, volatility) | correlations with the day's move between -0.12 and +0.15; scaling by volatility is worse later | not the cause |
| Size limit | broad story at most s x 1 = 1.43%, tech story at most 0.379 x 1.43 = 0.54% of the S&P 500 | a day like -4.36% could not be reached by news even with a perfect rating |

**What was registered first** (commit 4dd1b14, before any call or data for it): B11 in
`data/backtests/benchmarks.yaml`: four new ways of rating (material: only what changes profits, rates or credit
for the whole market; context: the previous 30 days' headlines; context and market: also the prior market moves;
material and context), one chosen on 2023 to June 2026 days only, then scored on days after the rater's training
data, with the decision written down in advance.

**Data.** 49 fresh days (`data/backtests/news_days_extra.yaml`, commit 8c2d416), every further qualifying event of
July to 6 November 2026 from the same Wikipedia revision, collected and written as neutral headlines without
looking at markets, plus B9's 8 hold-out days: 57 clean days. One further collected day (9 November) has no close
yet. Some fresh days coincide with a Fed decision or a jobs report, as some B9 days do; the rule excludes those
events, not the days (16 September is a Fed day, 28 October possibly; 7 August, 4 September, 2 October and 6 November are
probably jobs days). Of the 99 headlines, 42 have no known publication time (counted as public before the close, as the rule
says) and 34 rest on a report the collector could not open. The selection counts every US sanctions action, so many days carry minor news. Ratings cost
$9.01 in all (75 earlier days with four variants $4.58; the clean days $4.43).

| Choosing one, on 75 days from 2023 to June 2026 (leave-one-period-out) | Miss | Direction on rated days |
|---|---|---|
| Current rater | 0.762 | 35 of 58 |
| Material | **0.744** (chosen) | 19 of 28 |
| Context | 0.768 | 36 of 62 |
| Context and market | 0.757 | 35 of 60 |
| Material and context | 0.751 | 17 of 28 |

| 57 clean days | All | Fresh 49 | Hold-out 8 |
|---|---|---|---|
| Current engine (s = 1.43) | 0.710 | 0.655 | 1.050 |
| Without other investors (s = 0) | **0.586** | 0.558 | 0.754 |
| No change | 0.603 | 0.571 | 0.800 |
| Material rater, s refitted to 3.67 (95% interval 0.73 to 8.43) | 0.766 | 0.722 | 1.032 |

The current engine is worse than s = 0 by 0.125 (95% interval 0.048 to 0.202; resampling whole weeks, because
stories cluster, 0.056 to 0.186; on the fresh days alone 0.097, 0.024 to 0.163). The people's part alone is not
distinguishable from no change (difference -0.017, interval -0.071 to 0.033), so that part is a default, not a
demonstrated skill. Only s = 1.43 and s = 0 were decided between; refitting s on resampled clean days gives a median
of 0 with a 95% interval of 0 to 0.34. Ordinary-day movement is off in all of these scores (three seeds averaged). On
the 9 days of 1% or more: 1.509 with s = 1.43, 1.454 with s = 0, 1.441 for no change. Direction on rated clean days: 26 of 50
for the current rater, 19 of 32 for the material one; on the 9 days of 1% or more the current rater was right on 2 and
wrong on 6. The largest move among the 57 days was only 1.66%, so these days say nothing about crashes.

Exploratory, not used to decide: rating with recent headlines scored 0.638 (context, s = 1.66) and 0.606 (material and
context, s = 1.58): better than the current engine but not better than s = 0, and direction stayed 28 of 54 and 23 of 39.
The placebo was not run, because the chosen rater used no context. On the clean days themselves the best s for the
current rater is 0. One lead for later: when the material rater gave a strong rating (|rating x salience| of 0.3 or more)
it was right on 6 of 6 days (real moves averaging 0.92%); the current rater was right on 9 of 12. Six days are too few, and
this was found on these days, so it needs new days to test.

**Decision (registered).** 57 clean days is past the 20 required. The material rater did not beat s = 0, so it is not
adopted; s = 0 beats s = 1.43, so **other investors' news reaction now defaults to s = 0**. Ordinary-day movement stays.
Users can turn the reaction back on with `market_others_s`. Worlds already at version 2 keep the s they stored.

**What this does and does not show.** On these days (about four months, one calm stretch, largest move 1.66%), news from a
public chronology, one headline at a time, said almost nothing about which way the S&P 500 went that day. The 57 days
contain no shock, so they do not test crashes. Against the current engine the gain is 17%; against always predicting no
change it is about 3%, within noise. The trade-off: without other investors, typed big shocks look small (on 7 days
picked because markets moved, three of them before the rating model's training cutoff: 1.743 against 1.505 with them; days
picked for their size favour any reaction, and this comparison was never used to decide). The midterm week is still missed. The missing channels (oil,
interest rates, company results, sectors) are untested, and these 57 days are now used up: a new test needs new days.

Reproduce: `uv run python -X utf8 scripts/rater_variants.py choose` and `clean` (replays the recordings in
`runs/_benchmark`, git-ignored; `--rate` calls Claude for anything not recorded).

## Version 3: unscheduled news days and ordinary days (2026-11-08)

**Why.** Replaying the midterm week of November 2026 (real S&P 500 +1.58%, -4.36%, +0.3%; Jahan 0.00%, -0.15%,
+0.01 to +0.16%), the causes were traced, one factor at a time: the market heard news only through the few
simulated households who heard a story on its first day; Claude's ratings were not
defined for markets (valence for ordinary people; salience undefined although it enters the chance of
hearing a story cubed; topic weights unsourced); three headlines about one event added up; and kappa had been
fitted on hand-set ratings. Even three maximal stories reached only -2.3%.

**What was registered first** (commit 4886978, before any data was collected): B9, news days chosen by the
news (the first two qualifying events per month in Wikipedia's "<year> in the United States"), fit 2023-2025,
seen January-June 2026, hold-out from July 2026 (after the parser model's training data, June 2026), verdict
once 20 hold-out days exist; B10, ordinary days against real days; rule_v4, other investors' immediate,
permanent reaction with one fitted number s (kappa held at 0.048), s = 0 if no positive s beats it.

**What changed** (version 2 worlds): one price per event (headlines sharing a story key are priced as the
strongest of them); Claude rates each story's market_valence against what investors expected, its sector
(broad or tech) and salience against stated anchors; other investors react by s x market_valence x salience on
release (tech stories move the S&P 500 by information technology's 37.9% weight); ordinary days move US stocks,
gold and the euro with a GARCH(1,1)-t model fitted to 688 real days.

**Data.** 90 trading days with 104 headlines: 64 fit, 11 seen, 8 hold-out, 7 picked because markets moved
(scored, never fitted); `data/backtests/news_days.yaml`, each headline timed against a contemporaneous report.
Claude (claude-sonnet-5-5) rated every day twice: the two market ratings correlate 0.994 (same sign on 98.9%
of days). Cost of the ratings: $2.69.

| Check | Result | Passed |
|---|---|---|
| B9 fit (64 days, 2023-2025) | s = 1.43 (95% interval 0 to 4.78): release-day miss 0.7955 with other investors, 0.8341 without, 0.8384 for no change; on the 3 days of 2% or more, direction right 3 of 3, within a factor of 2 none | yes |
| B9 seen (11 days, Jan-Jun 2026) | 0.5145 with, 0.4764 without, 0.4509 no change | worse |
| B9 hold-out (8 days from July 2026, provisional) | 1.0499 with, 0.7537 without, 0.7998 no change | worse so far |
| B9 picked because markets moved (7 days) | 1.5045 with, 1.7431 without, 1.9650 no change; direction right 3 of 3, within a factor of 2 one (Greenland tariffs: -1.12% against -2.06%; 8 April ceasefire +1.08% against +2.51%; 5 November -0.19% against -4.36%) | reported |
| B9 tech line (fit days) | 1.0804 against 1.1201 for no change | reported |
| B9 the next 4 days after 1% days (fit) | real mean signed move +0.04%: no reversal to speak of, as the permanent reaction assumes | reported |
| B2 with the rated Iran example | largest day -1.05% (real -1.74%), 20 days -1.06% (real -7.45%) | yes |
| B10 ordinary days | US stocks: mean daily 0.66% vs 0.63%, 95th percentile 1.75% vs 1.68%, median 21-day 2.65% vs 3.22%; gold 0.82% vs 0.78%, 2.17% vs 2.26%, 2.92% vs 2.81%; euro 0.35% vs 0.34%, 0.94% vs 0.91%, 1.25% vs 1.38% | yes (each within 20%) |

**Reading it.** Other investors help a little on 2023-2025 and on the days picked because markets moved, get
the direction of every big day right, but stay far too small on big days, and on 2026 they make things worse.
The reason is in the ratings: Claude rated almost every Iran-war escalation of 2026 as bad for company profits,
while markets, which had priced the war since March, often rose (3 November strikes: simulated -0.86%, real
+0.36%). Each day is rated without the run-up, so "against what investors expected" cannot see an expectation
built by earlier news. The midterm result is rated neutral for markets and the 5 November data-center story a
moderate tech story; replaying the week with version 2 gives about 0.0%, -0.2% and +0.1%. The hold-out has 8
days; the registered verdict waits for 20.

Record (also in `data/global/investors.yaml`, `other_investors.fitted_v4`):

```yaml
fitted_v4:
  date: '2026-11-08'
  command: jahan benchmark --only B9 --tune --rate
  rule: data/global/investors.yaml other_investors.rule_v4 (registered 2026-11-08, before any B9 data)
  config: 10000 people, USA and DEU, everyday activity version 2, saving and ordinary days off, kappa 0.048 fixed,
    seeds [0, 1, 2]
  parser:
    model: claude-sonnet-5-5
    prompt_hash: 2efb3e1a29bea946
  fit_days: 64
  grid: 0 to 20, step 0.01
  s_optimum: 1.43
  boundary_solution: false
  s_ci95:
  - 0.0
  - 4.776999999999961
  substituted_for_b2: false
  s: 1.43
  constraints:
    b2:
      iran_example_ratings:
      - headline: Iran-US confrontation escalates
        market_valence: -0.4
        sectors: broad
      real:
        from: '2026-02-27'
        to: '2026-03-27'
        largest_daily_fall_pct: -1.7406
        largest_20_day_fall_pct: -7.4513
      runs:
        '1.43':
          s: 1.43
          sim:
            largest_daily_fall_pct: -1.0472229935514998
            largest_20_day_fall_pct: -1.06022297497822
            end_index: 99.4831579043887
          passed: true
    stability: a scripted -3% day has no rated story, so other investors cannot react (unchanged from trading.kappa.fitted_v3)
    b1: announcement headlines get no other-investor reaction, so B1 cannot change with s
  loss_at:
    '0': 0.8341
    '0.5': 0.8174
    '1': 0.8026
    '2': 0.8001
    '3': 0.8114
    '5': 0.8849
    '10': 1.3416
    '20': 2.3905
  fit_scores:
    days: 64
    day0: 0.7955
    day0_s0: 0.8341
    day0_no_change: 0.8384
    big_days: 3
    big_direction_right: 3
    big_within_factor_2: 0
    big_days_detail:
    - day: '2024-09-03'
      real: -2.12
      sim: -0.0
    - day: '2024-11-06'
      real: 2.53
      sim: 0.71
    - day: '2025-04-03'
      real: -4.84
      sim: -1.63
    tech_day0: 1.0804
    tech_day0_no_change: 1.1201
    next_4_days_after_1pct_days:
      days: 18
      mean_signed_move_pct: 0.041
  seen_scores:
    days: 11
    day0: 0.5145
    day0_s0: 0.4764
    day0_no_change: 0.4509
    big_days: 0
    tech_day0: 0.7963
    tech_day0_no_change: 0.6434
  outcome_scores:
    days: 7
    day0: 1.5045
    day0_s0: 1.7431
    day0_no_change: 1.965
    big_days: 3
    big_direction_right: 3
    big_within_factor_2: 1
    big_days_detail:
    - day: '2026-01-20'
      real: -2.06
      sim: -1.12
    - day: '2026-04-08'
      real: 2.51
      sim: 1.08
    - day: '2026-11-05'
      real: -4.36
      sim: -0.19
    tech_day0: 1.9099
    tech_day0_no_change: 2.8808
    next_4_days_after_1pct_days:
      days: 4
      mean_signed_move_pct: 0.844
  holdout_scores:
    days: 8
    day0: 1.0499
    day0_s0: 0.7537
    day0_no_change: 0.7998
    big_days: 0
    tech_day0: 1.3238
    tech_day0_no_change: 1.0793
    next_4_days_after_1pct_days:
      days: 3
      mean_signed_move_pct: -1.44
  ratings_retest:
    days_rated_both_times: 90
    rated_only_once: 0
    market_valence_correlation: 0.994
    mean_abs_difference: 0.01
    same_sign_share: 0.989
```

## Version 2 run (2026-10-17): how we got here

1. **Rule v2 (late reactions moved prices) chose kappa = 0.** Every positive kappa made the release day plus
   the next 4 trading days worse (2023-2025: 0.8405 at 0, 0.8486 at 0.05, 0.8695 at 0.1), so the registered
   outcome was "people's news reactions off".
2. **Diagnosis.** 87% of the simulated reaction fell on the four days after each release: people check their
   accounts late, most of them hear the news a day or more later, and Friday jobs news lands on Monday. The
   real S&P 500's moves on those days are unrelated to the news (correlations -0.18 to 0.20); the real index
   had reacted on the release day. On release days the people's direction did carry information (jobs
   reports: correlation 0.25 with what the published estimate missed).
3. **Fix (rule v3, registered before running it).** A story is priced when it first reaches a market's
   holders and reversed when it stops being tracked; later learners and late account checks trade demand
   already priced, as with steady saving. Late reactions still shape each person's holdings, not prices.
4. **Rule v3 result (below).** kappa = 0.048. Better than no people on 2023-2025, but the 95% interval of the
   difference includes zero, and on the 8 events of 2026 (seen before, exploratory) it is worse: both 2026
   jobs reports went the opposite way from the simulated people's reading (a strong report and stocks
   -2.6%; a weak one and +0.6%), the "good news is bad news" pattern of rate fears. Nothing sourced says in
   advance when that pattern holds, so it is reported, not modelled. The clean test is the forecast events
   from 2026-10-17 on, once 12 exist.

Other results to read with care: B4's login check fails (simulated US holders log in on 97 days a year on
average against 85.2 published: the interpolated spread itself implies 90.4, the rest is sampling of 512
simulated holders; since rule v3 this affects only each holder's displayed position, not prices), and B6 shows that job losses alone barely move the national mood (about 150 times less
than real sentiment moves with unemployment, which also reacts to news about jobs that a world without news
lacks). Both are reported, not tuned.

Run 2026-10-17 (`jahan benchmark --tune`); checks, limits and splits registered in `data/backtests/benchmarks.yaml` (commit adbc9a988936fd52c00edf02e609e0b50a0310d9) before any run. Simulated, not a forecast.

| Check | Result | Passed |
|---|---|---|
| B1 | kappa 0.048: 5-day miss 0.8276 with people vs 0.8405 without (no change 0.8395); day 0 1.0176 vs 1.0745 | yes |
| B2 | Iran example: largest day -0.54% (real -1.74%), 20 days -0.54% (real -7.45%) | yes |
| B3 | news part flat: True / True; yearly saving wander (sd) 0.311% at 3,000, 0.008% at 10,000 | yes |
| B4 | ownership within 5 points everywhere; days logged in: median 35.2 (36), mean 97.3 (85.2) | NO |
| B5 | 79 fit questions: time passing misses by 8.28 points, the flat price by 10.66 | yes |
| B6 | per point of unemployment: simulated mood -0.021% of its scale, real sentiment -3.29% of its range (exploratory) | reported only |
| B7 | No benchmark exists outside the US; every non-US size is an untested illustration using borrowed kappa and multiplier. | stated |
| B8 | Trading volume is not modelled: Jahan models each holder's whole stock position, not stock picking. | stated |

## Details

```yaml
B1:
  summary: 'kappa 0.048: 5-day miss 0.8276 with people vs 0.8405 without (no change
    0.8395); day 0 1.0176 vs 1.0745'
  passed: true
  fit_2023_2025: &id001
    events: 58.0
    five_day: 0.8276
    five_day_without_people: 0.8405
    five_day_no_change: 0.8395
    day0: 1.0176
    day0_without_people: 1.0745
    day0_no_change: 1.063
    five_day_diff_ci95:
    - -0.033048049535017825
    - 0.007462048275262097
  seen_2026_exploratory: &id002
    events: 8.0
    five_day: 0.8859
    five_day_without_people: 0.8693
    five_day_no_change: 0.8769
    day0: 1.1115
    day0_without_people: 1.0282
    day0_no_change: 1.0666
    five_day_diff_ci95:
    - -0.014224685856488006
    - 0.047599507505219775
  fit_record:
    date: '2026-10-17'
    command: jahan benchmark --tune
    rule: data/global/investors.yaml trading.kappa.rule_v3 (registered 2026-10-17,
      after the rule v2 run)
    registered_in: adbc9a988936fd52c00edf02e609e0b50a0310d9
    config: m1, 10000 people, everyday activity, saving off
    fit_events: 58
    windows_cut: 12
    grid: 0 to 2, step 0.001
    kappa_optimum: 0.048
    boundary_solution: false
    per_seed_optimum:
      '0': 0.058
      '1': 0.037
      '2': 0.053
    kappa: 0.048
    constraints:
      '0.048':
        day0: true
        b2:
          kappa: 0.048
          sim:
            largest_daily_fall_pct: -0.5351781039954151
            largest_20_day_fall_pct: -0.5351781039954151
            end_index: 99.99799792697
          real:
            from: '2026-02-27'
            to: '2026-03-27'
            largest_daily_fall_pct: -1.7406
            largest_20_day_fall_pct: -7.4513
          passed: true
        stability:
          kappa: 0.048
          people_index_min: 99.989
          people_index_max: 100.0
          people_index_day_60: 100.0
          bounded: true
          reverted: true
          passed: true
    shadow_vs_full_five_day:
      shadow: 0.8282
      full: 0.8282
    refinement: null
    fit_scores: *id001
    seen_2026_scores: *id002
    holdout: 0 forecast events from 2026-10-17; scored once 12 exist
    loss_at:
      '0': 0.8405
      '0.01': 0.8361
      '0.05': 0.8278
      '0.1': 0.8397
      '0.2': 0.9181
      '0.5': 1.3593
      '1': 2.1129
      '2': 3.6201
B2:
  summary: 'Iran example: largest day -0.54% (real -1.74%), 20 days -0.54% (real -7.45%)'
  passed: true
  iran:
    kappa: 0.048
    sim:
      largest_daily_fall_pct: -0.5351781039954151
      largest_20_day_fall_pct: -0.5351781039954151
      end_index: 99.99799792697
    real:
      from: '2026-02-27'
      to: '2026-03-27'
      largest_daily_fall_pct: -1.7406
      largest_20_day_fall_pct: -7.4513
    passed: true
  stability:
    kappa: 0.048
    people_index_min: 99.989
    people_index_max: 100.0
    people_index_day_60: 100.0
    bounded: true
    reverted: true
    passed: true
B3:
  summary: 'news part flat: True / True; yearly saving wander (sd) 0.311% at 3,000,
    0.008% at 10,000'
  sizes:
    '3000':
      news_part_flat: true
      saving_mean_step_pct: -7.143605018918245e-05
      saving_mean_step_se: 0.0002782942193354806
      saving_no_trend: true
      saving_largest_step_pct: 0.09214727232400007
      saving_year_end_pct:
      - -0.3821
      - 0.1884
      - 0.116
      saving_year_wander_sd_pct: 0.3106288557257872
      moods_steady: true
    '10000':
      news_part_flat: true
      saving_mean_step_pct: 0.00012691678763577897
      saving_mean_step_se: 0.00013789320553708363
      saving_no_trend: true
      saving_largest_step_pct: 0.03047480551018532
      saving_year_end_pct:
      - 0.0384
      - 0.0467
      - 0.0538
      saving_year_wander_sd_pct: 0.0077393061901286295
      moods_steady: true
  passed: true
B4:
  summary: 'ownership within 5 points everywhere; days logged in: median 35.2 (36),
    mean 97.3 (85.2)'
  ownership:
    USA:
      simulated: 56.26
      source: 57.7
      holders: 512
      passed: true
      gallup_2025: 62.0
    BRA:
      simulated: 1.9
      source: 2.0
      holders: 13
      passed: true
    DEU:
      simulated: 20.65
      source: 19.9
      holders: 95
      passed: true
    NGA:
      simulated: 3.73
      source: 2.33
      holders: 19
      passed: true
    IRN:
      simulated: 4.42
      source: 4.59
      holders: 20
      passed: true
    IND:
      simulated: 8.65
      source: 9.5
      holders: 146
      passed: true
    CHN:
      simulated: 21.06
      source: 21.7
      holders: 395
      passed: true
    IDN:
      simulated: 5.56
      source: 4.95
      holders: 44
      passed: true
  logins:
    median_days: 35.2
    mean_days: 97.3
    spread_mean_days: 90.4
    holders: 512
    published_median: 36
    published_mean: 85.2
    passed: false
  passed: false
B5:
  summary: '79 fit questions: time passing misses by 8.28 points, the flat price by
    10.66'
  time_passing:
    fit:
      by_date:
        questions: 79
        sim_points: 8.28
        flat_points: 10.66
        diff_ci95_points:
        - -3.42
        - -1.44
        no_worse: true
      all:
        questions: 79
        sim_points: 8.28
        flat_points: 10.66
        diff_ci95_points:
        - -3.42
        - -1.44
        no_worse: true
    holdout:
      by_date:
        questions: 199
        sim_points: 12.11
        flat_points: 15.53
        diff_ci95_points:
        - -4.2
        - -2.69
        no_worse: true
      holds_through:
        questions: 7
        sim_points: 10.69
        flat_points: 20.11
        diff_ci95_points:
        - -13.22
        - -3.91
        no_worse: true
      all:
        questions: 206
        sim_points: 12.06
        flat_points: 15.69
        diff_ci95_points:
        - -4.44
        - -2.89
        no_worse: true
  with_talk:
    fit:
      by_date:
        questions: 79
        sim_points: 8.26
        flat_points: 10.66
        diff_ci95_points:
        - -3.43
        - -1.44
        no_worse: true
      all:
        questions: 79
        sim_points: 8.26
        flat_points: 10.66
        diff_ci95_points:
        - -3.43
        - -1.44
        no_worse: true
    holdout:
      by_date:
        questions: 199
        sim_points: 12.09
        flat_points: 15.53
        diff_ci95_points:
        - -4.21
        - -2.7
        no_worse: true
      holds_through:
        questions: 7
        sim_points: 10.67
        flat_points: 20.11
        diff_ci95_points:
        - -13.26
        - -3.89
        no_worse: true
      all:
        questions: 206
        sim_points: 12.04
        flat_points: 15.69
        diff_ci95_points:
        - -4.45
        - -2.91
        no_worse: true
  questions: 303
  passed: true
  rows:
  - slug: us-x-iran-ceasefire-continues-through-october-31-20260917
    kind: holds_through
    end: '2026-10-31'
    days: 29
    sim: 0.05200512133541883
    flat: 0.1368965517241379
  - slug: israel-x-iran-ceasefire-continues-through-august-15-20260716224448969-246-815-987-693
    kind: holds_through
    end: '2026-08-15'
    days: 25
    sim: 0.09956867879855734
    flat: 0.22332000000000002
  - slug: israel-x-iran-ceasefire-continues-through-august-31-20260716224448970-754-896-823
    kind: holds_through
    end: '2026-08-31'
    days: 37
    sim: 0.12822354250767495
    flat: 0.26809459459459456
  - slug: israel-x-iran-ceasefire-continues-through-september-30
    kind: holds_through
    end: '2026-09-30'
    days: 50
    sim: 0.06601100487891261
    flat: 0.20919000000000001
  - slug: israel-x-iran-ceasefire-continues-through-october-31
    kind: holds_through
    end: '2026-10-31'
    days: 60
    sim: 0.12067776201789501
    flat: 0.2577499999999999
  - slug: israel-x-iran-ceasefire-continues-through-december-31-415
    kind: holds_through
    end: '2026-12-31'
    days: 60
    sim: 0.16724877626820228
    flat: 0.25066666666666665
  - slug: israel-x-iran-ceasefire-continues-through-november-30
    kind: holds_through
    end: '2026-11-30'
    days: 51
    sim: 0.11425109637083314
    flat: 0.06196078431372548
  - slug: us-iran-hormuz-agreement-by-september-15
    kind: by_date
    end: '2026-09-15'
    days: 14
    sim: 0.08119529194405128
    flat: 0.12142857142857143
  - slug: us-iran-hormuz-agreement-by-september-30
    kind: by_date
    end: '2026-09-30'
    days: 21
    sim: 0.19616847961043046
    flat: 0.26857142857142857
  - slug: will-the-us-invade-iran-before-2027
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.04353042357110589
    flat: 0.04525
  - slug: iran-leadership-change-by-september-30-20260629144005163-739-824
    kind: by_date
    end: '2026-09-30'
    days: 60
    sim: 0.02605518617674137
    flat: 0.03265
  - slug: iran-leadership-change-by-july-31-20260629143932628
    kind: by_date
    end: '2026-07-31'
    days: 1
    sim: 0.018567356607223154
    flat: 0.018499999999999996
  - slug: iran-leadership-change-by-august-31-669-893
    kind: by_date
    end: '2026-08-31'
    days: 10
    sim: 0.023575892498623175
    flat: 0.030800000000000004
  - slug: iran-leadership-change-by-october-31
    kind: by_date
    end: '2026-10-31'
    days: 48
    sim: 0.03968672846015164
    flat: 0.07235416666666668
  - slug: iran-leadership-change-by-november-30
    kind: by_date
    end: '2026-11-30'
    days: 60
    sim: 0.048737063282902865
    flat: 0.08491666666666668
  - slug: iran-leadership-change-by-april-30
    kind: by_date
    end: '2026-04-30'
    days: 42
    sim: 0.08543701076821628
    flat: 0.24479761904761904
  - slug: iran-leadership-change-by-december-31-974-976-658-482-568-685-996-469-827
    kind: by_date
    end: '2026-12-31'
    days: 38
    sim: 0.14765295927924807
    flat: 0.18631578947368424
  - slug: iran-leadership-change-by-may-31-593-194-829
    kind: by_date
    end: '2026-05-31'
    days: 50
    sim: 0.03316947039632811
    flat: 0.12963
  - slug: iran-leadership-change-by-june-30-689-922
    kind: by_date
    end: '2026-06-30'
    days: 41
    sim: 0.04335359198873904
    flat: 0.13475609756097562
  - slug: iran-full-airspace-closure-by-july-31-20260625195254150-838
    kind: by_date
    end: '2026-07-31'
    days: 34
    sim: 0.14762249988304424
    flat: 0.2749852941176471
  - slug: iran-full-airspace-closure-by-august-31-20260625195254151-668
    kind: by_date
    end: '2026-08-31'
    days: 58
    sim: 0.10972436704639606
    flat: 0.2427327586206897
  - slug: iran-full-airspace-closure-by-december-31-20260727170429885
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.22427918181280646
    flat: 0.2977966101694915
  - slug: iran-full-airspace-closure-by-october-31-2026-copy
    kind: by_date
    end: '2026-10-31'
    days: 30
    sim: 0.017778626353951922
    flat: 0.052999999999999985
  - slug: will-there-be-no-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 20
    sim: 0.01593526110260274
    flat: 0.016550000000000002
  - slug: will-muhammad-mirbaqiri-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.16098024057057694
    flat: 0.161
  - slug: will-sadegh-larijani-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.05991948695744667
    flat: 0.06
  - slug: will-mojtaba-khamenei-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 37
    sim: 0.28649866516238254
    flat: 0.2807702702702703
  - slug: will-hassan-khomeini-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.0001808302709203069
    flat: 0.0
  - slug: will-reza-pahlavi-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 52
    sim: 0.09352524412164884
    flat: 0.1096153846153846
  - slug: will-maryam-rajavi-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.059761643473531016
    flat: 0.060000000000000005
  - slug: will-massoud-rajavi-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.020099371197505572
    flat: 0.020000000000000004
  - slug: will-seyed-hossein-mousavian-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.014981408956960512
    flat: 0.015
  - slug: will-reza-pirzadeh-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.05998888535435262
    flat: 0.06
  - slug: will-mustafa-hijri-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.03490106627967348
    flat: 0.034999999999999996
  - slug: will-sadegh-mahsouli-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.04439669604414299
    flat: 0.044500000000000005
  - slug: will-masoud-pezeshkian-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 13
    sim: 0.06589596572942143
    flat: 0.06846153846153845
  - slug: will-hassan-rouhani-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.0049701009618027675
    flat: 0.0050000000000000044
  - slug: will-mohammad-khatami-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.004867800458384863
    flat: 0.0050000000000000044
  - slug: will-alireza-arafi-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 6
    sim: 0.012735065735006976
    flat: 0.01233333333333333
  - slug: will-ali-asghar-hejazi-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 3
    sim: 0.02192384377726987
    flat: 0.022000000000000002
  - slug: will-mohammad-bagher-ghalibaf-be-head-of-state-in-iran-end-of-2026
    kind: by_date
    end: '2026-12-31'
    days: 19
    sim: 0.015845388540529765
    flat: 0.012078947368421052
  - slug: us-iran-final-nuclear-deal-by-july-31-2026-20260621201255151
    kind: by_date
    end: '2026-07-31'
    days: 8
    sim: 0.0418257944585324
    flat: 0.048124999999999994
  - slug: us-iran-final-nuclear-deal-by-august-13-2026-20260621201255152
    kind: by_date
    end: '2026-08-13'
    days: 15
    sim: 0.04180934591799469
    flat: 0.060033333333333334
  - slug: us-iran-final-nuclear-deal-by-august-18-2026-20260621201255153
    kind: by_date
    end: '2026-08-18'
    days: 37
    sim: 0.054735506984172434
    flat: 0.11859459459459457
  - slug: us-iran-final-nuclear-deal-by-august-31-2026-20260621201255154
    kind: by_date
    end: '2026-08-31'
    days: 39
    sim: 0.09148748717169793
    flat: 0.1613076923076923
  - slug: us-iran-final-nuclear-deal-by-september-30-2026
    kind: by_date
    end: '2026-09-30'
    days: 49
    sim: 0.09151182430908418
    flat: 0.1583673469387755
  - slug: us-iran-final-nuclear-deal-by-december-31-2026-191
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.11717336175317523
    flat: 0.16872881355932204
  - slug: us-iran-final-nuclear-deal-by-november-30-2026
    kind: by_date
    end: '2026-11-30'
    days: 60
    sim: 0.06333632078132481
    flat: 0.0993333333333333
  - slug: us-iran-final-nuclear-deal-by-october-31-2026
    kind: by_date
    end: '2026-10-31'
    days: 17
    sim: 0.036448354175319386
    flat: 0.041999999999999996
  - slug: will-reza-pahlavi-lead-iran-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 38
    sim: 0.04305723095563759
    flat: 0.05169736842105264
  - slug: iran-oman-hormuz-agreement-by-september-30-823-929
    kind: by_date
    end: '2026-09-30'
    days: 46
    sim: 0.17218886555987184
    flat: 0.40434782608695646
  - slug: iran-oman-hormuz-agreement-by-october-31
    kind: by_date
    end: '2026-10-31'
    days: 49
    sim: 0.13963113646679434
    flat: 0.3092857142857143
  - slug: will-iran-withdraw-from-the-npt-before-2027
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.07750970114487475
    flat: 0.07625
  - slug: will-iran-target-jordan-by-october-31-2026-20260909
    kind: by_date
    end: '2026-10-31'
    days: 37
    sim: 0.30977753601573776
    flat: 0.45687837837837836
  - slug: iran-nuke-before-2027
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.08238935461865085
    flat: 0.09424999999999997
  - slug: will-a-1-year-enrichment-moratorium-be-in-a-us-iran-deal-in-2026-20260623023201635
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.053303088926087774
    flat: 0.05872881355932203
  - slug: will-any-uranium-enrichment-cap-1-year-be-in-a-us-iran-deal-in-2026-20260623023201636
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.19850755015685678
    flat: 0.25391666666666673
  - slug: will-a-5-uranium-enrichment-cap-1-year-be-in-a-us-iran-deal-in-2026-20260623023201637
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.07714514907573129
    flat: 0.12225
  - slug: will-iran-reconstruction-funding-be-in-a-us-iran-deal-in-2026-20260623023201638
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.33363446927479506
    flat: 0.39652542372881355
  - slug: will-dilution-of-iranian-uranium-be-in-a-us-iran-deal-in-2026-20260623023201639
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.3281525357350493
    flat: 0.3898305084745763
  - slug: will-a-us-iran-deal-in-2026-include-surrender-of-iranian-enriched-uranium-20260623023201640
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.09280241724784458
    flat: 0.12347457627118644
  - slug: iran-charges-hormuz-fees-by-august-31-20260625175039062-177
    kind: by_date
    end: '2026-08-31'
    days: 59
    sim: 0.06668911480098424
    flat: 0.22418644067796611
  - slug: iran-charges-hormuz-fees-by-july-31
    kind: by_date
    end: '2026-07-31'
    days: 25
    sim: 0.06916177637879437
    flat: 0.14704000000000003
  - slug: iran-charges-hormuz-fees-by-october-31
    kind: by_date
    end: '2026-10-31'
    days: 59
    sim: 0.06918434422621919
    flat: 0.1641525423728814
  - slug: iran-charges-hormuz-fees-by-december-31-20260706203029175-553
    kind: by_date
    end: '2026-12-31'
    days: 58
    sim: 0.11834797772665602
    flat: 0.18853448275862067
  - slug: iran-charges-hormuz-fees-by-september-30
    kind: by_date
    end: '2026-09-30'
    days: 43
    sim: 0.06327443155568643
    flat: 0.16119767441860464
  - slug: total-internet-blackout-in-iran-by-august-31-2026-20260720155238428
    kind: by_date
    end: '2026-08-31'
    days: 31
    sim: 0.14941218051669813
    flat: 0.2749838709677419
  - slug: total-internet-blackout-in-iran-by-september-30-2026
    kind: by_date
    end: '2026-09-30'
    days: 41
    sim: 0.2813923544438405
    flat: 0.44663414634146337
  - slug: total-internet-blackout-in-iran-by-december-31-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.17419502292538397
    flat: 0.2531666666666667
  - slug: iran-agrees-to-surrender-enriched-uranium-stockpile-by-may-31-2026
    kind: by_date
    end: '2026-05-31'
    days: 28
    sim: 0.030454298163537907
    flat: 0.030571428571428576
  - slug: iran-agrees-to-surrender-enriched-uranium-stockpile-by-july-31-2026
    kind: by_date
    end: '2026-07-31'
    days: 33
    sim: 0.076839230628957
    flat: 0.12493939393939392
  - slug: iran-agrees-to-surrender-enriched-uranium-stockpile-by-august-31-2026
    kind: by_date
    end: '2026-08-31'
    days: 18
    sim: 0.007830913743831028
    flat: 0.006638888888888885
  - slug: iran-agrees-to-surrender-enriched-uranium-stockpile-by-october-31-2026
    kind: by_date
    end: '2026-10-31'
    days: 47
    sim: 0.020885151428254464
    flat: 0.028723404255319166
  - slug: iran-agrees-to-surrender-enriched-uranium-stockpile-by-june-30-2026
    kind: by_date
    end: '2026-06-30'
    days: 44
    sim: 0.08478113625272729
    flat: 0.05159090909090909
  - slug: iran-agrees-to-surrender-enriched-uranium-stockpile-by-december-31-2026-752-299
    kind: by_date
    end: '2026-12-31'
    days: 41
    sim: 0.11291280162713138
    flat: 0.08182926829268292
  - slug: will-reza-pahlavi-enter-iran-by-june-30-555-947-362-379-213
    kind: by_date
    end: '2026-06-30'
    days: 60
    sim: 0.08273098877238595
    flat: 0.07275000000000002
  - slug: will-reza-pahlavi-enter-iran-by-march-31-282
    kind: by_date
    end: '2026-03-31'
    days: 25
    sim: 0.09979777481474508
    flat: 0.12933999999999998
  - slug: will-reza-pahlavi-enter-iran-by-february-28
    kind: by_date
    end: '2026-02-28'
    days: 12
    sim: 0.025095643004863596
    flat: 0.03558333333333333
  - slug: will-reza-pahlavi-enter-iran-by-december-31-898-927
    kind: by_date
    end: '2026-12-31'
    days: 37
    sim: 0.1560901337353937
    flat: 0.17837837837837842
  - slug: will-reza-pahlavi-enter-iran-by-april-30
    kind: by_date
    end: '2026-04-30'
    days: 5
    sim: 0.013303236877857231
    flat: 0.0157
  - slug: will-reza-pahlavi-enter-iran-by-may-31
    kind: by_date
    end: '2026-05-31'
    days: 8
    sim: 0.010770593292196297
    flat: 0.012000000000000002
  - slug: will-iran-target-oman-by-september-15-2026
    kind: by_date
    end: '2026-09-15'
    days: 19
    sim: 0.32597074994892217
    flat: 0.45552631578947367
  - slug: will-iran-target-oman-by-september-30-2026
    kind: by_date
    end: '2026-09-30'
    days: 27
    sim: 0.3160453750975788
    flat: 0.4371666666666667
  - slug: will-a-european-country-take-military-action-on-iran-by-august-31-20260720005529606
    kind: by_date
    end: '2026-08-31'
    days: 12
    sim: 0.01782767242384076
    flat: 0.020875000000000005
  - slug: will-a-european-country-take-military-action-on-iran-by-october-31
    kind: by_date
    end: '2026-10-31'
    days: 36
    sim: 0.32147215463724294
    flat: 0.4204861111111111
  - slug: will-a-european-country-take-military-action-on-iran-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 50
    sim: 0.32849935981007433
    flat: 0.40070000000000006
  - slug: us-reissues-iran-oil-sanction-relief-by-august-31-20260707233859434
    kind: by_date
    end: '2026-08-31'
    days: 48
    sim: 0.09468982338231903
    flat: 0.20709375
  - slug: us-reissues-iran-oil-sanction-relief-by-september-30
    kind: by_date
    end: '2026-09-30'
    days: 31
    sim: 0.16796049249739353
    flat: 0.31385483870967745
  - slug: us-reissues-iran-oil-sanction-relief-by-october-31
    kind: by_date
    end: '2026-10-31'
    days: 39
    sim: 0.18328063665818592
    flat: 0.32141025641025633
  - slug: saudi-arabia-military-action-against-iran-by-december-31-2026-20260916
    kind: by_date
    end: '2026-12-31'
    days: 29
    sim: 0.042678928166430374
    flat: 0.03586206896551724
  - slug: will-iran-target-syria-by-september-15-2026
    kind: by_date
    end: '2026-09-15'
    days: 16
    sim: 0.3227394280243645
    flat: 0.42440625
  - slug: will-iran-target-syria-by-september-30-2026
    kind: by_date
    end: '2026-09-30'
    days: 29
    sim: 0.29206850614010804
    flat: 0.41993103448275865
  - slug: will-iran-target-qatar-by-september-15-2026
    kind: by_date
    end: '2026-09-15'
    days: 26
    sim: 0.17748422400482344
    flat: 0.3473076923076924
  - slug: will-iran-target-qatar-by-september-30-2026
    kind: by_date
    end: '2026-09-30'
    days: 41
    sim: 0.14399167315567107
    flat: 0.3360609756097561
  - slug: iran-nuclear-test-before-2027
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.058185839825949694
    flat: 0.06041666666666667
  - slug: will-iran-target-saudi-arabia-by-september-15-2026
    kind: by_date
    end: '2026-09-15'
    days: 26
    sim: 0.17455949622424105
    flat: 0.35173076923076924
  - slug: will-iran-target-saudi-arabia-by-september-30-2026
    kind: by_date
    end: '2026-09-30'
    days: 35
    sim: 0.12224200154638824
    flat: 0.27298571428571433
  - slug: will-iran-target-saudi-arabia-by-october-31-2026
    kind: by_date
    end: '2026-10-31'
    days: 30
    sim: 0.09987726461957881
    flat: 0.19085000000000002
  - slug: will-israel-launch-a-ground-operation-in-iran-by-june-30-2026
    kind: by_date
    end: '2026-06-30'
    days: 14
    sim: 0.04524384370321163
    flat: 0.07017857142857144
  - slug: will-israel-launch-a-ground-operation-in-iran-by-july-31-2026
    kind: by_date
    end: '2026-07-31'
    days: 1
    sim: 0.04036660966452535
    flat: 0.040499999999999994
  - slug: will-israel-launch-a-ground-operation-in-iran-by-december-31-2026-20260701160841000-645
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.03282991419229875
    flat: 0.04889830508474577
  - slug: will-israel-launch-a-ground-operation-in-iran-by-august-31-2026
    kind: by_date
    end: '2026-08-31'
    days: 4
    sim: 0.062283860003844005
    flat: 0.06624999999999999
  - slug: will-israel-launch-a-ground-operation-in-iran-by-may-31-2026
    kind: by_date
    end: '2026-05-31'
    days: 42
    sim: 0.11098230257186759
    flat: 0.2919285714285715
  - slug: mohammed-ghalibaf-out-as-speaker-of-iran-parliament-by-september-30-2026-20260710000011404
    kind: by_date
    end: '2026-09-30'
    days: 37
    sim: 0.062294704981143374
    flat: 0.05945945945945946
  - slug: mohammed-ghalibaf-out-as-speaker-of-iran-parliament-by-december-31-2026-20260710000011405
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.059121515530772516
    flat: 0.09474576271186444
  - slug: iran-agrees-to-end-enrichment-of-uranium-by-december-31-1
    kind: by_date
    end: '2026-12-31'
    days: 56
    sim: 0.17723199579403395
    flat: 0.14892857142857147
  - slug: will-the-us-officially-declare-war-on-iran-by-march-31-2026
    kind: by_date
    end: '2026-03-31'
    days: 19
    sim: 0.00750581887818347
    flat: 0.008236842105263157
  - slug: will-the-us-officially-declare-war-on-iran-by-december-31-2026-746
    kind: by_date
    end: '2026-12-31'
    days: 54
    sim: 0.018771541661739784
    flat: 0.017685185185185186
  - slug: will-the-unsc-adopt-a-resolution-endorsing-the-final-us-iran-deal-by-date
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.20808143794673892
    flat: 0.26741666666666675
  - slug: iran-blockade-lifted-before-iran-oman-hormuz-agreement
    kind: by_date
    end: '2026-12-31'
    days: 43
    sim: 0.0645252002964288
    flat: 0.09348837209302327
  - slug: israel-x-hamas-ceasefire-phase-ii-by-december-31-632
    kind: by_date
    end: '2026-12-31'
    days: 58
    sim: 0.15705492104299465
    flat: 0.18448275862068966
  - slug: israel-x-hamas-ceasefire-phase-ii-by-december-31-521
    kind: by_date
    end: '2025-12-31'
    days: 51
    sim: 0.1413608348779842
    flat: 0.2516666666666667
  - slug: israel-x-hamas-ceasefire-phase-ii-by-november-30
    kind: by_date
    end: '2025-11-30'
    days: 12
    sim: 0.14549305279086644
    flat: 0.1841666666666667
  - slug: israel-x-hamas-ceasefire-phase-ii-by-march-31-117-352
    kind: by_date
    end: '2026-03-31'
    days: 60
    sim: 0.06722119470700925
    flat: 0.09508333333333331
  - slug: israel-x-hamas-ceasefire-phase-ii-by-january-31
    kind: by_date
    end: '2026-01-31'
    days: 35
    sim: 0.13894460160495933
    flat: 0.28742857142857137
  - slug: israel-x-hamas-ceasefire-phase-ii-by-june-30
    kind: by_date
    end: '2026-06-30'
    days: 60
    sim: 0.1350867555738509
    flat: 0.12166666666666666
  - slug: will-north-korea-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 34
    sim: 0.05450606325509347
    flat: 0.05558823529411765
  - slug: will-cuba-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 58
    sim: 0.017490587819050434
    flat: 0.021000000000000005
  - slug: will-saudi-arabia-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.14199107000032787
    flat: 0.16966101694915256
  - slug: will-lebanon-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.059544603082856196
    flat: 0.060254237288135594
  - slug: will-afghanistan-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 31
    sim: 0.18974038948504215
    flat: 0.2041935483870967
  - slug: will-iraq-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 30
    sim: 0.16582919130986246
    flat: 0.17783333333333326
  - slug: will-pakistan-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.08278941717054225
    flat: 0.09313559322033896
  - slug: will-syria-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.3143962281828835
    flat: 0.36058333333333337
  - slug: will-venezuela-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.07005094922030107
    flat: 0.08567796610169491
  - slug: will-tunisia-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.18866892323341183
    flat: 0.21822033898305088
  - slug: will-kuwait-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.02970113896176255
    flat: 0.035338983050847456
  - slug: will-qatar-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 40
    sim: 0.17204748676717282
    flat: 0.19150000000000006
  - slug: will-indonesia-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.09666912480023207
    flat: 0.11940677966101697
  - slug: will-malaysia-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.0531345145221476
    flat: 0.05538983050847458
  - slug: will-bangladesh-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.19185070939909435
    flat: 0.22216666666666668
  - slug: will-iran-recognize-israel-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 36
    sim: 0.1143263362056771
    flat: 0.12779166666666664
  - slug: israel-withdraws-from-lebanon-by-may-31-2026
    kind: by_date
    end: '2026-05-31'
    days: 4
    sim: 0.016161283921944547
    flat: 0.018749999999999996
  - slug: israel-withdraws-from-lebanon-by-june-30-2026
    kind: by_date
    end: '2026-06-30'
    days: 59
    sim: 0.025916741568653156
    flat: 0.07659322033898304
  - slug: israel-withdraws-from-lebanon-by-july-31-2026
    kind: by_date
    end: '2026-07-31'
    days: 34
    sim: 0.255502424121869
    flat: 0.3432352941176471
  - slug: israel-withdraws-from-lebanon-by-august-31-2026
    kind: by_date
    end: '2026-08-31'
    days: 15
    sim: 0.17376204504605866
    flat: 0.19653333333333337
  - slug: israel-withdraws-from-lebanon-by-september-30-2026
    kind: by_date
    end: '2026-09-30'
    days: 45
    sim: 0.25002200728214247
    flat: 0.3200111111111112
  - slug: israel-withdraws-from-lebanon-by-december-31-2026
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.5135758539233312
    flat: 0.5751694915254238
  - slug: will-israel-strike-0-countries-in-2026
    kind: by_date
    end: '2026-12-30'
    days: 2
    sim: 0.010075031777425688
    flat: 0.010000000000000002
  - slug: will-israel-strike-1-country-in-2026
    kind: by_date
    end: '2026-12-30'
    days: 47
    sim: 0.1302136790828502
    flat: 0.13693617021276597
  - slug: will-israel-strike-2-countries-in-2026
    kind: by_date
    end: '2026-12-30'
    days: 60
    sim: 0.06484248789707008
    flat: 0.07368333333333334
  - slug: will-israel-strike-3-countries-in-2026
    kind: by_date
    end: '2026-12-30'
    days: 60
    sim: 0.11366865609008438
    flat: 0.10408333333333335
  - slug: will-israel-strike-4-countries-in-november-2025
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.10204199330332328
    flat: 0.10842500000000002
  - slug: will-israel-strike-5-countries-in-november-2025
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.15636221917628432
    flat: 0.1672833333333333
  - slug: will-israel-strike-6-countries-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 13
    sim: 0.17098416921388784
    flat: 0.17307692307692307
  - slug: will-israel-strike-7-countries-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 18
    sim: 0.19569036864200418
    flat: 0.19944444444444442
  - slug: will-israel-strike-8-countries-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 19
    sim: 0.11259952497082161
    flat: 0.11489473684210523
  - slug: will-israel-strike-9-countries-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 18
    sim: 0.0671927780319267
    flat: 0.06763888888888889
  - slug: will-israel-strike-10-countries-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 19
    sim: 0.03741027694977379
    flat: 0.0358421052631579
  - slug: will-israel-strike-11-countries-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 13
    sim: 0.013576198934467754
    flat: 0.012038461538461536
  - slug: will-israel-strike-12-countries-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 5
    sim: 0.017065960300151183
    flat: 0.0171
  - slug: will-israel-strike-13-countries-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 19
    sim: 0.05705163914301342
    flat: 0.06128947368421053
  - slug: will-israel-strike-14-countries-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.0050120046871657276
    flat: 0.0050000000000000044
  - slug: will-israel-strike-15-or-more-countries-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 1
    sim: 0.004988775776357063
    flat: 0.0050000000000000044
  - slug: israel-military-action-against-syria-by-september-30-2026-21
    kind: by_date
    end: '2026-09-30'
    days: 35
    sim: 0.1429007347319938
    flat: 0.32841428571428577
  - slug: will-there-be-houthi-military-action-against-israel-by-august-31-2026
    kind: by_date
    end: '2026-08-31'
    days: 25
    sim: 0.15153097985199518
    flat: 0.26264
  - slug: will-there-be-houthi-military-action-against-israel-by-september-30-2026
    kind: by_date
    end: '2026-09-30'
    days: 21
    sim: 0.2905293439353288
    flat: 0.40130952380952384
  - slug: israel-x-lebanon-effective-ceasefire-by-september-30
    kind: by_date
    end: '2026-09-30'
    days: 20
    sim: 0.24975760182726137
    flat: 0.3846
  - slug: israel-x-turkey-military-clash-before-2027
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.03726159020216794
    flat: 0.038916666666666676
  - slug: israel-military-action-against-yemen-by-august-31
    kind: by_date
    end: '2026-08-31'
    days: 12
    sim: 0.12107717990865108
    flat: 0.15612499999999999
  - slug: israel-military-action-against-yemen-by-september-30
    kind: by_date
    end: '2026-09-30'
    days: 33
    sim: 0.07965778075387137
    flat: 0.15836363636363637
  - slug: israel-military-action-against-yemen-by-december-31
    kind: by_date
    end: '2026-12-31'
    days: 30
    sim: 0.0946554407599021
    flat: 0.1335
  - slug: israel-military-action-against-yemen-by-october-31
    kind: by_date
    end: '2026-10-31'
    days: 30
    sim: 0.20904594119016817
    flat: 0.3330666666666667
  - slug: will-israel-launch-a-major-ground-offensive-in-gaza-by-december-31
    kind: by_date
    end: '2025-10-30'
    days: 9
    sim: 0.07533311071376952
    flat: 0.10166666666666667
  - slug: will-israel-launch-a-major-ground-offensive-in-gaza-by-december-31-828
    kind: by_date
    end: '2026-12-31'
    days: 41
    sim: 0.08320703258260441
    flat: 0.10865853658536588
  - slug: israel-military-action-against-greater-beirut-by-august-31-20260708214159403
    kind: by_date
    end: '2026-08-31'
    days: 47
    sim: 0.08498383618807771
    flat: 0.23704255319148937
  - slug: israel-military-action-against-greater-beirut-by-september-30
    kind: by_date
    end: '2026-09-30'
    days: 31
    sim: 0.20185694546311836
    flat: 0.37798387096774194
  - slug: israel-military-action-against-greater-beirut-by-october-31
    kind: by_date
    end: '2026-10-31'
    days: 44
    sim: 0.028945444925949117
    flat: 0.08397727272727275
  - slug: israel-and-syria-normalize-relations-in-2025
    kind: by_date
    end: '2025-12-30'
    days: 60
    sim: 0.10865042661485218
    flat: 0.11975
  - slug: israel-and-syria-normalize-relations-by-december-31-2026-166
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.162707187264144
    flat: 0.188
  - slug: israel-and-syria-normalize-relations-by-june-30-2026
    kind: by_date
    end: '2025-12-30'
    days: 47
    sim: 0.10585496353157775
    flat: 0.26606382978723414
  - slug: israel-and-lebanon-normalize-relations-before-2027
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.05545832326491269
    flat: 0.06483333333333334
  - slug: israel-and-saudi-arabia-normalize-relations-before-2027
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.20240351690402647
    flat: 0.22808333333333322
  - slug: another-pandemic-before-gta-vi
    kind: by_date
    end: '2026-07-31'
    days: 60
    sim: 0.1644074902371789
    flat: 0.015416666666666681
  - slug: russia-ukraine-ceasefire-before-gta-vi-554
    kind: by_date
    end: '2026-07-31'
    days: 60
    sim: 0.08981172822405692
    flat: 0.11549999999999998
  - slug: new-rhianna-album-before-gta-vi-926
    kind: by_date
    end: '2026-07-31'
    days: 60
    sim: 0.12856527038780202
    flat: 0.10550000000000002
  - slug: new-playboi-carti-album-before-gta-vi-421
    kind: by_date
    end: '2026-07-31'
    days: 60
    sim: 0.14062239387894976
    flat: 0.14475
  - slug: will-jesus-christ-return-before-gta-vi-665
    kind: by_date
    end: '2026-07-31'
    days: 60
    sim: 0.014785868400137646
    flat: 0.01791666666666666
  - slug: trump-out-as-president-before-gta-vi-846
    kind: by_date
    end: '2026-07-31'
    days: 60
    sim: 0.020159960733891348
    flat: 0.02158333333333334
  - slug: will-china-invades-taiwan-before-gta-vi-716-644
    kind: by_date
    end: '2026-07-31'
    days: 60
    sim: 0.011001009234803875
    flat: 0.010749999999999997
  - slug: will-bitcoin-hit-1m-before-gta-vi-872-424
    kind: by_date
    end: '2026-07-31'
    days: 60
    sim: 0.0224045568470444
    flat: 0.022166666666666664
  - slug: will-gpt-6-be-released
    kind: by_date
    end: '2026-07-31'
    days: 60
    sim: 0.09561323241899615
    flat: 0.10883333333333335
  - slug: will-drake-release-iceman-before-gta-vi-355
    kind: by_date
    end: '2026-07-31'
    days: 54
    sim: 0.1985639937123114
    flat: 0.1469722222222222
  - slug: russia-x-ukraine-ceasefire-agreement-by-june-30-2026
    kind: by_date
    end: '2026-06-30'
    days: 35
    sim: 0.014928035568678453
    flat: 0.030471428571428573
  - slug: russia-x-ukraine-ceasefire-agreement-by-october-31-2026
    kind: by_date
    end: '2026-10-31'
    days: 60
    sim: 0.03499407405928283
    flat: 0.03516666666666667
  - slug: russia-x-ukraine-ceasefire-agreement-by-december-31-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.05254357047997365
    flat: 0.028000000000000004
  - slug: russia-x-ukraine-ceasefire-agreement-by-august-31-2026
    kind: by_date
    end: '2026-08-31'
    days: 50
    sim: 0.015140879269254448
    flat: 0.053819999999999986
  - slug: russia-x-ukraine-ceasefire-agreement-by-november-30-2026
    kind: by_date
    end: '2026-11-30'
    days: 37
    sim: 0.021297393169930078
    flat: 0.047162162162162166
  - slug: russia-x-ukraine-ceasefire-by-june-30-2026-631
    kind: by_date
    end: '2026-06-30'
    days: 15
    sim: 0.01712671028223573
    flat: 0.02663333333333334
  - slug: russia-x-ukraine-ceasefire-by-october-31-2026
    kind: by_date
    end: '2026-10-31'
    days: 60
    sim: 0.023017392607570505
    flat: 0.041249999999999995
  - slug: russia-x-ukraine-ceasefire-by-december-31-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.03233134380844581
    flat: 0.044
  - slug: russia-x-ukraine-ceasefire-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.05714153648760852
    flat: 0.03608333333333334
  - slug: russia-x-ukraine-ceasefire-by-june-30-2026
    kind: by_date
    end: '2026-06-29'
    days: 57
    sim: 0.05161086717377074
    flat: 0.09333333333333335
  - slug: russia-x-ukraine-ceasefire-by-january-31-2026
    kind: by_date
    end: '2026-01-30'
    days: 41
    sim: 0.042628410447268346
    flat: 0.09425609756097561
  - slug: russia-x-ukraine-ceasefire-before-july
    kind: by_date
    end: '2025-06-30'
    days: 60
    sim: 0.0993884910314667
    flat: 0.05625
  - slug: russia-x-ukraine-ceasefire-before-2027
    kind: by_date
    end: '2026-12-30'
    days: 60
    sim: 0.07504593522633826
    flat: 0.06641666666666668
  - slug: russia-x-ukraine-ceasefire-by-april-30-2026
    kind: by_date
    end: '2026-04-29'
    days: 27
    sim: 0.014846592134984038
    flat: 0.026074074074074076
  - slug: russia-x-ukraine-ceasefire-before-june
    kind: by_date
    end: '2025-05-31'
    days: 25
    sim: 0.04911357521260271
    flat: 0.10352
  - slug: russia-x-ukraine-ceasefire-by-february-28-2026
    kind: by_date
    end: '2026-02-27'
    days: 14
    sim: 0.02588315404299918
    flat: 0.04125
  - slug: russia-x-ukraine-ceasefire-by-november-30-513
    kind: by_date
    end: '2025-11-29'
    days: 13
    sim: 0.030988919704731444
    flat: 0.04311538461538462
  - slug: russia-x-ukraine-ceasefire-by-march-31-2026
    kind: by_date
    end: '2026-03-30'
    days: 60
    sim: 0.065765452154741
    flat: 0.08341666666666665
  - slug: will-no-press-briefing-air-by-august-31-20260723172550028
    kind: by_date
    end: '2026-08-31'
    days: 1
    sim: 0.003936213785440339
    flat: 0.003999999999999997
  - slug: will-black-sea-ceasefire-go-into-effect-before-may
    kind: by_date
    end: '2025-04-30'
    days: 26
    sim: 0.077871922153962
    flat: 0.17784615384615388
  - slug: will-china-invade-taiwan-before-2027
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.2183931579886928
    flat: 0.2358333333333333
  - slug: china-x-taiwan-military-clash-before-2027
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.1082647375918174
    flat: 0.12274999999999997
  - slug: will-china-blockade-taiwan-by-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.04881144102360514
    flat: 0.05597457627118644
  - slug: will-china-invade-taiwan-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.007466560053377832
    flat: 0.007083333333333334
  - slug: will-china-invade-taiwan-by-june-30-2026
    kind: by_date
    end: '2026-06-29'
    days: 60
    sim: 0.017312390067937326
    flat: 0.016433333333333338
  - slug: will-china-invade-taiwan-by-march-31-2026
    kind: by_date
    end: '2026-03-30'
    days: 10
    sim: 0.017208248679903077
    flat: 0.018499999999999996
  - slug: will-china-invade-taiwan-by-september-30-2026
    kind: by_date
    end: '2026-09-30'
    days: 38
    sim: 0.05787513708967428
    flat: 0.06894736842105265
  - slug: will-china-blockade-taiwan-by-june-30
    kind: by_date
    end: '2026-06-29'
    days: 60
    sim: 0.05293535669669196
    flat: 0.06375
  - slug: china-x-taiwan-military-clash-by-june-30
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.025490934959711487
    flat: 0.023125000000000003
  - slug: china-x-taiwan-military-clash-by-december-31
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.014282806317856358
    flat: 0.012666666666666666
  - slug: will-china-blockade-taiwan-by-december-31
    kind: by_date
    end: '2025-12-30'
    days: 13
    sim: 0.14655620476151263
    flat: 0.1566923076923077
  - slug: lai-ching-te-out-as-president-of-taiwan-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.34094066673325923
    flat: 0.36808333333333326
  - slug: will-xi-jinping-meet-with-cheng-li-wun-before-2027
    kind: by_date
    end: '2026-06-29'
    days: 60
    sim: 0.0683826936618203
    flat: 0.09108333333333328
  - slug: taiwanese-premier-cho-jung-tai-out-by-june-30-2026
    kind: by_date
    end: '2026-06-30'
    days: 58
    sim: 0.07009845853065125
    flat: 0.07112068965517242
  - slug: taiwanese-premier-cho-jung-tai-out-by-december-31-2026
    kind: by_date
    end: '2026-12-31'
    days: 55
    sim: 0.13503514890490856
    flat: 0.11854545454545459
  - slug: will-donald-trump-visit-lebanon-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 57
    sim: 0.14289533285796105
    flat: 0.15949122807017546
  - slug: will-donald-trump-visit-venezuela-in-2026-20260625195412667
    kind: by_date
    end: '2026-12-31'
    days: 59
    sim: 0.3444690507462758
    flat: 0.40161016949152545
  - slug: will-donald-trump-visit-the-united-kingdom-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.06321857655778224
    flat: 0.07024999999999995
  - slug: will-donald-trump-visit-canada-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.1055765087709946
    flat: 0.1296666666666667
  - slug: will-donald-trump-visit-mexico-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.23628021607299646
    flat: 0.22033333333333333
  - slug: will-donald-trump-visit-saudi-arabia-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.06589635839936686
    flat: 0.07525
  - slug: will-donald-trump-visit-japan-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.14607602136881176
    flat: 0.17333333333333334
  - slug: will-donald-trump-visit-south-korea-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.0869326529884014
    flat: 0.10016666666666668
  - slug: will-donald-trump-visit-france-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.04340036836732811
    flat: 0.042083333333333334
  - slug: will-donald-trump-visit-russia-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.10471416218371009
    flat: 0.09925
  - slug: will-donald-trump-visit-ukraine-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.21344590938395383
    flat: 0.23374999999999999
  - slug: will-donald-trump-visit-taiwan-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.26993075544233136
    flat: 0.29125
  - slug: will-donald-trump-visit-china-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.17476218270372473
    flat: 0.14633333333333334
  - slug: will-donald-trump-visit-italy-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.06623374337626962
    flat: 0.06608333333333334
  - slug: will-donald-trump-visit-oman-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.05716542970424607
    flat: 0.06391666666666666
  - slug: will-donald-trump-visit-india-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.1699392430330218
    flat: 0.14416666666666667
  - slug: will-donald-trump-visit-belarus-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.10317829681466631
    flat: 0.11083333333333333
  - slug: will-donald-trump-visit-turkey-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.04316429033611905
    flat: 0.055583333333333346
  - slug: will-donald-trump-visit-switzerland-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 36
    sim: 0.35442141905798663
    flat: 0.33923611111111107
  - slug: will-donald-trump-visit-syria-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.194225865183943
    flat: 0.22033333333333338
  - slug: will-donald-trump-visit-ireland-in-2026
    kind: by_date
    end: '2026-12-31'
    days: 40
    sim: 0.23599397891790286
    flat: 0.209375
  - slug: which-countries-will-donald-trump-visit-in-2025-will-donald-trump-visit-the-united-kingdom-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.15026140441034924
    flat: 0.11500833333333332
  - slug: which-countries-will-donald-trump-visit-in-2025-will-donald-trump-visit-israel-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.04300056644037466
    flat: 0.04774999999999999
  - slug: which-countries-will-donald-trump-visit-in-2025-will-donald-trump-visit-canada-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.019415890297922168
    flat: 0.03683333333333336
  - slug: which-countries-will-donald-trump-visit-in-2025-will-donald-trump-visit-mexico-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.026684786124497057
    flat: 0.03433333333333333
  - slug: which-countries-will-donald-trump-visit-in-2025-will-donald-trump-visit-saudi-arabia-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.08843738685627375
    flat: 0.05593333333333337
  - slug: which-countries-will-donald-trump-visit-in-2025-will-donald-trump-visit-japan-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.1106979752631009
    flat: 0.10558333333333332
  - slug: which-countries-will-donald-trump-visit-in-2025-will-donald-trump-visit-germany-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.1376233797754346
    flat: 0.1698333333333333
  - slug: which-countries-will-donald-trump-visit-in-2025-will-donald-trump-visit-south-korea-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.03236042999004829
    flat: 0.06666666666666662
  - slug: which-countries-will-donald-trump-visit-in-2025-will-donald-trump-visit-france-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.09582878292093473
    flat: 0.12899999999999998
  - slug: will-donald-trump-visit-russia-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.14174386585508889
    flat: 0.12933333333333333
  - slug: will-donald-trump-visit-ukraine-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.11831264942418149
    flat: 0.14283333333333337
  - slug: will-donald-trump-visit-taiwan-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.008993785699583482
    flat: 0.015583333333333336
  - slug: will-donald-trump-visit-china-in-2025-1
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.19146244960465578
    flat: 0.22525000000000006
  - slug: will-donald-trump-visit-italy-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 55
    sim: 0.14112271286317724
    flat: 0.10881818181818186
  - slug: will-donald-trump-visit-oman-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.1476730656177819
    flat: 0.21200000000000005
  - slug: will-donald-trump-visit-india-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.17150591813067587
    flat: 0.23625
  - slug: will-donald-trump-visit-belarus-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.2788726072633753
    flat: 0.34983333333333333
  - slug: will-donald-trump-visit-belarus-in-2025-323
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.027100232130917563
    flat: 0.026250000000000002
  - slug: us-recession-in-2025
    kind: by_date
    end: '2026-02-28'
    days: 60
    sim: 0.060995258097543206
    flat: 0.0565
  - slug: us-recession-announced-by-nber-before-june-2025
    kind: by_date
    end: '2025-04-30'
    days: 60
    sim: 0.10762425477964944
    flat: 0.14250000000000002
  - slug: canada-recession-before-2027
    kind: by_date
    end: '2026-12-30'
    days: 60
    sim: 0.07318403211377915
    flat: 0.06916666666666664
  - slug: government-shutdown-by-october-1-20260610162414910
    kind: by_date
    end: '2026-10-01'
    days: 59
    sim: 0.10039179546358429
    flat: 0.07177966101694916
  - slug: us-government-shutdown-in-2025
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.2293959062853638
    flat: 0.2198333333333333
  - slug: will-there-be-another-us-government-shutdown-by-december-31
    kind: by_date
    end: '2025-12-30'
    days: 39
    sim: 0.041698273211748786
    flat: 0.07282051282051283
  - slug: will-the-federal-government-be-shut-down-for-1521-days-in-2025
    kind: by_date
    end: '2025-12-30'
    days: 41
    sim: 0.17017080778976826
    flat: 0.17135365853658538
  - slug: debt-ceiling-raisedsuspended-before-trump-inauguration
    kind: by_date
    end: '2025-01-19'
    days: 6
    sim: 0.6441062313593974
    flat: 0.6739999999999999
  - slug: debt-ceiling-abolished-before-trump-inauguration
    kind: by_date
    end: '2025-01-19'
    days: 4
    sim: 0.03129611296404382
    flat: 0.034875
  - slug: will-warsh-be-out-as-fed-chair-by-december-31-2026-20260723173503767
    kind: by_date
    end: '2026-12-31'
    days: 53
    sim: 0.025138968848328915
    flat: 0.023424528301886797
  - slug: will-trump-try-to-fire-powell-as-fed-board-member-by-december-31-2026-239
    kind: by_date
    end: '2026-12-31'
    days: 57
    sim: 0.03032291034772069
    flat: 0.022105263157894742
  - slug: will-trump-try-to-fire-powell-as-fed-board-member-by-june-30-2026
    kind: by_date
    end: '2026-06-30'
    days: 57
    sim: 0.019973534838098737
    flat: 0.03528070175438597
  - slug: will-trump-try-to-fire-powell-as-fed-board-member-by-july-31-2026-20260626181928589
    kind: by_date
    end: '2026-07-31'
    days: 9
    sim: 0.3685772365820922
    flat: 0.4075555555555556
  - slug: will-trump-try-to-fire-powell-as-fed-board-member-by-august-31-2026
    kind: by_date
    end: '2026-08-31'
    days: 27
    sim: 0.1052469685554324
    flat: 0.26481481481481484
  - slug: will-trump-try-to-fire-powell-as-fed-board-member-by-september-30-2026
    kind: by_date
    end: '2026-09-30'
    days: 33
    sim: 0.1304422984545752
    flat: 0.12963636363636363
  - slug: lisa-cook-out-as-fed-governor-by-september-30-20260807204638263
    kind: by_date
    end: '2026-09-30'
    days: 1
    sim: 0.2945433058763036
    flat: 0.2945
  - slug: lisa-cook-out-as-fed-governor-by-october-31-20260807204638264
    kind: by_date
    end: '2026-10-31'
    days: 1
    sim: 0.4395264816405822
    flat: 0.44
  - slug: lisa-cook-out-as-fed-governor-by-november-30-20260807204638265
    kind: by_date
    end: '2026-11-30'
    days: 39
    sim: 0.34819695145343993
    flat: 0.4056410256410256
  - slug: lisa-cook-out-as-fed-governor-by-december-31-20260807204638266
    kind: by_date
    end: '2026-12-31'
    days: 60
    sim: 0.3220218132064253
    flat: 0.3918333333333334
  - slug: will-trump-remove-jerome-powell
    kind: by_date
    end: '2025-12-31'
    days: 60
    sim: 0.06885496617519128
    flat: 0.08050000000000002
  - slug: lisa-cook-out-as-fed-governor-by-september-30
    kind: by_date
    end: '2025-09-29'
    days: 28
    sim: 0.019575903095169208
    flat: 0.031607142857142854
  - slug: lisa-cook-out-as-fed-governor-by-december-31-293-933
    kind: by_date
    end: '2025-09-29'
    days: 32
    sim: 0.11414732216574652
    flat: 0.02109375000000001
  - slug: lisa-cook-out-as-fed-governor-by-february-28
    kind: by_date
    end: '2026-02-27'
    days: 18
    sim: 0.19191959558160399
    flat: 0.2236111111111111
  - slug: jerome-powell-out-as-fed-chair-by-march-31-2026
    kind: by_date
    end: '2026-05-13'
    days: 3
    sim: 0.030806450137678455
    flat: 0.031333333333333345
  - slug: jerome-powell-out-as-fed-chair-by-may-14-2026
    kind: by_date
    end: '2026-05-13'
    days: 48
    sim: 0.05524355262797959
    flat: 0.07405208333333334
  - slug: jerome-powell-out-as-fed-chair-by-august-31
    kind: by_date
    end: '2025-08-31'
    days: 14
    sim: 0.03251812411886002
    flat: 0.041464285714285724
  - slug: kevin-warsh-confirmed-as-fed-chair-by-may-15-961
    kind: by_date
    end: '2026-05-14'
    days: 51
    sim: 0.10399558249474906
    flat: 0.08833333333333333
  - slug: kevin-warsh-confirmed-as-fed-chair-by-may-1
    kind: by_date
    end: '2026-05-14'
    days: 57
    sim: 0.08442240554032428
    flat: 0.19482456140350882
B6:
  summary: 'per point of unemployment: simulated mood -0.021% of its scale, real sentiment
    -3.29% of its range (exploratory)'
  passed: null
  unemployment_rise_points: 1.132
  mood_change: -0.00047785938327092733
  sim_slope_per_point: -0.0004222464378040521
  sim_share_of_range_per_point: -0.00021112321890202605
  real_share_of_range_per_point: -0.03288
  real:
    months: 563
    range: 1978-01-01..2025-12-01
    index_min: 50.0
    index_max: 112.0
    slope_share_of_range: -0.03288
    slope_points_per_unemployment_point: -2.039
    sd_of_12_month_changes: 10.692
    slope_in_sd: -0.191
  ratio_sim_to_real: 0.006421022472689357
  flag: true
  note: 'reported only: real sentiment also moves with news about jobs; the two scales
    differ'
  runs:
    baseline:
      mood_end: -0.0018610062363233495
      unemployment_end_pct: 4.2926829268292686
      daily_change_sd: 8.353712379285498e-06
    layoffs:
      mood_end: -0.002338865619594277
      unemployment_end_pct: 5.424390243902439
      daily_change_sd: 9.54316491546075e-06
```
