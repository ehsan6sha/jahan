"""Behavioural coefficients.

All values are uncalibrated priors.  The media, sharing and adoption values were
tuned with ``scripts/calibrate_launch.py`` so a mid-priced product launch with a
30-day campaign lands inside PROVISIONAL target bands (awareness 30-50% at day
30 and 50-80% at day 180; adult adoption 2-8% at day 180; 20-50%+ of first
exposures via people).  Those bands are judgement, not sourced data; empirical
calibration against real launch and diffusion data is an M2 task.
"""

from __future__ import annotations

from pydantic import BaseModel


class SimParams(BaseModel):
    # contacts
    strength_household: float = 0.85
    strength_kin: float = 0.12
    strength_colleague: float = 0.35
    strength_classmate: float = 0.40
    strength_friend: float = 0.20
    strength_acquaintance: float = 0.04
    strength_cross_border: float = 0.03
    overlap_hours_full: float = 4.0  # hours of shared free time for full contact propensity
    married_social_time: float = 0.8  # HEURISTIC share of outside-household social time kept after marriage
    young_child_social_time: float = 0.7  # HEURISTIC ... while caring for a child under 5
    life_events: bool = True
    travel: bool = True
    travel_in_world_share: float = 0.15  # HEURISTIC share of international trips that go to simulated countries

    # media
    media_use_days_per_week: float = 0.6  # share of days a weekly user consumes the channel
    media_attention: float = 0.09  # chance a consumption session includes a salience-1 item
    media_salience_power: float = 3.0  # attention falls steeply for less salient items
    k_media: float = 0.15  # belief pull per repeated exposure
    # social feeds (real posts and public figures' posts on social channels only; HEURISTIC, 0 disables)
    social_interest_weight: float = 1.0  # exposure x clip(1 + w*(interest*N_TOPICS - 1), 0.25, 3): mean-preserving
    # exposure to a public figure's posts x clip(base + trust, 0.5, 1.5): with camp-based trust about 1.4 for
    # supporters and 0.6 for opponents, near the 70:30 aligned:cross-cutting exposure in Bakshy et al. 2015
    social_affinity_base: float = 0.5
    # reposting by ordinary people (sim/social.py; all HEURISTIC, unsourced)
    social_reposts: bool = True
    repost_b0: float = -3.5  # a fully convinced average person about 15%, a typical believer about 5%
    repost_visibility: float = 0.3  # per unit of log10 audience above 1
    repost_min_conviction: float = 0.3
    repost_window_days: int = 7  # the one day a person may repost an item falls in this window after hearing it
    repost_tail_days: int = 3  # a repost keeps the item on the social channel this long
    repost_bump_cap: float = 0.5  # most a push's intensity can rise in one day
    repost_max_life_days: int = 30  # reposts never keep an item tracked longer than this after its creation

    # sharing / acceptance (logistic coefficients)
    share_min_age: float = 8.0  # no product/news advice below this age
    share_full_age: float = 16.0  # full sharing propensity from this age
    share_b0: float = -4.6
    share_novelty_days: float = 21.0  # buzz decays exp(-days since heard/adopted / this)
    share_conviction: float = 1.8
    share_extraversion: float = 0.35
    share_interest: float = 0.25  # per unit of interest*N_TOPICS above 1
    share_adopted: float = 0.8
    accept_b0: float = -0.3
    accept_trust: float = 3.0
    accept_openness: float = 0.3
    accept_undecided: float = 0.8
    bounded_eps_base: float = 0.35
    bounded_eps_openness: float = 0.30
    k_wom: float = 0.5
    discernment_pull: float = 0.4
    distort_exaggeration: float = 1.6

    # news, mood and claims
    sentiment_rate: float = 0.1  # daily pull of mood toward the news an agent knows
    k_claim: float = 0.5  # how far a claim about a product moves the view of that product
    adopt_sentiment: float = 1.5  # bad mood delays purchases

    # people's trading in US stocks (sim/market.py); the values come from data/global/investors.yaml, copied
    # into each world when it is generated; these override them when set
    market_people: bool = True  # False: only announcements and scripted paths move the index
    market_kappa: float | None = None  # share of holdings moved per unit change of mood (fitted)
    market_multiplier: float | None = None  # price impact of net flows (Gabaix and Koijen 2021: about 5)
    market_requires_internet: bool | None = None  # HEURISTIC: only holders with internet trade daily

    # event odds (sim/questions.py; all HEURISTIC, unsourced)
    # log-odds shift per unit of a story's direction for someone who knows the question: a story rated +1
    # multiplies their odds of Yes by e (about 2.7)
    k_odds: float = 1.0
    odds_initial_sd: float = 0.5  # spread of traders' starting beliefs around the market price, on log-odds
    odds_partisan_shift: float = 1.0  # log-odds offset per unit of political lean x a question's partisanship

    # decisions
    adopt_b0: float = -8.6
    adopt_belief: float = 3.0
    adopt_afford: float = 8.0  # per unit of price / monthly income
    adopt_peer: float = 2.0
    adopt_risk: float = 1.0
    adopt_fit: float = 1.0
    adopt_interest: float = 0.3
    min_adopt_age: float = 16.0
    outcome_lag_days: int = 14
    outcome_fit_weight: float = 0.15
    outcome_noise_sd: float = 0.15

    # trust
    trust_k0: float = 4.0  # prior pseudo-count strength
    halo_fade_outcomes: float = 3.0
    negativity_weight: float = 2.5
    evidence_decay_per_day: float = 0.995  # toward the prior, applied lazily on update

    # time
    sleep_start_local: int = 23
    sleep_end_local: int = 7
