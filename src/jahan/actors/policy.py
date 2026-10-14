"""Role policies -> engine events.  No LLM here.

The actor chooses the policy and its own parameter (a tariff rate, a rate change in basis points); the
effect size comes from ``data/roles.yaml`` (``per_unit``, placeholder elasticities).  A policy the role
does not have is refused (the CLI logs ACTION_REJECTED).  Every policy is also announced as national news
from the actor, so people hear about it the same way they hear about anything else.
"""

from __future__ import annotations

from typing import Any

from jahan.actors.actions import Claim, Post, PriceChange
from jahan.actors.engine import display_name, notable_entry
from jahan.model.world import World
from jahan.scenario.spec import MacroPath, MacroStep, MediaPush, ProductLaunch


class PolicyError(ValueError):
    pass


def default_utc_hour(world: World, row: int, local_hour: int = 10) -> int:
    """UTC bin of a local hour for this agent (same rotation as the availability masks)."""
    off = int(world.pop["tz_offset_min"][row])
    return int((local_hour - (-((-off) // 60))) % 24)


def allowed_policies(world: World, agent_id: int) -> list[str]:
    entry = notable_entry(world, world.agent_row(agent_id))
    if entry is None or entry["role"] not in world.roles:
        return []
    return sorted(world.roles[entry["role"]].policies)


def compile_policy(
    world: World,
    actor: int,
    kind: str,
    params: dict[str, Any],
    claim: Claim | None = None,
    session: int = -1,
    utc_hour: int | None = None,
) -> list[Any]:
    """Events for ``kind`` by ``actor``.  ``claim`` is required for statements and directives."""
    row = world.agent_row(actor)
    entry = notable_entry(world, row)
    if entry is None:
        raise PolicyError("only public figures with a role can make policy (see `jahan create-agent --notable-role`)")
    role = world.roles.get(entry["role"])
    if role is None or kind not in role.policies:
        allowed = sorted(role.policies) if role else []
        raise PolicyError(f"{entry['office']} cannot {kind}; allowed: {allowed}")
    spec = role.policies[kind]
    hour = default_utc_hour(world, row) if utc_hour is None else utc_hour
    home = world.countries[int(entry["country"])].iso3
    codes = [c.iso3 for c in world.countries]
    name = display_name(world, row)
    events: list[Any] = []

    def announce(headline: str, stance: float, topic: str = "economy") -> Post:
        return Post(
            actor=actor,
            audience="national",
            claim=Claim(headline=headline[:300], topic=topic, stance=stance),
            utc_hour=hour,
            session=session,
        )

    if spec.effect == "news":
        if claim is None:
            raise PolicyError(f"{kind} needs the text of the statement")
        audience = str(params.get("audience", "national"))
        if audience not in ("national", "global"):
            raise PolicyError("audience must be national or global")
        return [
            Post(
                actor=actor,
                audience=audience,
                claim=claim,
                utc_hour=hour,
                session=session,
                salience_bonus=spec.salience_bonus,
            )
        ]

    def param_value() -> float:
        p = spec.param
        if p is None:
            raise PolicyError(f"{kind} is missing its parameter in roles.yaml")
        if p.name not in params:
            raise PolicyError(f"{kind} needs {p.name}= (between {p.min:g} and {p.max:g})")
        try:
            v = float(params[p.name])
        except (TypeError, ValueError) as e:
            raise PolicyError(f"{p.name} must be a number") from e
        if not p.min <= v <= p.max:
            raise PolicyError(f"{p.name} must be between {p.min:g} and {p.max:g}")
        return v

    if spec.effect == "macro":
        value = param_value()
        pct = value * spec.per_unit
        assert spec.param is not None and spec.variable is not None
        if spec.scope == "target_country":
            target = str(params.get("target", "")).upper()
            if target not in codes or target == home:
                raise PolicyError(f"{kind} needs target= one of {[c for c in codes if c != home]}")
            country = target
        else:
            country = home
        if spec.variable == "fx_local_per_usd" and world.countries[codes.index(country)].currency == "USD":
            raise PolicyError(f"{country} uses the US dollar; there is no exchange rate to move")
        events.append(
            MacroPath(
                variable=spec.variable,
                country=country,
                steps=[MacroStep(start_day_offset=0, days=spec.days, pct_per_day=pct)],
                auto_news=True,
            )
        )
        on = f" on {country}" if spec.scope == "target_country" else ""
        what = kind.replace("_", " ")
        if spec.variable == "sentiment":
            stance = 0.5 if pct > 0 else (-0.5 if pct < 0 else 0.0)
        elif spec.scope == "own_country":
            stance = -0.5 if pct > 0 else (0.5 if pct < 0 else 0.0)  # a weaker own currency is bad news at home
        else:
            stance = 0.0
        events.append(announce(f"{name} announces {what}{on} ({spec.param.name} {value:g})", stance))
        return events

    if spec.effect == "product_price":
        product = str(params.get("product", ""))
        pct = param_value()
        if world.macro.get("product_owner", {}).get(product) != actor:
            raise PolicyError(f"{name} can only change the price of products they launched")
        events.append(PriceChange(actor=actor, product=product, pct=pct, session=session))
        stance = 0.5 if pct < 0 else -0.5
        events.append(announce(f"{name} changes the price of {product} by {pct:+g}%", stance, "tech"))
        return events

    if spec.effect == "product":
        try:
            launch = ProductLaunch(
                name=str(params["name"]),
                topic=str(params.get("topic", "tech")),
                true_quality=float(params.get("quality", 0.6)),
                price_usd=float(params["price_usd"]),
                countries=[c.strip().upper() for c in str(params.get("countries", home)).split(",")],
                owner_agent=actor,
                pushes=[MediaPush(channel_kinds=["online", "social", "tv"], duration_days=30)],
            )
        except KeyError as e:
            raise PolicyError(
                f"product_launch needs {e.args[0]}= (name, price_usd; optional quality, countries)"
            ) from e
        except ValueError as e:
            raise PolicyError(f"product_launch: {e}") from e
        if launch.name in world.product_names:
            raise PolicyError(f"a product named {launch.name!r} already exists")
        bad = [c for c in launch.countries if c not in codes]
        if bad:
            raise PolicyError(f"countries not in this world: {bad}")
        return [launch]

    raise PolicyError(f"unsupported policy effect {spec.effect}")  # pragma: no cover
