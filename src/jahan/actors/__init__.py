"""Agents acting on purpose: people who create or control agents, and public figures on autopilot.

``actions`` holds the typed actions (they are scenario events, so the ledger, reset, experiments and
transcripts handle them like any prompt).  ``engine`` applies them inside the tick and never calls an
LLM; ``policy`` turns role policies into engine events with placeholder elasticities.  The LLM parts
(text to claim, in-character replies, notable autonomy, roster drafts) live in their own modules and
are only imported by the CLI.
"""
