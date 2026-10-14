"""LLM layer.  The deterministic core (sim, gen, model, store) never imports this package.

The LLM only ever produces scenario parameters, classifications or text; every state change is a typed
event produced by the engine.  All calls are recorded (request key -> response) so replays never call
the network, and a per-run budget stops spending at its cap.
"""
