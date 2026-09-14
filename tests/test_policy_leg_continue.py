"""Ending a policy leg WITHOUT ending the rollout ("continue to the next phase", SIGUSR2).

A BC policy leg has no termination signal, so it runs out `hitl.policy_max_steps` whatever it is
doing. The only way to cut one short used to be Preempt, which unwinds the whole rollout -- and on a
`tamp -> policy -> tamp` plan that discards the final tamp leg the operator was waiting for. SIGUSR2
ends the leg alone and the plan carries on.

    tiptop/.pixi/envs/default/bin/python -m pytest tiptop/tests/test_policy_leg_continue.py -q
"""

from __future__ import annotations

import signal
from unittest import mock

import pytest

from tiptop import tiptop_run as tr
from tiptop.hitl.config import HITLConfig
from tiptop.hitl.planning import initial_state_for
from tiptop.hitl.proposal import parse_plan_response
from tiptop.hitl.session import HITLSession

# robot -> human -> robot: the shape the button exists for. The middle phase is the policy's.
PLAN = {
    "new_predicates": [{"name": "IsOpen", "instructions": "the container {0} is open"}],
    "phases": [
        {"executor": "robot", "description": "clear the box",
         "atoms": [{"predicate": "On", "args": ["blue_toy", "table"]}]},
        {"executor": "human", "description": "open the box", "instructions": "Open the white_box.",
         "atoms": [{"predicate": "IsOpen", "args": ["white_box"]}]},
        {"executor": "robot", "description": "put the toy in the box",
         "atoms": [{"predicate": "On", "args": ["blue_toy", "white_box"]}]},
    ],
}


def _session(cfg: HITLConfig) -> HITLSession:
    spec = parse_plan_response(PLAN, "open the box and fill it", ["blue_toy", "white_box"], "table", 1)
    return HITLSession(
        cfg=cfg, instruction=spec.instruction, trajectory_id="traj-1", spec=spec,
        initial_state=initial_state_for(spec.scene_types),
    )


class FakeLeg:
    """A policy leg that is alive until it is signalled."""

    def __init__(self):
        self.signals: list[int] = []
        self.alive = True

    def poll(self):
        return None if self.alive else 0

    def send_signal(self, sig):
        self.signals.append(sig)
        self.alive = False


@pytest.fixture(autouse=True)
def _clean_globals():
    """The flags are module globals, so leave them as they were found."""
    yield
    tr._policy_leg_proc = None
    tr._policy_leg_finished_by_operator = False


def test_it_signals_the_leg_and_only_the_leg():
    leg = FakeLeg()
    tr._policy_leg_proc = leg
    with mock.patch.object(tr, "_emit_event") as emit:
        tr._sigusr2_finish_policy_leg(signal.SIGUSR2, None)
    # SIGINT, not SIGTERM: it is the one policy_capture handles by writing the frames it captured.
    assert leg.signals == [signal.SIGINT]
    assert any(c.args[0].get("event") == "policy_leg_finish_requested" for c in emit.call_args_list)
    assert tr._consume_policy_leg_finish() is True
    assert tr._consume_policy_leg_finish() is False, "one-shot: it must not skip the NEXT phase's check"


def test_it_does_nothing_at_all_with_no_leg_running():
    """It must never raise. The handler runs in whatever the main thread was doing -- planning,
    executing, sitting at a prompt -- and a stray press there is a no-op, not a preempt."""
    tr._policy_leg_proc = None
    tr._sigusr2_finish_policy_leg(signal.SIGUSR2, None)
    assert tr._consume_policy_leg_finish() is False

    dead = FakeLeg()
    dead.alive = False
    tr._policy_leg_proc = dead
    tr._sigusr2_finish_policy_leg(signal.SIGUSR2, None)
    assert dead.signals == [], "a leg that already exited is not signalled"
    assert tr._consume_policy_leg_finish() is False


def test_the_plan_advances_to_the_next_phase_without_verifying(monkeypatch):
    """The point of the button: the phase is done on the operator's word and the plan goes ON.

    Verification is deliberately skipped. It could only second-guess someone who just watched the
    leg, and a "not done" verdict would spend a `verify_retries` attempt by restarting the very leg
    they stopped -- so the rollout would never reach the final tamp phase.
    """
    cfg = HITLConfig(enabled=True, policy_type="diffusion", policy_checkpoint="/tmp/ckpt")
    session = _session(cfg)
    session.advance()  # park on the human phase, which policy_type hands to the policy
    assert not session.is_final_phase(), "phase 1 of 3 -- a tamp phase still follows"

    monkeypatch.setattr(tr, "_hitl_session", session)
    monkeypatch.setattr(tr, "_hitl_cfg", cfg)
    # The leg runs, and the operator presses "continue to the next phase" while it does.
    monkeypatch.setattr(tr, "_run_policy_phase", lambda *a, **k: tr._sigusr2_finish_policy_leg(0, None))
    tr._policy_leg_proc = FakeLeg()

    async def _must_not_verify(*a, **k):
        raise AssertionError("the classifier was consulted after the operator had already decided")

    monkeypatch.setattr("tiptop.hitl.grounding.verify_phase", _must_not_verify)
    events: list[dict] = []
    monkeypatch.setattr(tr, "_emit_event", events.append)

    import asyncio
    ended = asyncio.run(tr._hitl_human_phase(object(), session.current, can_finish=False, output_dir="/tmp"))

    assert ended is False, "the trajectory stays OPEN, so the final tamp leg still runs"
    assert session.index == 2 and not session.finished, "advanced onto the last phase"
    verified = [e for e in events if e.get("event") == "human_phase_verified"]
    assert verified and verified[-1]["ok"] and verified[-1]["skipped"], "recorded as not-checked, not as passed"
    assert verified[-1]["ended_by_operator"] is True


def test_a_leg_that_ends_on_its_own_is_still_verified(monkeypatch):
    """The button is the ONLY thing that skips the check: a leg that runs out its step budget is
    verified exactly as before, and a failed check still costs a retry."""
    cfg = HITLConfig(enabled=True, policy_type="diffusion", policy_checkpoint="/tmp/ckpt", verify_retries=0)
    session = _session(cfg)
    session.advance()

    monkeypatch.setattr(tr, "_hitl_session", session)
    monkeypatch.setattr(tr, "_hitl_cfg", cfg)
    monkeypatch.setattr(tr, "_run_policy_phase", lambda *a, **k: None)
    monkeypatch.setattr(tr, "_emit_event", lambda e: None)
    monkeypatch.setattr(tr, "_hitl_verification_image", lambda c: None)
    calls: list[int] = []

    async def _verify(*a, **k):
        calls.append(1)
        return True, []

    monkeypatch.setattr("tiptop.hitl.grounding.verify_phase", _verify)

    import asyncio
    ended = asyncio.run(tr._hitl_human_phase(object(), session.current, can_finish=False, output_dir="/tmp"))
    assert calls == [1], "the check ran"
    assert ended is False and session.index == 2
