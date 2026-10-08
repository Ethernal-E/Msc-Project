# android_world/agents/gui_agent/schemas.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreBreakdown:
    """Component scores for one particle."""

    goal: float = 0.0
    grounding: float = 0.0
    history: float = 0.0
    future: float = 0.0
    recovery: float = 0.0
    risk: float = 0.0
    confidence: float = 0.0

    # NEW
    current_action_quality: float = 0.0
    repetition_penalty: float = 0.0
    effectiveness_prior: float = 0.0

    # Whether the predicted consequence of this particle matched
    # a later real observation.
    observation_consistency: float | None = None

    total: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "grounding": self.grounding,
            "history": self.history,
            "future": self.future,
            "recovery": self.recovery,
            "risk": self.risk,
            "confidence": self.confidence,
            "observation_consistency": self.observation_consistency,
            "total": self.total,
            "current_action_quality": self.current_action_quality,
            "repetition_penalty": self.repetition_penalty,
            "effectiveness_prior": self.effectiveness_prior,
        }


@dataclass
class Particle:
    """
    One GUI hypothesis.

    IMPORTANT:
    - action is the ONLY action that may actually be executed.
    - future_steps are internal symbolic lookahead only.
    """

    particle_id: str

    reasoning: str

    # Human-facing guidance for the current action.
    instruction: str

    # AndroidWorld-compatible action.
    action: dict[str, Any]

    # What the model expects immediately after the current action.
    expected_outcome: str

    # Symbolic future lookahead.
    # These actions are NOT automatically executed.
    future_steps: list[str] = field(default_factory=list)

    # Pixel bbox [x1, y1, x2, y2], filled from AndroidWorld accessibility
    # information where possible.
    bbox: list[int] | None = None

    scores: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    generation_round: int = 0
    parent_id: str | None = None

    # Used by power sampling.
    log_weight: float = 0.0

    raw_model_output: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "particle_id": self.particle_id,
            "reasoning": self.reasoning,
            "instruction": self.instruction,
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "future_steps": self.future_steps,
            "bbox": self.bbox,
            "scores": self.scores.as_dict(),
            "generation_round": self.generation_round,
            "parent_id": self.parent_id,
            "log_weight": self.log_weight,
        }


@dataclass
class ObservationFeedback:
    """
    Comparison between the previous particle prediction and
    the newly observed real GUI state.
    """

    consistency: float = 1.0
    anomaly: bool = False
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "consistency": self.consistency,
            "anomaly": self.anomaly,
            "summary": self.summary,
        }


@dataclass
class HistoryStep:
    step_number: int

    instruction: str
    action: dict[str, Any]
    expected_outcome: str
    future_steps: list[str]

    score: float

    # Filled when the next real observation becomes available.
    observation_feedback: ObservationFeedback | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "instruction": self.instruction,
            "action": self.action,
            "expected_outcome": self.expected_outcome,
            "future_steps": self.future_steps,
            "score": self.score,
            "observation_feedback": (
                self.observation_feedback.as_dict()
                if self.observation_feedback
                else None
            ),
        }


@dataclass
class SearchConfig:
    # ---------- Proposal ----------
    initial_particles: int = 4

    # Symbolic lookahead depth.
    lookahead_horizon: int = 3

    # Maximum number of proposal particles generated in ONE real GUI step.
    proposal_budget: int = 16

    # Maximum ASMC/power-sampling rounds inside one GUI step.
    max_rounds: int = 3

    # Number of top particles deterministically retained between rounds.
    elite_count: int = 2

    # ---------- Power sampling ----------
    # p_i proportional to weight_i ** power_alpha.
    power_alpha: float = 2.0

    # ---------- Adaptive resampling ----------
    low_confidence_threshold: float = 0.72

    # If top two particles are too close, continue searching.
    min_top_margin: float = 0.08

    # Fraction of distinct current actions that indicates strong disagreement.
    disagreement_threshold: float = 0.50

    # ---------- Main score weights ----------
    goal_weight: float = 0.35
    grounding_weight: float = 0.25
    history_weight: float = 0.20
    future_weight: float = 0.20

    # ---------- Error-state weights ----------
    error_goal_weight: float = 0.25
    error_grounding_weight: float = 0.15
    error_history_weight: float = 0.25
    error_future_weight: float = 0.15
    error_recovery_weight: float = 0.20

    # Risk is always a penalty.
    risk_penalty: float = 0.15

    # ---------- Interaction ----------
    max_real_steps: int = 30

    # Human-in-the-loop mode waits for the physical GUI to change.
    human_wait_timeout_seconds: float = 60.0
    human_poll_interval_seconds: float = 0.5


@dataclass
class SearchResult:
    selected: Particle
    particles: list[Particle]

    rounds_used: int
    proposals_generated: int

    resampling_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.as_dict(),
            "particles": [p.as_dict() for p in self.particles],
            "rounds_used": self.rounds_used,
            "proposals_generated": self.proposals_generated,
            "resampling_reasons": self.resampling_reasons,
        }