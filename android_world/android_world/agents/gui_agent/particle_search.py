# android_world/agents/gui_agent/particle_search.py

from __future__ import annotations

import math
import uuid

import numpy as np

from android_world.agents.gui_agent.model_client import Qwen3VLClient

from android_world.agents.gui_agent.schemas import (
    HistoryStep,
    ObservationFeedback,
    Particle,
    ScoreBreakdown,
    SearchConfig,
    SearchResult,
)
from android_world.agents.gui_agent.utils import (
    action_effectiveness_prior,
    action_signature,
    clamp01,
    deterministic_grounding_score,
    get_action_bbox,
    repetition_penalty,
)
from android_world.env import representation_utils


class ParticleSearch:
    """
    Adaptive test-time particle search for GUI guidance.

    No GUI actions are executed here.

    This class only decides:
        "What should the ONE next action be?"
    """

    def __init__(
        self,
        model: Qwen3VLClient,
        config: SearchConfig | None = None,
        random_seed: int = 0,
    ):
        self.model = model
        self.config = config or SearchConfig()

        self.rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Particle construction
    # ------------------------------------------------------------------

    def _make_particle(
        self,
        raw: dict,
        *,
        generation_round: int,
        parent_id: str | None,
        ui_elements: list[representation_utils.UIElement],
    ) -> Particle:

        action = raw.get("action", {})

        if not isinstance(action, dict):
            action = {}

        action_type = action.get("action_type")
        # Validate text-input affordance.
        #
        # input_text is only valid for UI elements that actually support editing.
        # This prevents the model from treating arbitrary text/display elements
        # as editable fields.
        if action_type == "input_text":
            index = action.get("index")

            if (
                not isinstance(index, int)
                or index < 0
                or index >= len(ui_elements)
            ):
                raise ValueError(
                    f"Invalid input_text target index: {index}"
                )

            target = ui_elements[index]

            if not target.is_editable:
                raise ValueError(
                    "Invalid input_text action: target UI element "
                    f"{index} is not editable."
                )

        if action_type == "status":
            goal_status = action.get("goal_status")

            if goal_status not in {"complete", "infeasible"}:
                raise ValueError(
                    "Invalid status action: goal_status must be "
                    "'complete' or 'infeasible'. "
                    f"Received: {action}"
                )

        future_steps = raw.get("future_steps", [])

        if not isinstance(future_steps, list):
            future_steps = []

        # Never allow the lookahead list to silently grow arbitrarily.
        future_steps = [
            str(x)
            for x in future_steps[: self.config.lookahead_horizon]
        ]

        particle = Particle(
            particle_id=uuid.uuid4().hex[:12],
            reasoning=str(raw.get("reasoning", "")),
            instruction=str(raw.get("instruction", "")),
            action=action,
            expected_outcome=str(raw.get("expected_outcome", "")),
            future_steps=future_steps,
            generation_round=generation_round,
            parent_id=parent_id,
            raw_model_output=raw,
        )

        particle.bbox = get_action_bbox(
            particle.action,
            ui_elements,
        )

        return particle

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_total_score(
        self,
        scores: ScoreBreakdown,
        error_state: bool,
    ) -> float:

        if error_state:

            total = (
                0.20 * scores.current_action_quality
                + 0.20 * scores.goal
                + 0.10 * scores.grounding
                + 0.10 * scores.history
                + 0.10 * scores.future
                + 0.20 * scores.recovery
                + 0.05 * scores.confidence
                + 0.05 * scores.effectiveness_prior
                - 0.25 * scores.repetition_penalty
            )

        else:

            total = (
                0.20 * scores.current_action_quality
                + 0.30 * scores.goal
                + 0.15 * scores.grounding
                + 0.10 * scores.history
                + 0.15 * scores.future
                + 0.05 * scores.confidence
                + 0.05 * scores.effectiveness_prior
                - 0.20 * scores.repetition_penalty
            )
        # ----------------------------------------------------------
        # Hard loop suppression
        #
        # Repeated actions are strongly downweighted after multiple
        # unsuccessful attempts, even if the self-scorer still assigns
        # high semantic scores.
        # ----------------------------------------------------------

        if scores.repetition_penalty >= 1.0:
            total *= 0.10

        elif scores.repetition_penalty >= 0.80:
            total *= 0.30

        return clamp01(total)

    

    def _score_population(
        self,
        *,
        particles: list[Particle],
        goal: str,
        image: np.ndarray,
        ui_elements: list[representation_utils.UIElement],
        ui_elements_text: str,
        history: list[HistoryStep],
        observation_feedback: ObservationFeedback | None,
    ) -> None:

        if not particles:
            return

        error_state = bool(
            observation_feedback is not None
            and observation_feedback.anomaly
        )

        # --------------------------------------------------------------
        # 1. Deterministic per-particle signals
        # --------------------------------------------------------------

        grounding_scores: dict[str, float] = {}
        repetition_penalties: dict[str, float] = {}
        effectiveness_priors: dict[str, float] = {}

        for particle in particles:

            grounding_scores[particle.particle_id] = (
                deterministic_grounding_score(
                    particle.action,
                    ui_elements,
                )
            )

            repetition_penalties[particle.particle_id] = (
                repetition_penalty(
                    particle.action,
                    history,
                )
            )

            effectiveness_priors[particle.particle_id] = (
                action_effectiveness_prior(
                    particle.action,
                )
            )

        # --------------------------------------------------------------
        # 2. ONE model call for the whole population
        # --------------------------------------------------------------

        semantic_scores = self.model.score_particles(
            goal=goal,
            image=image,
            ui_elements_text=ui_elements_text,
            history=history,
            particles=particles,
            grounding_scores=grounding_scores,
            repetition_penalties=repetition_penalties,
            effectiveness_priors=effectiveness_priors,
            observation_feedback=observation_feedback,
        )

        # --------------------------------------------------------------
        # 3. Attach scores
        # --------------------------------------------------------------

        for particle in particles:

            semantic = semantic_scores.get(
                particle.particle_id,
                {},
            )

            current_action_quality = clamp01(
                semantic.get(
                    "current_action_quality",
                    0.0,
                )
            )

            goal_score = clamp01(
                semantic.get(
                    "goal",
                    0.0,
                )
            )

            history_score = clamp01(
                semantic.get(
                    "history",
                    0.0,
                )
            )

            raw_future_score = clamp01(
                semantic.get(
                    "future",
                    0.0,
                )
            )

            recovery_score = clamp01(
                semantic.get(
                    "recovery",
                    0.0,
                )
            )

            confidence_score = clamp01(
                semantic.get(
                    "confidence",
                    0.0,
                )
            )

            grounding_score = grounding_scores[
                particle.particle_id
            ]

            repeat_penalty = repetition_penalties[
                particle.particle_id
            ]

            effectiveness = effectiveness_priors[
                particle.particle_id
            ]

            # ----------------------------------------------------------
            # IMPORTANT:
            #
            # Future score is GATED by current action quality.
            #
            # This prevents:
            #
            #   useless current click
            #       +
            #   beautiful imagined future
            #
            # from receiving an unfairly high score.
            # ----------------------------------------------------------

            gated_future_score = (
                raw_future_score
                * current_action_quality
            )

            scores = ScoreBreakdown(
                goal=goal_score,
                grounding=grounding_score,
                history=history_score,
                future=gated_future_score,
                recovery=recovery_score,
                confidence=confidence_score,
                current_action_quality=current_action_quality,
                repetition_penalty=repeat_penalty,
                effectiveness_prior=effectiveness,
            )

            scores.total = self._compute_total_score(
                scores,
                error_state,
            )

            particle.scores = scores

            particle.log_weight = math.log(
                max(scores.total, 1e-8)
            )
    # ------------------------------------------------------------------
    # Adaptive resampling decision
    # ------------------------------------------------------------------

    def _resampling_reasons(
        self,
        particles: list[Particle],
        observation_feedback: ObservationFeedback | None,
        round_index: int,
    ) -> list[str]:

        if not particles:
            return ["no_particles"]

        particles = sorted(
            particles,
            key=lambda p: p.scores.total,
            reverse=True,
        )

        reasons: list[str] = []

        top = particles[0]

        # 1. Low confidence.
        if (
            top.scores.total
            < self.config.low_confidence_threshold
        ):
            reasons.append("low_confidence")

        # 2. Top particles are too close.
        # 2. Best DISTINCT actions are too close.
        #
        # Multiple particles may propose the same action. That is consensus,
        # not uncertainty. Only compare the best scores of different actions.
        best_by_action: dict[tuple, float] = {}

        for particle in particles:
            signature = action_signature(particle.action)
            score = particle.scores.total

            if (
                signature not in best_by_action
                or score > best_by_action[signature]
            ):
                best_by_action[signature] = score

        distinct_scores = sorted(
            best_by_action.values(),
            reverse=True,
        )

        if len(distinct_scores) >= 2:
            margin = distinct_scores[0] - distinct_scores[1]

            if margin < self.config.min_top_margin:
                reasons.append("small_top_margin")
        

        # 3. Current actions disagree strongly.
        signatures = {
            action_signature(p.action)
            for p in particles
        }

        disagreement = len(signatures) / max(len(particles), 1)

        if disagreement >= self.config.disagreement_threshold:
            reasons.append("particle_disagreement")

        # 4. Real world violated the previous prediction.
        if (
            round_index == 0
            and observation_feedback is not None
            and observation_feedback.anomaly
        ):
            reasons.append("observed_transition_anomaly")

        return reasons

    # ------------------------------------------------------------------
    # Power sampling
    # ------------------------------------------------------------------

    def _power_sample_parents(
        self,
        particles: list[Particle],
        n: int,
    ) -> list[Particle]:

        if not particles or n <= 0:
            return []

        weights = np.array(
            [
                max(p.scores.total, 1e-8)
                for p in particles
            ],
            dtype=np.float64,
        )

        # This is the power-sampling step:
        #
        #       P(i) ∝ w_i ^ alpha
        #
        powered = np.power(
            weights,
            self.config.power_alpha,
        )

        powered_sum = powered.sum()

        if not np.isfinite(powered_sum) or powered_sum <= 0:
            probabilities = np.ones_like(powered) / len(powered)
        else:
            probabilities = powered / powered_sum

        sampled_indices = self.rng.choice(
            len(particles),
            size=n,
            replace=True,
            p=probabilities,
        )

        return [
            particles[int(i)]
            for i in sampled_indices
        ]

    # ------------------------------------------------------------------
    # Population utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_population(
        particles: list[Particle],
    ) -> list[Particle]:

        return sorted(
            particles,
            key=lambda p: p.scores.total,
            reverse=True,
        )

    @staticmethod
    def _deduplicate(
        particles: list[Particle],
    ) -> list[Particle]:
        """
        Remove exact duplicate action + expected outcome hypotheses,
        retaining the highest-score version.
        """

        particles = sorted(
            particles,
            key=lambda p: p.scores.total,
            reverse=True,
        )

        seen = set()
        output = []

        for particle in particles:
            key = (
                action_signature(particle.action),
                particle.expected_outcome.strip().lower(),
            )

            if key in seen:
                continue

            seen.add(key)
            output.append(particle)

        return output

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(
        self,
        *,
        goal: str,
        image: np.ndarray,
        ui_elements: list[representation_utils.UIElement],
        ui_elements_text: str,
        history: list[HistoryStep],
        observation_feedback: ObservationFeedback | None,
    ) -> SearchResult:

        config = self.config

        proposals_generated = 0
        rounds_used = 0
        all_reasons: list[str] = []

        # --------------------------------------------------------------
        # Round 0: broad initial exploration
        # --------------------------------------------------------------

        initial_count = min(
            config.initial_particles,
            config.proposal_budget,
        )

        population: list[Particle] = []

        for _ in range(initial_count):
            try:
                raw = self.model.generate_proposal(
                    goal=goal,
                    image=image,
                    ui_elements_text=ui_elements_text,
                    history=history,
                    horizon=config.lookahead_horizon,
                    observation_feedback=observation_feedback,
                    parent=None,
                )
            except Exception:
                # A malformed generation should not kill the whole search.
                continue

            particle = self._make_particle(
                raw,
                generation_round=0,
                parent_id=None,
                ui_elements=ui_elements,
            )

            population.append(particle)
            proposals_generated += 1

        if not population:
            raise RuntimeError(
                "Qwen did not produce any valid initial particles."
            )

        self._score_population(
            particles=population,
            goal=goal,
            image=image,
            ui_elements=ui_elements,
            ui_elements_text=ui_elements_text,
            history=history,
            observation_feedback=observation_feedback,
        )

        rounds_used = 1

        # --------------------------------------------------------------
        # Adaptive refinement / power resampling
        # --------------------------------------------------------------

        for round_index in range(1, config.max_rounds):

            reasons = self._resampling_reasons(
                population,
                observation_feedback,
                round_index=round_index - 1,
            )

            if not reasons:
                break

            all_reasons.extend(reasons)

            remaining_budget = (
                config.proposal_budget
                - proposals_generated
            )

            if remaining_budget <= 0:
                break

            population = self._sort_population(population)

            elites = population[
                : min(config.elite_count, len(population))
            ]

            # Generate approximately one fresh branch per initial particle,
            # unless compute budget is exhausted.
            n_children = min(
                config.initial_particles,
                remaining_budget,
            )

            parents = self._power_sample_parents(
                population,
                n_children,
            )

            children: list[Particle] = []

            for parent in parents:
                try:
                    raw = self.model.generate_proposal(
                        goal=goal,
                        image=image,
                        ui_elements_text=ui_elements_text,
                        history=history,
                        horizon=config.lookahead_horizon,
                        observation_feedback=observation_feedback,
                        parent=parent,
                    )
                except Exception:
                    continue

                child = self._make_particle(
                    raw,
                    generation_round=round_index,
                    parent_id=parent.particle_id,
                    ui_elements=ui_elements,
                )

                children.append(child)
                proposals_generated += 1

            if not children:
                break

            self._score_population(
                particles=children,
                goal=goal,
                image=image,
                ui_elements=ui_elements,
                ui_elements_text=ui_elements_text,
                history=history,
                observation_feedback=observation_feedback,
            )

            # Keep strong hypotheses while allowing new branches to compete.
            population = elites + children

            population = self._deduplicate(population)

            population = self._sort_population(population)

            rounds_used += 1

        population = self._sort_population(population)

        selected = population[0]

        return SearchResult(
            selected=selected,
            particles=population,
            rounds_used=rounds_used,
            proposals_generated=proposals_generated,
            resampling_reasons=list(dict.fromkeys(all_reasons)),
        )