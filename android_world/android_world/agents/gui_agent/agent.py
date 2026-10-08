# android_world/agents/gui_agent/agent.py

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from android_world.agents import base_agent
from android_world.agents.gui_agent.model_client import Qwen3VLClient
from android_world.agents.gui_agent.particle_search import ParticleSearch
from android_world.agents.gui_agent.schemas import (
    HistoryStep,
    ObservationFeedback,
    Particle,
    SearchConfig,
)
from android_world.agents.gui_agent.utils import (
    to_android_world_action,
    ui_elements_to_text,
    wait_for_human_transition,
)
from android_world.env import interface


class GUIGuidanceAgent(base_agent.EnvironmentInteractingAgent):
    """
    Human-in-the-loop GUI guidance agent using adaptive particle search.

    Default behaviour:
        - model chooses guidance
        - guidance is shown to the human
        - human performs the action
        - agent observes what actually happened

    For later automated AndroidWorld experiments:
        human_in_the_loop=False
    executes the selected first action automatically.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        *,
        model: Qwen3VLClient | None = None,
        config: SearchConfig | None = None,
        human_in_the_loop: bool = True,
        guidance_output_dir: str = "gui_guidance_overlays",
        name: str = "gui_particle_qwen3vl",
    ):
        super().__init__(
            env=env,
            name=name,
            transition_pause=None,
        )

        self.model = model or Qwen3VLClient()

        self.config = config or SearchConfig()

        self.searcher = ParticleSearch(
            model=self.model,
            config=self.config,
        )

        self.human_in_the_loop = human_in_the_loop
        self.guidance_output_dir = Path(guidance_output_dir)
        self.guidance_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history: list[HistoryStep] = []

        # Particle selected on the PREVIOUS real step.
        # Its expected outcome is evaluated when the next real state arrives.
        self.pending_particle: Particle | None = None

        self.latest_observation_feedback: (
            ObservationFeedback | None
        ) = None

        self.real_step_count = 0

    def reset(
        self,
        go_home_on_reset: bool = False,
    ) -> None:

        super().reset(go_home_on_reset)

        self.env.hide_automation_ui()

        self.history = []
        self.pending_particle = None
        self.latest_observation_feedback = None
        self.real_step_count = 0

    # ------------------------------------------------------------------
    # Previous-action verification
    # ------------------------------------------------------------------

    def _resolve_previous_transition(
        self,
        *,
        goal: str,
        current_state: interface.State,
        current_ui_text: str,
    ) -> ObservationFeedback | None:

        if self.pending_particle is None:
            return None

        feedback = self.model.compare_observation(
            goal=goal,
            previous_particle=self.pending_particle,
            image=current_state.pixels,
            current_ui_elements_text=current_ui_text,
        )

        if self.history:
            self.history[-1].observation_feedback = feedback

        logging.info(
            "Previous transition: consistency=%.3f anomaly=%s summary=%s",
            feedback.consistency,
            feedback.anomaly,
            feedback.summary,
        )

        return feedback

    # ------------------------------------------------------------------
    # Guidance presentation
    # ------------------------------------------------------------------

    def _display_guidance(
        self,
        particle: Particle,
        state: interface.State,
    ) -> str:
        """
        Create a desktop-side guidance screenshot.

        The current Android screenshot is copied, the selected target bbox
        is highlighted, and the human-facing instruction is rendered on top.

        Returns:
            Path to the saved guidance image.
        """

        pixels = state.pixels

        if not isinstance(pixels, np.ndarray):
            raise TypeError(
                f"Expected state.pixels to be np.ndarray, got {type(pixels)}"
            )

        if pixels.dtype != np.uint8:
            pixels = np.clip(
                pixels,
                0,
                255,
            ).astype(np.uint8)

        # AndroidWorld screenshot -> PIL image.
        image = Image.fromarray(
            pixels
        ).convert("RGB")

        # Use RGBA so that the instruction banner can be transparent.
        image = image.convert("RGBA")

        overlay = Image.new(
            "RGBA",
            image.size,
            (0, 0, 0, 0),
        )

        draw = ImageDraw.Draw(overlay)

        width, height = image.size

        # ----------------------------------------------------------
        # 1. Draw the target bounding box.
        # ----------------------------------------------------------

        if particle.bbox is not None:
            x1, y1, x2, y2 = map(
                int,
                particle.bbox,
            )

            # Keep bbox inside screenshot bounds.
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(255, 0, 0, 255),
                width=8,
            )

        # ----------------------------------------------------------
        # 2. Draw instruction banner.
        # ----------------------------------------------------------

        # Larger font for human-readable guidance.
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                size=42,
            )
        except OSError:
            font = ImageFont.load_default()

        margin = 40
        padding = 28
        max_text_width = width - 2 * margin - 2 * padding

        

        words = particle.instruction.split()

        lines: list[str] = []
        current_line = ""

        for word in words:
            candidate = (
                f"{current_line} {word}".strip()
            )

            bbox = draw.textbbox(
                (0, 0),
                candidate,
                font=font,
            )

            text_width = bbox[2] - bbox[0]

            if (
                text_width > max_text_width
                and current_line
            ):
                lines.append(current_line)
                current_line = word
            else:
                current_line = candidate

        if current_line:
            lines.append(current_line)

        # Limit extremely long model instructions.
        lines = lines[:4]

        sample_bbox = draw.textbbox(
            (0, 0),
            "Ag",
            font=font,
        )

        font_height = sample_bbox[3] - sample_bbox[1]
        line_height = font_height + 12

        

        banner_height = (
            padding * 2
            + line_height * len(lines)
        )
        banner_top = 100

        draw.rounded_rectangle(
            [
                margin,
                banner_top,
                width - margin,
                banner_top + banner_height,
            ],
            radius=16,
            fill=(0, 0, 0, 190),
        )

        text_y = banner_top + padding

        for line in lines:
            draw.text(
                (
                    margin + padding,
                    text_y,
                ),
                line,
                fill=(255, 255, 255, 255),
                font=font,
            )

            text_y += line_height

        # ----------------------------------------------------------
        # 3. Merge overlay onto screenshot.
        # ----------------------------------------------------------

        image = Image.alpha_composite(
            image,
            overlay,
        ).convert("RGB")

        # ----------------------------------------------------------
        # 4. Save guidance screenshot.
        # ----------------------------------------------------------

        output_path = (
            self.guidance_output_dir
            / f"step_{self.real_step_count:03d}_guidance.png"
        )

        image.save(output_path)

        logging.info(
            "Saved human guidance overlay to %s",
            output_path,
        )

        # Keep the existing AndroidWorld message as a text fallback.
        #self.env.display_message(
            #particle.instruction,
            #header="GUI Assistant",
        #)

        return str(output_path)
    # ------------------------------------------------------------------
    # Main AndroidWorld interface
    # ------------------------------------------------------------------

    def step(
        self,
        goal: str,
    ) -> base_agent.AgentInteractionResult:

        # --------------------------------------------------------------
        # Stop on our own real interaction budget.
        # --------------------------------------------------------------

        max_steps = (
            self._max_steps
            if self._max_steps is not None
            else self.config.max_real_steps
        )

        if self.real_step_count >= max_steps:
            return base_agent.AgentInteractionResult(
                done=True,
                data={
                    "termination_reason": "max_real_steps_exceeded",
                    "history": [
                        h.as_dict()
                        for h in self.history
                    ],
                },
            )

        # --------------------------------------------------------------
        # 1. Observe current REAL Android state.
        # --------------------------------------------------------------

        state = self.get_post_transition_state()

        ui_text = ui_elements_to_text(
            state.ui_elements
        )

        # --------------------------------------------------------------
        # 2. Compare actual state with previous expected outcome.
        # --------------------------------------------------------------

        observation_feedback = (
            self._resolve_previous_transition(
                goal=goal,
                current_state=state,
                current_ui_text=ui_text,
            )
        )

        self.latest_observation_feedback = (
            observation_feedback
        )

        # Previous prediction has now been resolved.
        self.pending_particle = None

        # --------------------------------------------------------------
        # 3. Adaptive particle search for ONE next action.
        # --------------------------------------------------------------

        search_result = self.searcher.search(
            goal=goal,
            image=state.pixels,
            ui_elements=state.ui_elements,
            ui_elements_text=ui_text,
            history=self.history,
            observation_feedback=observation_feedback,
        )

        selected = search_result.selected

        logging.info(
            "Selected particle %s score=%.3f action=%s",
            selected.particle_id,
            selected.scores.total,
            selected.action,
        )

        # --------------------------------------------------------------
        # 4. Termination action proposed by model.
        # --------------------------------------------------------------

        if selected.action.get("action_type") == "status":

            goal_status = selected.action.get(
                "goal_status"
            )

            if goal_status not in {
                "complete",
                "infeasible",
            }:
                raise ValueError(
                    "Selected malformed status action: "
                    f"{selected.action}"
                )

            return base_agent.AgentInteractionResult(
                done=True,
                data={
                    "goal_status": goal_status,
                    "search": search_result.as_dict(),
                    "history": [
                        h.as_dict()
                        for h in self.history
                    ],
                    "model_calls": self.model.model_calls,
                    "generated_tokens": self.model.generated_tokens,
                },
            )

        # --------------------------------------------------------------
        # 5. Record CURRENT guidance before interaction.
        # --------------------------------------------------------------

        self.real_step_count += 1

        history_step = HistoryStep(
            step_number=self.real_step_count,
            instruction=selected.instruction,
            action=selected.action,
            expected_outcome=selected.expected_outcome,
            future_steps=selected.future_steps,
            score=selected.scores.total,
        )

        self.history.append(history_step)

        # The next observed screen will verify this particle.
        self.pending_particle = selected

        # --------------------------------------------------------------
        # 6A. HUMAN-IN-THE-LOOP MODE
        # --------------------------------------------------------------

        if self.human_in_the_loop:

            guidance_image_path = self._display_guidance(
                particle=selected,
                state=state,
            )
            print("\n" + "=" * 80)
            print("HUMAN GUIDANCE")
            print("=" * 80)
            print(f"Instruction: {selected.instruction}")
            print(f"BBox: {selected.bbox}")
            print(f"Guidance image: {guidance_image_path}")
            print("Waiting for human action...")
            print("=" * 80 + "\n")

            

            # The human performs exactly ONE physical action.
            #
            # They are free to:
            # - perform the suggested action,
            # - make a mistake,
            # - perform another action.
            #
            # We do NOT force the action here.
            logging.info(
                "Waiting for human action..."
            )

            changed = False

            while not changed:
                changed = wait_for_human_transition(
                    env=self.env,
                    before_state=state,
                    timeout_seconds=(
                        self.config.human_wait_timeout_seconds
                    ),
                    poll_interval_seconds=(
                        self.config.human_poll_interval_seconds
                    ),
                )

                if not changed:
                    logging.info(
                        "No human transition detected yet. "
                        "Continuing to wait..."
                    )
            
            return base_agent.AgentInteractionResult(
                done=False,
                data={
                    "mode": "human_in_the_loop",
                    "human_transition_detected": changed,
                    "selected_instruction": selected.instruction,
                    "selected_bbox": selected.bbox,
                    "selected_action": selected.action,
                    "search": search_result.as_dict(),
                    "model_calls": self.model.model_calls,
                    "generated_tokens": self.model.generated_tokens,
                    "guidance_image_path": guidance_image_path,
                },
            )

        # --------------------------------------------------------------
        # 6B. AUTOMATED EXPERIMENT MODE
        # --------------------------------------------------------------

        android_action = to_android_world_action(
            selected.action
        )

        self.env.execute_action(
            android_action
        )

        return base_agent.AgentInteractionResult(
            done=False,
            data={
                "mode": "automatic",
                "selected_instruction": selected.instruction,
                "selected_bbox": selected.bbox,
                "selected_action": selected.action,
                "search": search_result.as_dict(),
                "model_calls": self.model.model_calls,
                "generated_tokens": self.model.generated_tokens,
            },
        )