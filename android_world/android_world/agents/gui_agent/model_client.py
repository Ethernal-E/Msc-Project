# android_world/agents/gui_agent/model_client.py
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from android_world.agents.gui_agent import prompts
from android_world.agents.gui_agent.schemas import (
    HistoryStep,
    ObservationFeedback,
    Particle,
)
from android_world.agents.gui_agent.utils import (
    clamp01,
    extract_json_object,
)


class Qwen3VLClient:
    """
    Thin wrapper around Qwen3-VL-4B-Instruct.

    It performs:
    - multimodal generation
    - proposal generation
    - self-verification/scoring
    - transition verification

    It does NOT implement particle search.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_new_tokens: int = 768,
    ):
        self.model_name = model_name

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )

        self.model.eval()

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

        # Useful later for compute/latency experiments.
        self.model_calls = 0
        self.generated_tokens = 0

    @staticmethod
    def _to_pil(image: np.ndarray | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected numpy image or PIL image, got {type(image)}"
            )

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        return Image.fromarray(image).convert("RGB")

    def generate_text(
        self,
        prompt: str,
        image: np.ndarray | Image.Image,
        *,
        do_sample: bool,
        max_new_tokens: int | None = None,
    ) -> str:

        image = self._to_pil(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": do_sample,
        }

        if do_sample:
            generation_kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )

        self.model_calls += 1

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[-1]

        generated_only = generated_ids[:, input_length:]

        self.generated_tokens += int(generated_only.shape[-1])

        output = self.processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output.strip()

    def generate_json(
        self,
        prompt: str,
        image: np.ndarray | Image.Image,
        *,
        do_sample: bool,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:

        text = self.generate_text(
            prompt,
            image,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )

        return extract_json_object(text)

    def generate_proposal(
        self,
        *,
        goal: str,
        image: np.ndarray,
        ui_elements_text: str,
        history: list[HistoryStep],
        horizon: int,
        observation_feedback: ObservationFeedback | None,
        parent: Particle | None = None,
    ) -> dict[str, Any]:

        prompt = prompts.build_proposal_prompt(
            goal=goal,
            ui_elements=ui_elements_text,
            history=history,
            horizon=horizon,
            observation_feedback=observation_feedback,
            parent=parent,
        )

        # Proposal generation intentionally samples.
        result = self.generate_json(
            prompt,
            image,
            do_sample=True,
            max_new_tokens=768,
        )

        if isinstance(result.get("action"), dict):
            result["action"] = self._normalize_action(
                result["action"]
            )

        return result
        

    
    
    def score_particles(
        self,
        *,
        goal: str,
        image: np.ndarray,
        ui_elements_text: str,
        history: list[HistoryStep],
        particles: list[Particle],
        grounding_scores: dict[str, float],
        repetition_penalties: dict[str, float],
        effectiveness_priors: dict[str, float],
        observation_feedback: ObservationFeedback | None,
    ) -> dict[str, dict[str, float]]:

        prompt = prompts.build_population_scoring_prompt(
            goal=goal,
            ui_elements=ui_elements_text,
            history=history,
            particles=particles,
            grounding_scores=grounding_scores,
            repetition_penalties=repetition_penalties,
            effectiveness_priors=effectiveness_priors,
            observation_feedback=observation_feedback,
        )

        result = self.generate_json(
            prompt,
            image,
            do_sample=False,
            max_new_tokens=640,
        )

        output: dict[str, dict[str, float]] = {}

        for item in result.get("ranking", []):
            particle_id = item.get("particle_id")

            if particle_id is None:
                continue

            output[particle_id] = {
                "current_action_quality": clamp01(item.get("current_action_quality", 0.0)),
                "goal": clamp01(item.get("goal", 0.0)),
                "history": clamp01(item.get("history", 0.0)),
                "future": clamp01(item.get("future", 0.0)),
                "recovery": clamp01(item.get("recovery", 0.0)),
                "confidence": clamp01(item.get("confidence", 0.0)),
                
            }

        return output
    def _normalize_action(
        self,
        action: dict[str, Any],
    ) -> dict[str, Any]:

        action = dict(action)

        if action.get("action_type") != "status":
            return action

        goal_status = action.get("goal_status")

        # Accept a few harmless synonymous outputs.
        if goal_status is None:
            goal_status = action.get("status")

        if isinstance(goal_status, str):
            normalized = goal_status.strip().lower()

            if normalized in {
                "complete",
                "completed",
                "done",
                "success",
                "successful",
                "finished",
            }:
                action["goal_status"] = "complete"

            elif normalized in {
                "infeasible",
                "impossible",
                "failed",
                "cannot_complete",
            }:
                action["goal_status"] = "infeasible"

        # Remove accidental alternate field.
        action.pop("status", None)

        return action

    def compare_observation(
        self,
        *,
        goal: str,
        previous_particle: Particle,
        image: np.ndarray,
        current_ui_elements_text: str,
    ) -> ObservationFeedback:

        prompt = prompts.build_observation_check_prompt(
            goal=goal,
            previous_particle=previous_particle,
            current_ui_elements=current_ui_elements_text,
        )

        result = self.generate_json(
            prompt,
            image,
            do_sample=False,
            max_new_tokens=256,
        )

        consistency = clamp01(result.get("consistency", 0.0))

        anomaly = bool(result.get("anomaly", consistency < 0.5))

        summary = str(
            result.get(
                "summary",
                "No transition summary was produced.",
            )
        )

        return ObservationFeedback(
            consistency=consistency,
            anomaly=anomaly,
            summary=summary,
        )