# android_world/agents/gui_agent/prompts.py

from __future__ import annotations

import json

from android_world.agents.gui_agent.schemas import (
    HistoryStep,
    ObservationFeedback,
    Particle,
)


ACTION_SCHEMA = """
Allowed Android actions:

1. Click:
{"action_type": "click", "index": <ui_element_index>}

2. Long press:
{"action_type": "long_press", "index": <ui_element_index>}

3. Input or replace text:

Append/type text into a field:
{"action_type": "input_text",
 "index": <ui_element_index>,
 "text": "<text>",
 "clear_text": false}

Replace ALL existing text in a field:
{"action_type": "input_text",
 "index": <ui_element_index>,
 "text": "<correct complete text>",
 "clear_text": true}

IMPORTANT:
If a text field already contains incorrect, duplicated, stale, or unwanted text,
prefer input_text with clear_text=true instead of repeatedly clicking,
long-pressing, or manually trying to delete characters.

Example:
Current field contains "PereiraPereira"
Desired value is "Pereira"

Use:
{"action_type": "input_text",
 "index": 8,
 "text": "Pereira",
 "clear_text": true}

4. Scroll:
{"action_type": "scroll", "direction": "up|down|left|right"}

5. Navigate back:
{"action_type": "navigate_back"}

6. Navigate home:
{"action_type": "navigate_home"}

7. Open app:
{"action_type": "open_app", "app_name": "<app name>"}

8. Press keyboard enter:
{"action_type": "keyboard_enter"}

9. Wait:
{"action_type": "wait"}

10. Task complete:
{"action_type": "status", "goal_status": "complete"}

11. Task infeasible:
{"action_type": "status", "goal_status": "infeasible"}

IMPORTANT:
If action_type is "status", goal_status is REQUIRED.

Never output:
{"action_type": "status"}

Valid examples are ONLY:
{"action_type": "status", "goal_status": "complete"}
or
{"action_type": "status", "goal_status": "infeasible"}
"""


PROPOSAL_SYSTEM_PROMPT = """
You are a mobile GUI guidance model.

You DO NOT autonomously execute a long trajectory.

Your job is to:
1. Understand the user's overall task.
2. Understand the current Android GUI.
3. Reason about previous interactions and possible mistakes.
4. Propose exactly ONE action to perform now.
5. Internally look ahead a few future steps to judge whether this action
   leads toward the goal.

The future steps are predictions only.
Only the current action will be executed.

A previous mistake does NOT automatically imply that you must press Back.
Recovery, Back, continuing from the current page, searching for another route,
or taking an alternative route are all valid hypotheses and should compete
normally.


IMPORTANT TERMINATION RULE:

Judge whether the user's goal is satisfied by SEMANTIC MEANING, not by exact
surface-form or literal string equality.

Android applications may automatically change how values are displayed without
changing their underlying meaning.

Harmless presentation differences may include:
- spaces, punctuation, separators, or parentheses,
- capitalization,
- locale-specific formatting,
- date or time presentation,
- currency presentation,
- application-controlled formatting,
- equivalent labels or UI representations.

Do NOT modify a value merely because its displayed representation differs from
the exact wording or formatting used in the user's request.

Before proposing an action, consider whether the relevant part of the user's
goal may already be satisfied.

If a field, setting, or subgoal is already semantically correct, prefer making
progress on the remaining task rather than editing the already-correct state.

Only attempt to correct an existing value when there is meaningful evidence
that the underlying semantic value is actually wrong, missing, duplicated,
incomplete, or inconsistent with the user's goal.

Repeatedly editing an already-satisfied value is a failure mode and should be
avoided.

IMPORTANT SEQUENTIAL-INTERACTION RULE:

A GUI state does NOT need to equal the final requested value after every
individual action.

Some interfaces use sequential, cumulative, positional, or incremental input.
In these interfaces, one action may produce only a partial intermediate state.

Do NOT treat a partial state as an error merely because it does not yet equal
the final goal.

Before attempting recovery, consider whether the current state could be a
valid intermediate result of previous actions and whether continuing the
interaction sequence can still reach the user's goal.

This is especially important for interfaces such as:
- numeric keypads,
- PIN or code entry,
- timer and duration entry,
- date/time pickers,
- multi-digit numeric controls,
- incremental controls,
- multi-step forms.

For example, when a multi-digit value is entered one digit at a time, the
display after the first digit may represent only a partial prefix or
intermediate positional value. This is not by itself evidence of failure.

Only attempt recovery when there is meaningful evidence that the observed
state is incompatible with reasonable continued progress toward the goal.

IMPORTANT POSITIONAL NUMERIC INPUT RULE:

Some numeric interfaces do not expose separate editable fields for each unit.
Instead, they treat key presses as one continuous digit sequence and
right-align or shift digits across positions or units.

In such interfaces:
- the newest digit may initially appear in the lowest-order position,
- earlier digits may shift left or into higher-order units as more digits are
  entered,
- an intermediate display may temporarily look numerically different from the
  final requested value,
- this does NOT imply the earlier digits were entered into the wrong field.

Do not assume that visible hours/minutes/seconds, date components, or other
numeric units are independently editable unless the UI provides separate
editable controls.

When the interface appears to use a shared numeric keypad, consider whether
continued digit entry will shift or reinterpret the existing digits before
attempting correction or deletion.

You must return valid JSON only.

If you believe the user's task has been successfully completed, output:

{
  "action_type": "status",
  "goal_status": "complete"
}

If the task cannot be completed, output:

{
  "action_type": "status",
  "goal_status": "infeasible"
}

Never output a status action without goal_status.
"""


def _history_to_text(history: list[HistoryStep]) -> str:
    if not history:
        return "No previous interaction."

    rows = []
    for h in history[-8:]:
        feedback = ""
        if h.observation_feedback is not None:
            feedback = (
                f"\nObserved result: "
                f"{h.observation_feedback.summary} "
                f"(consistency={h.observation_feedback.consistency:.2f}, "
                f"anomaly={h.observation_feedback.anomaly})"
            )

        rows.append(
            f"Step {h.step_number}:\n"
            f"Instruction: {h.instruction}\n"
            f"Action: {json.dumps(h.action, ensure_ascii=False)}\n"
            f"Expected outcome: {h.expected_outcome}"
            f"{feedback}"
        )

    return "\n\n".join(rows)




def build_proposal_prompt(
    goal: str,
    ui_elements: str,
    history: list[HistoryStep],
    horizon: int,
    observation_feedback: ObservationFeedback | None = None,
    parent: Particle | None = None,
) -> str:

    observation_text = "No abnormal transition has been detected."

    if observation_feedback is not None:
        observation_text = (
            f"Previous transition consistency: "
            f"{observation_feedback.consistency:.2f}\n"
            f"Possible anomaly: {observation_feedback.anomaly}\n"
            f"Observation summary: {observation_feedback.summary}"
        )

    parent_text = ""
    if parent is not None:
        parent_text = f"""
This is a refinement/branching round.

A previously promising particle was:

{json.dumps(parent.as_dict(), ensure_ascii=False, indent=2)}

Use it as a hypothesis, NOT as guaranteed truth.

You may:
- preserve its current action,
- refine its reasoning,
- propose a better alternative,
- propose Back/recovery,
- continue from the current state,
- abandon the parent entirely if it appears wrong.
"""

    return f"""
{PROPOSAL_SYSTEM_PROMPT}

USER GOAL:
{goal}

CURRENT ACCESSIBILITY / UI ELEMENTS:
{ui_elements}

INTERACTION HISTORY:
{_history_to_text(history)}

LATEST REAL-OBSERVATION FEEDBACK:
{observation_text}

{parent_text}

{ACTION_SCHEMA}

Generate one candidate particle.

The particle must contain:
- reasoning:
  short explanation of the current situation and why this action is useful.
- instruction:
  concise human-facing instruction for ONLY the next action.
- action:
  exactly one AndroidWorld-compatible action.
- expected_outcome:
  Describe the expected SEMANTIC state change immediately after the current
  action.
  
  Focus on what should become true in the GUI, not on exact cosmetic formatting.
  
  Do not require exact spacing, punctuation, capitalization, separators,
  date/time formatting, or other application-controlled presentation unless
  those details are themselves essential to the user's goal.
- future_steps:
  up to {horizon} short SYMBOLIC future steps showing why the current action
  leads toward the goal.

Do not execute future_steps.
Do not assume future GUI states with certainty.
If the current page looks wrong, recovery actions and continuation actions
should both be considered possible hypotheses.

Return exactly:

{{
  "reasoning": "...",
  "instruction": "...",
  "action": {{
    "action_type": "..."
  }},
  "expected_outcome": "...",
  "future_steps": [
    "...",
    "..."
  ]
}}
"""


def build_scoring_prompt(
    goal: str,
    ui_elements: str,
    history: list[HistoryStep],
    particle: Particle,
    deterministic_grounding_score: float,
    observation_feedback: ObservationFeedback | None,
) -> str:

    anomaly = bool(
        observation_feedback is not None
        and observation_feedback.anomaly
    )

    return f"""
You are evaluating ONE candidate action for a mobile GUI guidance system.

USER GOAL:
{goal}

CURRENT UI:
{ui_elements}

HISTORY:
{_history_to_text(history)}

CURRENT STATE IS SUSPECTED TO BE AN ERROR/DEVIATION:
{anomaly}

CANDIDATE:
{json.dumps(particle.as_dict(), ensure_ascii=False, indent=2)}

A deterministic accessibility-based grounding checker already produced:
grounding_score = {deterministic_grounding_score:.3f}

Evaluate the candidate independently.

Scores must be between 0.0 and 1.0.

Definitions:

goal:
How strongly does the CURRENT action move toward the user's true goal?

history:
Does the CURRENT action make sense given what has already been tried and what
the current GUI now shows?

A high history score should reflect appropriate adaptation to new evidence.

A low history score should be given when:
- the same action has already been tried without useful progress,
- the candidate keeps editing something that already appears semantically
  correct,
- the candidate ignores evidence that the application has only changed the
  display representation rather than the underlying value.

future:
Are the symbolic future steps plausible, coherent, and goal-directed?

recovery:
If the current state is actually wrong, how strongly does the CURRENT action
repair it?

Do not give a high recovery score when there is no meaningful semantic error
to recover from.

Do not treat harmless display-formatting differences as recovery-worthy errors.

risk:
How risky, destructive, irreversible, or dangerous is the CURRENT action?
0 = low risk.
1 = high risk.

confidence:
Overall confidence in this candidate.

Do NOT alter the supplied deterministic grounding score.

Return JSON only:

{{
  "goal": 0.0,
  "history": 0.0,
  "future": 0.0,
  "recovery": 0.0,
  "risk": 0.0,
  "confidence": 0.0
}}
"""
# android_world/agents/gui_agent/prompts.py

def build_population_scoring_prompt(
    goal: str,
    ui_elements: str,
    history: list[HistoryStep],
    particles: list[Particle],
    grounding_scores: dict[str, float],
    repetition_penalties: dict[str, float],
    effectiveness_priors: dict[str, float],
    observation_feedback: ObservationFeedback | None,
) -> str:

    anomaly = bool(
        observation_feedback is not None
        and observation_feedback.anomaly
    )

    particle_text = []

    for p in particles:
        particle_text.append(
            json.dumps(
                {
                    "particle_id": p.particle_id,
                    "reasoning": p.reasoning,
                    "instruction": p.instruction,
                    "action": p.action,
                    "expected_outcome": p.expected_outcome,
                    "future_steps": p.future_steps,
                    "grounding_score": grounding_scores[p.particle_id],
                    "deterministic_repetition_penalty": (
                        repetition_penalties[p.particle_id]
                    ),
                    "action_effectiveness_prior": (
                        effectiveness_priors[p.particle_id]
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return f"""
You are ranking competing candidate actions for a mobile GUI agent.

USER GOAL:
{goal}

CURRENT UI:
{ui_elements}

HISTORY:
{_history_to_text(history)}

ABNORMAL / FAILED PREVIOUS TRANSITION:
{anomaly}

CANDIDATES:

{chr(10).join(particle_text)}

Compare the candidates RELATIVE TO EACH OTHER.

SEMANTIC EQUIVALENCE RULE:

Evaluate the current GUI and each candidate using semantic task satisfaction,
not literal visual or string equality.

A displayed value may differ in formatting from the user's original wording
while still representing the same intended value.

Examples of harmless differences include:
- spacing,
- punctuation,
- separators,
- parentheses,
- capitalization,
- locale-specific formatting,
- date/time formatting,
- currency formatting,
- application-added presentation formatting.

Do NOT reward a candidate for editing a value solely because its display
format differs from the user's original input.

If the relevant value or subgoal appears semantically satisfied, candidates
that move on to the next useful task action should rank higher than candidates
that continue editing the same state.

Candidates that repeatedly modify an already-satisfied field should receive
LOW:
- current_action_quality
- goal
- history
- recovery
- confidence

This rule is general and is not specific to phone numbers or any particular
application.

The most important question is:

Which CURRENT action is most appropriate for the current GUI state and the
user's goal?

For non-terminal actions, prefer actions that materially improve the GUI state
toward the goal.

If the goal is already semantically satisfied, a correct status=complete
action is preferable to unnecessary additional interaction.

Score each candidate using the following dimensions:

current_action_quality:
How useful is the CURRENT action itself?

TERMINATION ACTION RULE:

A status action is a legitimate CURRENT action.

If the user's goal is already semantically satisfied, then:

{{"action_type": "status", "goal_status": "complete"}}

should receive HIGH:
- current_action_quality
- goal
- history
- confidence

A correct termination action does not need to change the GUI.

Do NOT give status=complete a low current_action_quality merely because it
produces no additional interface transition.

For a correct completion action:
- current_action_quality may be high,
- future may legitimately be low or zero,
- recovery may legitimately be low or zero.

If the goal is NOT yet satisfied, status=complete should receive very low
current_action_quality and goal scores.

This is extremely important.

A candidate MUST NOT receive a high score merely because its future_steps
contain good actions.

Example:

Candidate A:
current action = click the same already-focused text field
future = delete bad text, type correct value, save

Candidate B:
current action = input_text with clear_text=true and the correct value

Candidate B should receive a MUCH higher current_action_quality score.

goal:
How much does the CURRENT action itself advance the real task goal?

history:
Does the CURRENT action make sense given recent attempts and failures?

future:
How plausible and useful are the predicted future steps AFTER the current
action?

IMPORTANT:
future must be evaluated conditionally on the current action being useful.
A weak current action followed by an excellent imagined future should NOT
receive a high future score.

recovery:
If the current state contains an error, duplication, wrong page, failed
transition, or deviation, how strongly does the CURRENT action repair it?

confidence:
Overall confidence.

Rules:

1. Do not reward verbose or convincing reasoning by itself.

2. Penalize repeating an action that recently failed to change the state.

3. If a text field contains wrong or duplicated text, prefer:
   input_text + clear_text=true + correct complete value
   over repeatedly clicking or long-pressing the field.

4. Recovery score should be high only when the CURRENT action itself is
   capable of changing the incorrect state.

5. Future plans cannot compensate for a poor current action.

6. Use deterministic_repetition_penalty as evidence of looping.

7. Use action_effectiveness_prior as a weak prior, especially when recovering
   from an anomaly.

8. Use the supplied grounding_score. Do not overwrite it.

IMPORTANT:
You MUST return exactly one ranking item for EVERY candidate particle provided above.

Do not omit any candidate.
Do not return only the best candidate.

If there are N candidates in CANDIDATES, the ranking array MUST contain exactly N items.

Every particle_id must appear exactly once.

Order the full list from best candidate to worst candidate.

The two items shown below are only a formatting example.
Do NOT copy the number of items from the example.
The actual output must contain exactly {len(particles)} ranking items.

Return JSON only:

{{
  "ranking": [
    {{
      "particle_id": "candidate_id_1",
      "current_action_quality": 0.0,
      "goal": 0.0,
      "history": 0.0,
      "future": 0.0,
      "recovery": 0.0,
      "confidence": 0.0
    }},
    {{
      "particle_id": "candidate_id_2",
      "current_action_quality": 0.0,
      "goal": 0.0,
      "history": 0.0,
      "future": 0.0,
      "recovery": 0.0,
      "confidence": 0.0
    }}
  ]
}}

The example above is illustrative only.
The actual number of ranking items MUST equal the number of candidates.

The first item must be the best candidate.
Use meaningful differences between candidates.
Do not assign identical scores unless the candidates are genuinely equivalent.
"""

def build_observation_check_prompt(
    goal: str,
    previous_particle: Particle,
    current_ui_elements: str,
) -> str:

    return f"""
You are checking whether a real Android GUI transition matched a previous
prediction.

USER GOAL:
{goal}

PREVIOUS GUIDANCE:
{previous_particle.instruction}

PREVIOUS ACTION:
{json.dumps(previous_particle.action, ensure_ascii=False)}

EXPECTED IMMEDIATE OUTCOME:
{previous_particle.expected_outcome}

PREVIOUS INTERNAL LOOKAHEAD:
{json.dumps(previous_particle.future_steps, ensure_ascii=False)}

CURRENT REAL UI AFTER THE HUMAN/ACTION:
{current_ui_elements}

Judge whether the current real GUI is SEMANTICALLY consistent with what was
expected.

IMPORTANT SEMANTIC EQUIVALENCE RULE:

Do NOT require literal visual or string equality between the expected outcome
and the observed GUI.

Android applications may automatically change how a value is displayed while
preserving the same underlying semantic value.

Harmless representation changes may include:
- spaces,
- punctuation,
- separators,
- parentheses,
- capitalization,
- locale-specific formatting,
- date/time formatting,
- currency formatting,
- application-controlled display formatting,
- equivalent labels or UI representations.

If the action successfully entered or changed the intended value, but the
application displays that value using a different harmless format, treat the
transition as CONSISTENT and do NOT mark it as an anomaly.

Example:
An action enters a value in one textual format and Android displays the same
semantic value with application-added spacing or punctuation. This should not
by itself count as a failed transition.

IMPORTANT SEQUENTIAL-STATE RULE:

A real transition may be correct even when the observed GUI shows only a
partial intermediate state rather than the final requested value.

Some controls accumulate or reposition input across multiple actions.
Examples include numeric keypads, PIN entry, timer/duration controls,
date/time controls, multi-digit numeric entry, and incremental controls.

When checking a transition, ask:

1. Is the observed state a plausible immediate result of the action that was
   actually taken?

2. Can this state reasonably serve as an intermediate state from which
   continued actions could still reach the user's goal?

If both are true, treat the transition as consistent even if the displayed
value does not yet equal the final target.

Do NOT mark anomaly=true merely because:
- only part of a multi-step value has been entered,
- a digit currently appears in a lower-order position,
- additional actions are still required to reach the final value,
- or the current state is an expected intermediate stage of a sequential
  interaction.

Only mark an anomaly when the observed result is meaningfully incompatible
with the performed action or with reasonable continued progress.

For positional or right-shifting numeric input, an observed value may be a
valid intermediate state even if the currently visible unit labels appear to
place digits in different units than the final target.

Judge the transition based on whether the newly pressed digit was incorporated
plausibly into the ongoing numeric sequence.

Do not mark a state as anomalous merely because digits have not yet shifted
into their final units.

Only mark anomaly=true when there is meaningful evidence that:
- the intended state change did not occur,
- the wrong semantic value is present,
- the action affected the wrong element,
- the application navigated somewhere unexpected,
- or another meaningful mismatch occurred.

Important:
- A low consistency score can mean the human performed another action.
- It can mean the proposed action was wrong.
- It can mean the app behaved unexpectedly.
- Do NOT force a specific diagnosis unless evidence supports it.
- Cosmetic formatting differences alone are NOT anomalies.

Return JSON only:

{{
  "consistency": 0.0,
  "anomaly": true,
  "summary": "brief description of what appears to have happened"
}}
"""