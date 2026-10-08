# Human-in-the-Loop Mobile GUI Guidance with Adaptive Particle Search

This repository contains the implementation for an MSc dissertation project on human-in-the-loop mobile GUI assistance using multimodal large language models.

The system is built on top of [AndroidWorld](https://github.com/google-research/android_world), a dynamic benchmarking environment for Android agents.

Instead of allowing the agent to directly execute every action, the proposed system operates as a GUI guidance assistant. At each interaction step, the agent observes the current Android interface, generates multiple candidate action hypotheses, performs adaptive particle-based search, and presents the selected action to the human user as:

- a natural-language instruction; and
- a bounding-box highlight indicating where the action should be performed.

The human performs the action manually. The resulting Android state is then observed by the agent and used for the next decision step.

The implementation supports:

- Qwen3-VL-4B-Instruct as the base multimodal model;
- particle-based candidate action search;
- short-horizon action lookahead;
- adaptive search and resampling;
- observation-conditioned rescoring;
- recovery after unexpected user actions;
- human-in-the-loop execution;
- screenshot guidance overlays;
- AndroidWorld task evaluation; and
- detailed trajectory and search logging.

---

## Overview

The interaction loop used in this project is:

```text
Current Android state
        |
        v
Screenshot + UI information
        |
        v
Candidate action generation
        |
        v
Particle-based search
        |
        v
Candidate scoring
        |
        v
Selected action hypothesis
        |
        +--------------------------+
        |                          |
        v                          v
Natural-language instruction   Bounding box
        |                          |
        +------------+-------------+
                     |
                     v
                Human action
                     |
                     v
              New Android state
                     |
                     v
       Observation consistency check
                     |
                     v
              Next decision step
```

Each particle represents a candidate current action together with a short sequence of future actions.

Candidate particles are scored using several factors, including:

- goal consistency;
- UI grounding confidence;
- history consistency;
- future-step usefulness;
- recovery suitability;
- action risk;
- confidence;
- observation consistency; and
- repetition penalties.

Additional computation can be allocated adaptively when the search state is uncertain, candidate particles disagree, or the observed interface differs from the predicted outcome.

Recovery is handled within the same particle-search procedure rather than through a separate recovery module.

---

## AndroidWorld

This project uses [AndroidWorld](https://github.com/google-research/android_world) as the Android task environment and evaluation benchmark.

AndroidWorld contains 116 hand-crafted tasks across 20 Android applications and supports dynamically generated task parameters.

Examples include:

```text
ContactsAddContact
ClockTimerEntry
MarkorCreateNote
OsmAndFavorite
FilesMoveFile
SimpleSmsSend
```

AndroidWorld provides:

- Android application setup;
- task definitions;
- dynamically generated task parameters;
- emulator interaction;
- task-specific success evaluators; and
- benchmark execution infrastructure.

The custom guidance agent in this repository extends the AndroidWorld agent interface.

---

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd android_world
```

The original AndroidWorld repository is available at:

```text
https://github.com/google-research/android_world
```

### 2. Set up the Android Emulator

AndroidWorld requires an Android Virtual Device.

The configuration used for this project is:

```text
Device: Pixel 6
Android version: Android 13
API level: 33
AVD name: AndroidWorldAvd
gRPC port: 8554
```

Example launch command:

```bash
export ANDROID_SDK_ROOT=/path/to/android-sdk
export ANDROID_HOME=/path/to/android-sdk
export ANDROID_AVD_HOME=/path/to/android-avd

$ANDROID_SDK_ROOT/emulator/emulator \
  -avd AndroidWorldAvd \
  -no-window \
  -grpc 8554 \
  -no-audio
```

Check that the emulator is running:

```bash
adb devices
```

Expected output should contain an emulator such as:

```text
List of devices attached
emulator-5554    device
```

### 3. Python environment

Python 3.11 is recommended.

```bash
python3.11 -m venv /tmp/qwen-venv
source /tmp/qwen-venv/bin/activate
```

Install the AndroidWorld requirements:

```bash
pip install -r requirements.txt
```

Install the additional dependencies used by the Qwen3-VL guidance agent:

```bash
pip install transformers accelerate qwen-vl-utils safetensors huggingface-hub
```

### 4. Model

The multimodal model used in this project is:

```text
Qwen/Qwen3-VL-4B-Instruct
```

A CUDA-capable GPU is strongly recommended. The development experiments were run using an NVIDIA RTX 3090.

Verify CUDA availability:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

---

## Docker Setup

The experimental configuration used in this project separates the Android emulator from the model container.

```text
Host machine
|
+-- Android Emulator
|
+-- Docker
    |
    +-- AndroidWorld
    +-- Qwen3-VL
    +-- GUI guidance agent
```

Example Docker command:

```bash
docker run --rm -it \
  --gpus '"device=0"' \
  --network host \
  --name gui-agent \
  -v /path/to/android_world:/workspace/android_world \
  -v /path/to/models:/workspace/models \
  -v /path/to/huggingface-cache:/root/.cache/huggingface \
  -v /path/to/outputs:/workspace/outputs/android_world \
  neural-os/android-world:latest
```

Verify GPU access inside the container:

```bash
nvidia-smi
```

---

## ADB Access from Docker

The Android emulator runs on the host while the agent runs inside Docker.

The configuration used in the experiments is:

```text
ADB path:
/opt/android/platform-tools/adb

Emulator console port:
5554
```

Verify the connection from inside the container:

```bash
/opt/android/platform-tools/adb devices
```

Expected output:

```text
emulator-5554    device
```

---

## Quickstart

Activate the Python environment:

```bash
source /tmp/qwen-venv/bin/activate
```

Enter the AndroidWorld repository:

```bash
cd /workspace/android_world
```

Run the custom GUI guidance agent on a single task:

```bash
python run.py \
  --suite_family=android_world \
  --agent_name=gui_particle_qwen3vl \
  --tasks=ContactsAddContact \
  --n_task_combinations=1 \
  --adb_path=/opt/android/platform-tools/adb \
  --console_port=5554 \
  --output_path=/workspace/outputs/android_world
```

The custom agent is registered as:

```text
gui_particle_qwen3vl
```

---

## Run the Benchmark

A specific AndroidWorld task can be selected using:

```bash
--tasks=<TASK_NAME>
```

For example:

```bash
python run.py \
  --suite_family=android_world \
  --agent_name=gui_particle_qwen3vl \
  --tasks=ClockTimerEntry \
  --n_task_combinations=1 \
  --adb_path=/opt/android/platform-tools/adb \
  --console_port=5554 \
  --output_path=/workspace/outputs/android_world
```

Other tested tasks include:

```text
ContactsAddContact
ClockTimerEntry
MarkorCreateNote
OsmAndFavorite
FilesMoveFile
SimpleSmsSend
```

Multiple task parameterisations can be evaluated using:

```bash
--n_task_combinations=N
```

AndroidWorld dynamically instantiates task parameters, so different runs of the same task may use different names, phone numbers, messages, timer values, filenames, file locations, or geographic coordinates.

---

## Human-in-the-Loop Mode

The custom agent is configured to operate in human-in-the-loop mode.

Instead of directly performing the selected action, the agent displays guidance such as:

```text
================================================================================
HUMAN GUIDANCE
================================================================================
Instruction: Enter the last name 'Pereira' in the last name field.
BBox: [137, 1043, 933, 1192]
Guidance image: /workspace/outputs/android_world/gui_guidance_overlays/step_004_guidance.png
Waiting for human action...
================================================================================

Press ENTER here after you finish the action in the emulator...
```

The human user performs the action and then presses `ENTER` in the agent terminal.

The human is not required to follow the recommendation exactly. Unexpected human actions are intentionally supported because they allow the system's recovery behaviour to be evaluated.

---

## Guidance Overlay

For each selected action, the agent produces an annotated screenshot containing:

- the natural-language instruction; and
- the target bounding box.

Guidance images are written to:

```text
/workspace/outputs/android_world/gui_guidance_overlays/
```

---

## Manual Interaction with ADB

When the Android emulator is running headlessly, the human can execute actions through ADB.

### Tap

```bash
adb -s emulator-5554 shell input tap X Y
```

### Swipe

```bash
adb -s emulator-5554 shell input swipe X1 Y1 X2 Y2 DURATION
```

### Long press

```bash
adb -s emulator-5554 shell input swipe X Y X Y 800
```

### Text input

```bash
adb -s emulator-5554 shell input text 'Hello'
```

Spaces can be represented using `%s`:

```bash
adb -s emulator-5554 shell input text 'Ignorance%sis%sbliss.'
```

### Back

```bash
adb -s emulator-5554 shell input keyevent 4
```

### Backspace

```bash
adb -s emulator-5554 shell input keyevent 67
```

### Move cursor to end

```bash
adb -s emulator-5554 shell input keyevent 123
```

### Enter a plus sign

```bash
adb -s emulator-5554 shell input keyevent 81
```

---

## Inspecting the Android UI

The current Android accessibility hierarchy can be inspected using UIAutomator.

```bash
adb -s emulator-5554 shell uiautomator dump /sdcard/window.xml
```

Example lookup:

```bash
adb -s emulator-5554 shell cat /sdcard/window.xml | grep -o 'text="Last name"[^>]*'
```

Example output:

```text
text="Last name" class="android.widget.EditText" clickable="true" bounds="[137,1043][933,1192]"
```

This is useful when scrolling, dialogs, keyboard visibility, or navigation make an earlier bounding box stale.

---

## Screenshots

The current emulator screen can be captured using:

```bash
adb -s emulator-5554 exec-out screencap -p > current_screen.png
```

---

## Adaptive Particle Search

The main planning procedure uses a population of candidate particles.

A particle contains information such as:

```text
particle_id
reasoning
instruction
action
expected_outcome
future_steps
bbox
scores
generation_round
parent_id
log_weight
```

Each particle represents a hypothesis about both what action should be performed now and what short sequence of future actions is likely to follow.

Particles are generated, scored, retained, refined, and optionally resampled. Additional computation can be allocated when candidate actions disagree, confidence is low, the observed state differs from the predicted state, or the current trajectory appears anomalous.

---

## Observation-Conditioned Recovery

After the human performs an action, the agent observes the new Android state and compares it with the predicted outcome of the selected particle.

Observation feedback can include:

```text
consistency
anomaly
summary
```

Recovery therefore emerges from the same search process used for normal navigation.

Possible recovery hypotheses include:

```text
continue from the current state
retry the previous interaction
navigate backwards
select an alternative route
correct an incorrectly entered value
```

For example, if the agent recommends entering `Pereira` but the human enters `PereiraPereira`, the next observation may cause the agent to generate a correction instruction.

Recovery is not guaranteed. Failed recoveries are retained for qualitative analysis.

---

## Repository Structure

The custom guidance implementation is located under:

```text
android_world/agents/gui_agent/
```

The main components include:

```text
android_world/
|
+-- agents/
|   |
|   +-- gui_agent/
|       |
|       +-- agent.py
|       +-- particle_search.py
|       +-- ...
|
+-- run.py
```

### `agent.py`

The main GUI guidance agent. Responsibilities include retrieving the current Android state, obtaining screenshots, invoking particle search, selecting the next candidate action, grounding the action to a GUI element, producing user-facing guidance, generating overlays, waiting for the human action, observing the resulting state, and maintaining interaction history.

### `particle_search.py`

Implements candidate generation, particle construction, scoring, elite retention, proposal refinement, adaptive resampling, observation-conditioned scoring, and final particle selection.

---

## Output and Logging

Each benchmark execution creates a timestamped run directory:

```text
/workspace/outputs/android_world/run_<timestamp>/
```

Episode data are stored in compressed pickle files such as:

```text
ContactsAddContact_0.pkl.gz
```

Top-level episode fields include:

```text
goal
task_template
episode_data
is_successful
run_time
finish_dtime
episode_length
aux_data
screen_config
exception_info
seed
agent_name
instance_id
```

The custom `episode_data` contains information such as:

```text
mode
human_transition_detected
selected_instruction
selected_bbox
selected_action
search
model_calls
generated_tokens
guidance_image_path
step_number
goal_status
history
```

The `search` field contains:

```text
selected
particles
rounds_used
proposals_generated
resampling_reasons
```

Observation feedback can contain:

```text
consistency
anomaly
summary
```

---

## Evaluation

Task success is evaluated using the original AndroidWorld task evaluator rather than by the guidance agent itself.

Metrics recorded during experiments include:

```text
task success
episode length
runtime
model calls
generated tokens
number of proposals
number of search rounds
particle scores
resampling events
observation consistency
recovery behaviour
```

`model_calls` and `generated_tokens` are stored cumulatively during an episode, so per-step computation should be calculated using differences between consecutive logged values rather than by summing cumulative values.

---

## Known Limitations

### Dynamic UI indices

Accessibility indices can become invalid after interface transitions.

### Bounding-box drift

Scrolling, dialogs, keyboard visibility, and navigation can move interface elements. Bounding boxes should always be interpreted relative to the screenshot from which they were generated.

### Application selection

The multimodal model may occasionally select a semantically similar but incorrect application. For example, an SMS task requiring Simple SMS Messenger may be incorrectly attempted using Google Messages.

### Semantic recovery

The system can recover from some local text or navigation errors but does not always identify more complex semantic mistakes, such as values entered into the wrong form field.

### Human-action logging

The system records the agent-selected action, instruction, bounding box, and resulting state, but arbitrary human override actions are not currently stored as explicit structured action labels.

---

## Original AndroidWorld

This repository builds on:

```text
AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents
Christopher Rawles et al.
```

Original repository:

```text
https://github.com/google-research/android_world
```

AndroidWorld is used for Android environment setup, benchmark tasks, application state management, task parameter generation, and success evaluation.

The following components are project-specific extensions:

```text
human-in-the-loop guidance
Qwen3-VL integration
particle-based action search
adaptive search
observation-conditioned recovery
guidance overlays
custom search logging
```

---

## Citation

If you use AndroidWorld, please cite the original AndroidWorld paper:

```bibtex
@misc{rawles2024androidworlddynamicbenchmarkingenvironment,
  title={AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents},
  author={Christopher Rawles and Sarah Clinckemaillie and Yifan Chang and
          Jonathan Waltz and Gabrielle Lau and Marybeth Fair and Alice Li and
          William Bishop and Wei Li and Folawiyo Campbell-Ajala and Daniel Toyama
          and Robert Berry and Divya Tyamagundlu and Timothy Lillicrap and
          Oriana Riva},
  year={2024},
  eprint={2405.14573},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2405.14573}
}
```

---

## Acknowledgements

This project uses AndroidWorld as its Android execution and evaluation environment and Qwen3-VL-4B-Instruct as the base multimodal model.

AndroidWorld and Qwen3-VL are external projects. The human-in-the-loop guidance architecture, particle-search procedure, adaptive search behaviour, recovery mechanism, guidance overlays, and custom logging described above are project-specific extensions.
