# System Architecture

High-level architecture of the Agentic Adaptive Repair Robot (AARR) system.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STREAMLIT DASHBOARD                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────────────────────┐ │
│  │  Scan   │ │  Plan   │ │ Approve │ │ Execute │ │     Metrics Display    │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────────────────────────┘ │
└───────┼──────────┼──────────┼──────────┼────────────────────────────────────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              CORE MODULES                                     │
│                                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   VISION     │  │    AGENT     │  │   PLANNING   │  │   CONTROL    │      │
│  │  - Camera    │  │  - LangGraph │  │  - Paths     │  │  - IK Solver │      │
│  │  - Detector  │  │  - Prompts   │  │  - TSP       │  │  - Motor Ctl │      │
│  │  - Localizer │  │  - Fallback  │  │  - Velocity  │  │  - Collision │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │                 │               │
└─────────┼─────────────────┼─────────────────┼─────────────────┼───────────────┘
          │                 │                 │                 │
          ▼                 ▼                 ▼                 ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           SIMULATION ENGINE                                   │
│                                                                               │
│  ┌──────────────────────┐  ┌────────────────┐  ┌────────────────────────────┐│
│  │   PyBullet Physics   │  │   KUKA iiwa    │  │      Workpiece + Defects   ││
│  │   - Collision        │  │   7-DOF Robot  │  │      - Visual markers      ││
│  │   - Ray casting      │  │   - URDF model │  │      - State tracking      ││
│  └──────────────────────┘  └────────────────┘  └────────────────────────────┘│
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL SERVICES                                   │
│                                                                               │
│  ┌──────────────────────┐  ┌────────────────────────────────────────────────┐│
│  │   Ollama (Qwen3)     │  │              OpenAI API (GPT-4o)               ││
│  │   Local LLM          │  │              + GPT-4o Vision (multimodal)      ││
│  └──────────────────────┘  └────────────────────────────────────────────────┘│
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
┌─────────┐    RGB/Depth    ┌─────────┐    Pixel (u,v)   ┌─────────┐
│ Camera  │ ───────────────▶│Detector │ ────────────────▶│Localizer│
└─────────┘                 └─────────┘                  └────┬────┘
                                                              │
                                                    3D Pose + Normal
                                                              │
                                                              ▼
┌─────────┐    Repair Plan  ┌─────────┐    DefectInfo    ┌─────────┐
│  Path   │ ◀───────────────│  Agent  │ ◀────────────────│  Agent  │
│Generator│                 │ (LLM)   │                  │  Input  │
└────┬────┘                 └─────────┘                  └─────────┘
     │
     │ Waypoints
     ▼
┌─────────┐    Joint Angles ┌─────────┐    Motor Cmds   ┌─────────┐
│   IK    │ ───────────────▶│ Control │ ───────────────▶│ PyBullet│
│ Solver  │                 │  Loop   │                 │  Robot  │
└─────────┘                 └─────────┘                 └─────────┘
```

---

## Module Responsibilities

### Vision (`src/vision/`)

| File | Responsibility |
|------|----------------|
| `camera.py` | Capture RGB/depth from PyBullet |
| `detector.py` | HSV color detection + morphology |
| `localization.py` | Pixel→3D, ray casting, surface normals |

### Agent (`src/agent/`)

| File | Responsibility |
|------|----------------|
| `prompts.py` | System prompts and templates |
| `tools.py` | DefectInfo, RepairPlan, fallbacks |
| `graph.py` | LangGraph workflow, LLM calls |
| `supervisor_agent.py` | GPT-4o conversational agent with function calling |
| `knowledge_base.py` | RAG system: SOP documents + vector search |

#### RAG Knowledge Base

The `knowledge_base.py` provides retrieval-augmented generation for repair parameters:

- **SOP Documents**: Industrial rules for materials (Aluminum, Steel, Composite) and defects (Rust, Crack, Dent)
- **Vector Search**: Numpy-based cosine similarity for lightweight retrieval
- **Tool**: `consult_manual(query)` - Called by Supervisor Agent to cite verified specifications

Example flow:
```
User: "How do we fix Steel?"
  → Agent calls consult_manual("Steel repair")
  → Returns: "Material: Steel | Tool: Grinder | Speed: 3000 RPM | Pressure: High"
  → Agent: "According to the SOP, for Steel, I am setting the robot to 3000 RPM."
```

#### Visual Inspection Feature

The `supervisor_agent.py` includes multimodal (vision) capability:

- **Trigger phrases**: "Look at this", "What do you see?", "Analyze the screen"
- **Process**: Captures 3D viewer → sends to GPT-4o Vision → returns defect description
- **Tool**: `analyze_visual` triggers `CAPTURE_SNAPSHOT` UI command
- **Image capture**: Uses Kaleido for server-side PNG generation

#### Voice Control Feature

The `streamlit_app.py` includes voice-to-text capability for glove-wearing operators:

- **Widget**: `st.audio_input("🎤 Push to Speak Command")` native Streamlit widget
- **Processing**: Audio sent to OpenAI Whisper API (`whisper-1`) for transcription
- **Auto-submit**: Transcribed text auto-injected into chat and sent to Supervisor Agent
- **Use case**: Operators say "Inspect the top corner" and the AI Agent responds immediately

### Planning (`src/planning/`)

| File | Responsibility |
|------|----------------|
| `paths.py` | Spiral/raster path generation |
| `tsp.py` | Multi-defect ordering (NN + 2-opt) |

### Control (`src/control/`)

| File | Responsibility |
|------|----------------|
| `controller.py` | IK, motor control, collision check |

### Simulation (`src/simulation/`)

| File | Responsibility |
|------|----------------|
| `environment.py` | PyBullet setup, KUKA loading |
| `defects.py` | Defect spawning, repair marking |

---

## Configuration

All parameters in `config.yaml`:

```yaml
simulation:
  seed: 42
  gui: true

camera:
  width: 640
  height: 480
  fov: 60

vision:
  rust_hsv_lower: [0, 100, 100]
  rust_hsv_upper: [10, 255, 255]
  min_contour_area: 100

agent:
  provider: ollama
  model: qwen3:14b
  timeout: 30

safety:
  collision_distance: 0.01
  workspace_bounds:
    x: [0.2, 0.8]
    y: [-0.4, 0.4]
    z: [0.05, 0.6]

path:
  max_velocity: 0.1
  max_acceleration: 0.05
```

---

## Dependencies

```
pybullet        - Physics simulation
numpy           - Numerical computing
opencv-python   - Computer vision
scipy           - Surface normal math
langchain       - LLM framework
langgraph       - Agent workflow
langchain-ollama- Ollama integration
streamlit       - Web dashboard
kaleido         - Plotly image export (for visual inspection)
```
