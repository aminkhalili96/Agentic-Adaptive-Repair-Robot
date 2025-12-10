# Agentic Adaptive Repair Robot (AARR)

> 🤖 LLM-powered robotic system for autonomous surface defect detection and repair

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyBullet](https://img.shields.io/badge/Simulation-PyBullet-green)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 🎯 Overview

AARR demonstrates a complete **Scan-to-Path** automation workflow for MRO (Maintenance, Repair, and Overhaul) operations:

1. **Scan** — Camera captures workpiece image
2. **Detect** — Computer vision identifies defects (rust, cracks, dents)
3. **Localize** — 3D positioning with surface normal estimation
4. **Plan** — LLM agent reasons about repair strategy
5. **Approve** — Human-in-the-loop confirmation
6. **Execute** — Robot follows surface-perpendicular toolpath

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    STREAMLIT DASHBOARD                      │
│   [ Scan ]  [ Plan ]  [ ✓ Approve ]  [ Execute ]           │
└──────────────────────────┬─────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐       ┌──────────┐       ┌──────────┐
   │ Vision  │       │  Agent   │       │ Control  │
   │ OpenCV  │       │LangGraph │       │ PyBullet │
   └─────────┘       └──────────┘       └──────────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                    ┌──────▼──────┐
                    │   KUKA iiwa │
                    │  Simulation │
                    └─────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Interactive 3D Viewer** | Plotly-based mesh visualization with zoom/rotate |
| **Premium Industrial Meshes** | Turbine blade, gear, pipe assembly, gripper, bracket |
| **Multi-Agent Chat** | 🤖 Supervisor, 👁️ Inspector, 🔧 Engineer team |
| **Voice Control** | 🎤 Push-to-speak commands via OpenAI Whisper |
| **Interactive Segmentation** | ✂️ Zero-shot defect masking using SAM (Segment Anything) |
| **Real Computer Vision** | HSV detection with morphological cleanup |
| **Surface Normal Alignment** | Tool perpendicular to curved surfaces |
| **LLM Agent** | LangGraph + Qwen3/GPT-4 for repair planning |
| **ML Predictor** | RandomForest model predicts repair time from defect data |
| **Human-in-the-Loop** | Mandatory approval before execution |
| **Multi-Defect TSP** | Optimized visit order (NN + 2-opt) |
| **Demo Mode** | Procedural meshes with vertex-colored defects |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Conda (recommended)
- Ollama (for local LLM)

### Installation

# Clone the repository
git clone https://github.com/yourusername/robotic_ai.git
cd robotic_ai

# Install Python dependencies
pip install -r requirements.txt

# Note for Mac M1/M2 Users:
# If pybullet fails to install via pip (clang error), you can:
# 1. Try conda: conda install -c conda-forge pybullet
# 2. Or continue without it - the app will automatically run in "Simulation Mock Mode" (features limited to UI & Agent).

# Install Ollama and pull model
brew install ollama
ollama pull qwen3:14b
```

### Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

This opens:
1. **Browser** — Interactive 3D viewer at http://localhost:8501
2. **Sidebar** — Part selection, chat with Factory Team

### Workflow

1. Select **Premium Parts** → Turbine Blade (or other mesh)
2. View 3D model with defect markers
3. 🎤 **Voice**: Click mic, say "Inspect the top corner" → Agent responds
4. Or type: "Show me high severity defects" → Inspector highlights
5. Chat: "Plan the repair" → Engineer gives strategy
6. Click **Generate Plan** → **Approve** → **Execute**

---

## 📁 Project Structure

```
robotic_ai/
├── app/
│   └── streamlit_app.py     # Web dashboard with 3D viewer
├── src/
│   ├── simulation/          # PyBullet environment
│   ├── vision/              # Camera, detection, localization
│   ├── visualization/       # Plotly 3D, premium meshes, demo parts
│   ├── planning/            # Paths, TSP optimization
│   ├── control/             # Robot controller, IK
│   ├── agent/               # LangGraph workflow, multi-agent chat
│   └── ml/                  # ML predictor for repair time estimation
├── docs/
│   ├── ARCHITECTURE.md      # System design
│   ├── MATH.md              # Coordinate transforms
│   └── SAFETY.md            # Safety architecture
├── assets/
│   └── premium_meshes/      # Generated STL files
└── requirements.txt
```

---

## 📊 Documentation

| Document | Contents |
|----------|----------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System diagrams, data flow |
| [MATH.md](docs/MATH.md) | Coordinate frames, surface normals |
| [PROMPTS.md](docs/PROMPTS.md) | LLM prompting strategy |
| [SAFETY.md](docs/SAFETY.md) | Safety layers, human-in-the-loop |
| [SIM_TO_REAL.md](docs/SIM_TO_REAL.md) | Real-world deployment |
| [ML_PREDICTOR.md](docs/ML_PREDICTOR.md) | ML model architecture, API reference |
| [VOICE_CONTROL.md](docs/VOICE_CONTROL.md) | Voice input, Whisper API integration |

---

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
agent:
  provider: ollama          # or "openai"
  model: qwen3:14b          # or "gpt-4o"

safety:
  collision_distance: 0.01
  workspace_bounds:
    x: [0.2, 0.8]
    y: [-0.4, 0.4]
    z: [0.05, 0.6]
```

---

## 🎥 Demo

> The robot detects a rust defect, plans a spiral sanding path, and executes with tool perpendicular to the surface.

*Coming soon: Split-screen demo video*

---

## 🛡️ Safety

- **Workspace bounds** — All positions validated
- **Collision detection** — Checked every N waypoints
- **Human approval** — Required before execution
- **LLM fallback** — Deterministic rules if AI fails

---

## 📈 Future Roadmap

- [ ] Point cloud input from RealSense
- [ ] Fine-tuned defect classification model
- [ ] Real robot integration (KUKA RSI)
- [ ] Force-feedback during execution
- [ ] Multi-robot coordination
- [x] ML predictor for repair time estimation

---

## 🙏 Acknowledgments

- [PyBullet](https://pybullet.org/) — Physics simulation
- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent framework
- [Ollama](https://ollama.ai/) — Local LLM inference
- [Augmentus](https://augmentus.tech/) — Inspiration for scan-to-path

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.
