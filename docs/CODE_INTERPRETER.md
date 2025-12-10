# Code Interpreter for Custom Path Generation

The AARR (Agentic Adaptive Repair Robot) system includes a **Code Interpreter** capability that enables true "General Intelligence" - the robot adapts to novel requests by writing its own path generation algorithms.

## Overview

When a user requests a geometric pattern not available as a built-in option (spiral, raster), the AI agent:
1. Writes Python code to generate the custom path
2. Executes it in a secure sandbox
3. Visualizes the result on the 3D viewer

**Example**: "Scan this area using a Star-Shaped pattern" → Agent generates star pattern code → Path appears on viewer

---

## Security Model

The code interpreter uses a **sandboxed execution environment** that:

| Feature | Implementation |
|---------|----------------|
| **Allowed** | `numpy`, `math`, basic builtins (`range`, `len`, `float`, `list`, etc.) |
| **Blocked** | `os`, `sys`, `subprocess`, `open`, `exec`, `eval`, `__import__`, file I/O |
| **Timeout** | 2 second limit (configurable) |
| **Validation** | AST-based analysis before execution |
| **Return Type** | Must be numpy array of shape (N, 3) |

### Blocked Keywords
```python
FORBIDDEN_KEYWORDS = {
    'import',  # Only numpy/math allowed
    'open', 'file', 'exec', 'eval', 'compile',
    '__import__', 'subprocess', 'os', 'sys', 'shutil',
    'socket', 'urllib', 'requests', 'pickle', ...
}
```

---

## Usage

### In Chat
```
User: "Scan the defect area using a hexagon pattern"
Agent: ✨ **Custom Path Generated**: Hexagonal pattern
       Created 7 waypoints. The path is now displayed on the 3D viewer in magenta.
       
       ```python
       def generate_custom_path(center, radius):
           import numpy as np
           cx, cy, cz = center
           points = []
           for i in range(7):
               angle = (i * 2 * np.pi / 6) - np.pi/2
               points.append([cx + radius*np.cos(angle), cy + radius*np.sin(angle), cz])
           return np.array(points)
       ```
```

### Supported Patterns
The agent can generate any mathematical pattern including:
- **Star** (5-pointed, 6-pointed, etc.)
- **Hexagon**, **Pentagon**, **Triangle**
- **Zigzag**, **Sine wave**
- **Flower**, **Spiral (custom variants)**
- **Any parametric curve** using `np.sin`, `np.cos`, `np.linspace`

---

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  User Request   │────▶│  Supervisor Agent    │────▶│ Code Interpreter│
│ "star pattern"  │     │  (GPT-4o Function    │     │ Sandbox         │
└─────────────────┘     │   Calling)           │     └────────┬────────┘
                        └──────────────────────┘              │
                                                              ▼
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  3D Viewer      │◀────│  Streamlit App       │◀────│ CustomPathResult│
│  (Magenta path) │     │  (CUSTOM_PATH_       │     │ {points, success}│
└─────────────────┘     │   GENERATED cmd)     │     └─────────────────┘
                        └──────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| [code_interpreter.py](file:///Users/amin/dev/Robotic%20AI/src/planning/code_interpreter.py) | Sandbox execution, validation, example patterns |
| [supervisor_agent.py](file:///Users/amin/dev/Robotic%20AI/src/agent/supervisor_agent.py) | `generate_custom_path` tool and system prompt |
| [streamlit_app.py](file:///Users/amin/dev/Robotic%20AI/app/streamlit_app.py) | Visualization of custom paths (magenta `Scatter3d`) |

---

## API Reference

### `exec_custom_path(code_str, center, radius, timeout_seconds=2)`

Execute custom path generation code in sandbox.

**Parameters:**
- `code_str`: Python code containing `generate_custom_path(center, radius)` function
- `center`: Tuple `(x, y, z)` - center point for the path
- `radius`: Float - size parameter for the path
- `timeout_seconds`: Maximum execution time (default 2s)

**Returns:** `CustomPathResult`
- `points`: numpy array of shape (N, 3) or None
- `success`: boolean
- `error_message`: string if failed
- `generated_code`: the executed code

### Example Code (Star Pattern)
```python
def generate_custom_path(center, radius):
    import numpy as np
    cx, cy, cz = center
    points = []
    num_points = 5
    inner_radius = radius * 0.4
    
    for i in range(num_points * 2):
        angle = (i * np.pi / num_points) - np.pi / 2
        r = radius if i % 2 == 0 else inner_radius
        points.append([cx + r*np.cos(angle), cy + r*np.sin(angle), cz])
    
    points.append(points[0])  # Close the star
    return np.array(points)
```

---

## Safety Considerations

> [!CAUTION]
> The Code Interpreter executes LLM-generated code. While sandboxed, this is inherently riskier than calling pre-defined functions.

**Mitigations:**
1. **AST Validation**: Code is parsed and analyzed before execution
2. **Restricted Builtins**: Only safe functions available
3. **Import Whitelist**: Only `numpy` and `math`
4. **Timeout**: Hard 2-second limit prevents infinite loops
5. **Return Validation**: Must return proper numpy array shape
