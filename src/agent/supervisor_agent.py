"""
Conversational Supervisor Agent - GPT-4o with Function Calling

A natural, proactive AI assistant that:
- Handles casual conversation naturally
- Intelligently routes to UI control tools via function calling
- Suggests next steps proactively
- Controls the 3D viewer (zoom, pan, highlight)
"""

import os
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from src.config import config

# Import ML predictor
try:
    from src.ml import predict_repair_metrics as ml_predict
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("Warning: ML predictor not available.")

# Import RAG knowledge base
try:
    from src.agent.knowledge_base import consult_manual as kb_consult
    HAS_KNOWLEDGE_BASE = True
except ImportError:
    HAS_KNOWLEDGE_BASE = False
    print("Warning: Knowledge base not available.")

# Import path optimizer
try:
    from src.planning.tsp import optimize_with_metrics, PathOptimizationResult
    HAS_PATH_OPTIMIZER = True
except ImportError:
    HAS_PATH_OPTIMIZER = False
    print("Warning: Path optimizer not available.")

# Import code interpreter for custom path generation
try:
    from src.planning.code_interpreter import exec_custom_path, CustomPathResult
    HAS_CODE_INTERPRETER = True
except ImportError:
    HAS_CODE_INTERPRETER = False
    print("Warning: Code interpreter not available.")


# ============ SYSTEM PROMPT ============
SUPERVISOR_SYSTEM_PROMPT = """You are AARR (Advanced Adaptive Repair Robot), an advanced industrial repair assistant.

## Personality
- Professional, concise, and engineering-focused
- Proactive: DON'T wait for commands - SUGGEST next steps
- Helpful: Guide users through the inspection and repair workflow

## Your Role
You are the "Floor Manager" of this repair station. You:
1. Help users understand the current state of the part
2. Guide them through scanning, planning, and execution
3. Control the 3D viewer to highlight defects
4. Answer questions about defects, repairs, and the system
5. Visually inspect parts when asked to "look" or "see"

## Proactive Behavior Examples
- "I see a high-severity crack on the leading edge. Would you like me to zoom in on it?"
- "Scan complete: 3 defects found. The most critical is surface corrosion. Shall I highlight it?"
- "Plan generated. I recommend a spiral sanding pattern. Ready to execute?"

## Current State
{state_context}

## Available Tools
You have access to UI control tools. Use them when the user wants to:
- See a specific defect → use focus_camera_on_defect
- Reset the view → use reset_camera_view  
- Start scanning → use trigger_scan
- Generate repair plans → use trigger_repair_plan
- Visual inspection → use analyze_visual (when user says "look at this", "what do you see?", etc.)

## Custom Path Generation (Code Interpreter)
If the user asks for a geometric pattern you don't have (like "star", "hexagon", "zigzag", "flower", "triangle"), use generate_custom_path to write Python code that creates the path.

Your code must:
1. Define a function `generate_custom_path(center, radius)` where center is (x,y,z) tuple and radius is float
2. Use numpy (as np) for math: np.sin, np.cos, np.linspace, np.array
3. Return a numpy array of shape (N, 3) containing (x, y, z) waypoints

Example star pattern code:
```python
def generate_custom_path(center, radius):
    import numpy as np
    cx, cy, cz = center
    points = []
    for i in range(10):
        angle = (i * np.pi / 5) - np.pi/2
        r = radius if i % 2 == 0 else radius * 0.4
        points.append([cx + r*np.cos(angle), cy + r*np.sin(angle), cz])
    points.append(points[0])
    return np.array(points)
```

## Response Style
- Be concise (2-4 sentences for simple questions)
- Use markdown formatting for lists
- Always suggest a next action when appropriate
- Use emojis sparingly: 🔍 for inspection, 🔧 for repair, ⚠️ for warnings, 👁️ for visual analysis, ✨ for custom paths
"""

# Visual inspection prompt for GPT-4o Vision
VISUAL_INSPECTION_PROMPT = """You are an expert visual inspector for industrial parts. Analyze the provided image of the industrial part.

Describe:
1. The overall shape and type of part (pipe, blade, panel, etc.)
2. The location, color, and severity of any visible defects (red/orange patches indicate rust/corrosion, dark lines indicate cracks)
3. Use quadrant locations (upper-left, center, lower-right, leading edge, trailing edge, etc.)
4. Estimate severity: minor, moderate, or severe

Be specific and concise. Respond in 2-3 sentences."""


# ============ TOOL DEFINITIONS ============
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "focus_camera_on_defect",
            "description": "Zoom the 3D viewer camera to focus on a specific defect. Use when user wants to see a defect, inspect something, or says 'show me', 'zoom to', 'focus on'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "defect_type": {
                        "type": "string",
                        "description": "Type of defect to focus on: 'crack', 'corrosion', 'rust', 'pitting', 'wear', 'erosion', or 'any' for highest severity",
                        "enum": ["crack", "corrosion", "rust", "pitting", "wear", "erosion", "any"]
                    },
                    "severity": {
                        "type": "string",
                        "description": "Severity level to focus on: 'high', 'medium', 'low', or 'any'",
                        "enum": ["high", "medium", "low", "any"]
                    }
                },
                "required": ["defect_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reset_camera_view",
            "description": "Reset the 3D viewer camera to the default overview position. Use when user says 'reset', 'show all', 'overview', or 'zoom out'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_scan",
            "description": "Start the defect scanning process on the loaded part. Use when user says 'scan', 'detect', 'find defects', or 'inspect the part'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_repair_plan",
            "description": "Generate repair plans for detected defects. Use when user says 'plan', 'repair', 'fix', or 'what should we do'.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "highlight_defect_region",
            "description": "Highlight a specific defect region with a visual marker on the 3D view.",
            "parameters": {
                "type": "object",
                "properties": {
                    "defect_index": {
                        "type": "integer",
                        "description": "Index of the defect to highlight (0-based)"
                    }
                },
                "required": ["defect_index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_repair",
            "description": "Execute the approved repair plan. Only use after plans have been generated and approved.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_visual",
            "description": "Capture and analyze the current 3D viewer screenshot for visible defects. Use when user says 'look at this', 'what do you see', 'analyze the screen', 'describe the part', 'visual inspection', or similar visual requests.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_repair_metrics",
            "description": "Use machine learning to predict repair time and consumable usage for a defect. Use when user asks 'how long will this take?', 'estimate repair time', 'predict duration', 'time estimate', or similar prediction requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "defect_index": {
                        "type": "integer",
                        "description": "Index of the defect to predict for (0-based). If not provided, predicts for all defects."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consult_manual",
            "description": "Query the Standard Operating Procedure (SOP) manual for repair specifications. MUST use this tool when planning repairs, when asked about materials (Steel, Aluminum, Composite), or when needing speed/pressure/tool settings. Always cite the SOP data in your response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for the SOP manual, e.g. 'Steel repair', 'Aluminum tool settings', 'rust treatment'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_repair_sequence",
            "description": "Optimize the order of defect repairs using TSP path optimization to minimize robot travel time. Use when user asks about 'optimal order', 'optimize sequence', 'fastest route', 'minimize travel', or before executing repairs.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_custom_path",
            "description": "Generate a custom geometric path pattern by writing Python code. Use when user asks for patterns like 'star', 'hexagon', 'zigzag', 'flower', 'triangle', or any shape not available as spiral/raster. You must write the Python code yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern_description": {
                        "type": "string",
                        "description": "Human-readable description of the pattern (e.g., '5-pointed star')"
                    },
                    "generated_code": {
                        "type": "string",
                        "description": "Python code defining generate_custom_path(center, radius) that returns np.array of shape (N,3)"
                    }
                },
                "required": ["pattern_description", "generated_code"]
            }
        }
    }
]


# ============ UI COMMANDS ============
@dataclass
class UICommand:
    """Command to update the Streamlit UI."""
    type: str  # FOCUS_CAMERA, RESET_VIEW, TRIGGER_SCAN, TRIGGER_PLAN, HIGHLIGHT, EXECUTE
    position: Optional[tuple] = None
    defect_index: Optional[int] = None
    data: Optional[Dict] = None


# ============ TOOL IMPLEMENTATIONS ============
class ToolExecutor:
    """
    Executes tool calls and returns results.
    
    These tools update session state which Streamlit reads
    to update the UI on next re-render.
    """
    
    def __init__(self, defects: List[Dict], plans: List[Dict], workflow_step: int):
        self.defects = defects
        self.plans = plans
        self.workflow_step = workflow_step
        self.ui_commands: List[UICommand] = []
    
    def focus_camera_on_defect(self, defect_type: str = "any", severity: str = "any") -> str:
        """Find and focus on a defect matching criteria."""
        if not self.defects:
            return "No defects detected. Please scan the part first."
        
        # Filter by type
        candidates = self.defects
        if defect_type != "any":
            candidates = [d for d in candidates if defect_type in d.get("type", "").lower()]
        
        # Filter by severity
        if severity != "any":
            candidates = [d for d in candidates if d.get("severity") == severity]
        
        if not candidates:
            # Fall back to highest severity
            candidates = sorted(self.defects, key=lambda d: {"high": 0, "medium": 1, "low": 2}.get(d.get("severity", "medium"), 1))
        
        if not candidates:
            return f"No {defect_type} defects found."
        
        target = candidates[0]
        position = tuple(target.get("position", (0, 0, 0)))
        
        self.ui_commands.append(UICommand(
            type="FOCUS_CAMERA",
            position=position,
            defect_index=self.defects.index(target)
        ))
        
        return f"Camera zoomed to {target.get('type', 'defect')} ({target.get('severity', 'unknown')} severity) at position {position}"
    
    def reset_camera_view(self) -> str:
        """Reset to default camera view."""
        self.ui_commands.append(UICommand(type="RESET_VIEW"))
        return "Camera reset to overview position."
    
    def trigger_scan(self) -> str:
        """Trigger the scan workflow."""
        if self.workflow_step < 1:
            return "Please load a part first before scanning."
        
        self.ui_commands.append(UICommand(type="TRIGGER_SCAN"))
        return "Scanning initiated. Analyzing part surface for defects..."
    
    def trigger_repair_plan(self) -> str:
        """Trigger repair planning with automatic path optimization."""
        if not self.defects:
            return "No defects to plan repairs for. Please scan first."
        
        self.ui_commands.append(UICommand(type="TRIGGER_PLAN"))
        
        # Automatically optimize path if available
        optimization_msg = ""
        if HAS_PATH_OPTIMIZER and len(self.defects) > 1:
            result = optimize_with_metrics(self.defects)
            optimization_msg = f"\n\n🚀 **Path Optimization**: {result.get_summary_message()}"
        
        return f"Generating repair plans for {len(self.defects)} detected defects...{optimization_msg}"
    
    def highlight_defect_region(self, defect_index: int) -> str:
        """Highlight a specific defect."""
        if defect_index >= len(self.defects):
            return f"Invalid defect index. Only {len(self.defects)} defects detected."
        
        defect = self.defects[defect_index]
        position = tuple(defect.get("position", (0, 0, 0)))
        
        self.ui_commands.append(UICommand(
            type="HIGHLIGHT",
            position=position,
            defect_index=defect_index
        ))
        
        return f"Highlighted defect #{defect_index + 1}: {defect.get('type', 'unknown')}"
    
    def execute_repair(self) -> str:
        """Execute the repair plan."""
        if not self.plans:
            return "No repair plan to execute. Generate and approve a plan first."
        
        self.ui_commands.append(UICommand(type="EXECUTE"))
        return "Executing repair sequence. Robot arm moving to first waypoint..."
    
    def analyze_visual(self) -> str:
        """Trigger visual inspection of the 3D viewer."""
        self.ui_commands.append(UICommand(type="CAPTURE_SNAPSHOT"))
        return "Capturing screenshot for visual analysis..."
    
    def consult_manual(self, query: str) -> str:
        """Query the SOP knowledge base for repair specifications."""
        if not HAS_KNOWLEDGE_BASE:
            return "📋 Knowledge base not available. Using default parameters."
        
        result = kb_consult(query)
        return result
    
    def predict_repair_metrics(self, defect_index: int = None) -> str:
        """Predict repair time using ML model."""
        if not self.defects:
            return "No defects detected. Please scan the part first."
        
        if not HAS_ML:
            return "ML predictor not available. Please install scikit-learn."
        
        results = []
        
        if defect_index is not None:
            # Predict for specific defect
            if defect_index >= len(self.defects):
                return f"Invalid defect index. Only {len(self.defects)} defects detected."
            
            defect = self.defects[defect_index]
            prediction = ml_predict(defect=defect)
            
            results.append(
                f"**Defect #{defect_index + 1} ({defect.get('type', 'unknown')})**\n"
                f"  ⏱️ Predicted Time: **{prediction['repair_time_seconds']:.1f}s** "
                f"({prediction['confidence_interval']['lower']:.0f}-{prediction['confidence_interval']['upper']:.0f}s range)\n"
                f"  🔧 Consumables: {prediction['consumable_estimate']}"
            )
        else:
            # Predict for all defects
            total_time = 0
            for i, defect in enumerate(self.defects):
                prediction = ml_predict(defect=defect)
                total_time += prediction['repair_time_seconds']
                
                results.append(
                    f"• Defect #{i + 1} ({defect.get('type', 'unknown')}): "
                    f"**{prediction['repair_time_seconds']:.1f}s**"
                )
            
            results.append(f"\n📊 **Total Estimated Time: {total_time:.1f}s** ({total_time/60:.1f} min)")
        
        return "🤖 **ML Prediction Results**\n\n" + "\n".join(results)
    
    def optimize_repair_sequence(self) -> str:
        """Optimize the repair sequence using TSP path optimization."""
        if not self.defects:
            return "No defects detected. Please scan the part first."
        
        if not HAS_PATH_OPTIMIZER:
            return "Path optimizer not available."
        
        if len(self.defects) <= 1:
            return "Only one defect detected - no optimization needed."
        
        result = optimize_with_metrics(self.defects)
        
        # Store optimization result for UI
        self.ui_commands.append(UICommand(
            type="PATH_OPTIMIZED",
            data={
                "original_distance": result.original_distance,
                "optimized_distance": result.optimized_distance,
                "efficiency_gain": result.efficiency_gain_percent,
                "algorithm": result.algorithm_used
            }
        ))
        
        return result.get_summary_message()
    
    def generate_custom_path(self, pattern_description: str, generated_code: str) -> str:
        """Execute LLM-generated path code in sandbox and store result."""
        if not HAS_CODE_INTERPRETER:
            return "Code interpreter not available. Please install dependencies."
        
        # Use a default center and radius for demonstration
        # In production, this would come from the selected defect
        center = (0.5, 0.0, 0.3)
        radius = 0.05
        
        result = exec_custom_path(generated_code, center, radius)
        
        if result.success:
            # Store the custom path for visualization
            self.ui_commands.append(UICommand(
                type="CUSTOM_PATH_GENERATED",
                data={
                    "pattern": pattern_description,
                    "points": result.points.tolist(),
                    "num_points": len(result.points),
                    "code": generated_code
                }
            ))
            return f"✨ **Custom Path Generated**: {pattern_description}\n\n" \
                   f"Created {len(result.points)} waypoints. The path is now displayed on the 3D viewer in magenta.\n\n" \
                   f"```python\n{generated_code}\n```"
        else:
            return f"❌ **Path Generation Failed**: {result.error_message}\n\nPlease check the code and try again."
    
    def execute_tool(self, name: str, arguments: Dict) -> str:
        """Execute a tool by name with arguments."""
        tool_map = {
            "focus_camera_on_defect": self.focus_camera_on_defect,
            "reset_camera_view": self.reset_camera_view,
            "trigger_scan": self.trigger_scan,
            "trigger_repair_plan": self.trigger_repair_plan,
            "highlight_defect_region": self.highlight_defect_region,
            "execute_repair": self.execute_repair,
            "analyze_visual": self.analyze_visual,
            "predict_repair_metrics": self.predict_repair_metrics,
            "consult_manual": self.consult_manual,
            "optimize_repair_sequence": self.optimize_repair_sequence,
            "generate_custom_path": self.generate_custom_path,
        }
        
        if name in tool_map:
            return tool_map[name](**arguments)
        return f"Unknown tool: {name}"


# ============ SUPERVISOR AGENT ============
class SupervisorAgent:
    """
    GPT-4o powered conversational supervisor agent.
    
    Uses OpenAI function calling for natural UI control.
    Falls back to heuristic responses if OpenAI unavailable.
    """
    
    def __init__(self):
        self.client = None
        self.model = "gpt-4o"
        self.conversation_history: List[Dict] = []
        
        # Initialize OpenAI client
        if HAS_OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
    
    def _build_state_context(self, defects: List, plans: List, workflow_step: int, mesh_name: str) -> str:
        """Build context string for the system prompt."""
        context = []
        context.append(f"**Active Part**: {mesh_name or 'None loaded'}")
        context.append(f"**Workflow Step**: {workflow_step}/5 (Load → Scan → Plan → Approve → Execute)")
        
        if defects:
            context.append(f"\n**Defects Detected**: {len(defects)}")
            for i, d in enumerate(defects):
                sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(d.get("severity"), "⚪")
                context.append(f"  {i+1}. {sev_icon} {d.get('type', 'unknown')} ({d.get('severity', 'unknown')}) at {d.get('position', 'unknown')}")
        else:
            context.append("\n**Defects**: Not scanned yet")
        
        if plans:
            context.append(f"\n**Repair Plans**: {len(plans)} ready")
        
        return "\n".join(context)
    
    def _call_vision_api(self, image_base64: str) -> str:
        """Send image to GPT-4o for visual analysis."""
        if not self.client:
            return "Vision API unavailable. Please check OpenAI API key."
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": VISUAL_INSPECTION_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this industrial part for defects."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Vision API error: {e}")
            return f"Visual analysis failed: {str(e)}"
    
    def process_visual_analysis(self, image_base64: str) -> Dict[str, Any]:
        """
        Process a visual inspection request with the captured image.
        
        Args:
            image_base64: Base64-encoded PNG image of the 3D viewer
            
        Returns:
            Dict with content (visual analysis) and metadata
        """
        analysis = self._call_vision_api(image_base64)
        
        # Create a formatted response
        response_text = f"👁️ **Visual Inspection Complete**\n\n{analysis}"
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        return {
            "content": response_text,
            "ui_commands": [],
            "tool_calls": [{"name": "analyze_visual", "args": {}, "result": "analysis_complete"}],
            "agent": "supervisor",
            "avatar": "👁️"
        }
    
    def process_message(
        self,
        message: str,
        defects: List[Dict],
        plans: List[Dict],
        workflow_step: int,
        mesh_name: str,
        image_base64: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and return response with UI commands.
        
        Args:
            message: User's message
            defects: Current defect list
            plans: Current repair plans
            workflow_step: Current workflow step (0-5)
            mesh_name: Name of loaded part
            image_base64: Optional base64 image for visual analysis
            
        Returns:
            Dict with:
                - content: Response text
                - ui_commands: List of UI command dicts
                - tool_calls: List of tools that were called
                - requires_snapshot: True if agent wants to capture image
        """
        # If image provided, do visual analysis
        if image_base64:
            return self.process_visual_analysis(image_base64)
        
        # Build context
        state_context = self._build_state_context(defects, plans, workflow_step, mesh_name)
        system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(state_context=state_context)
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Tool executor
        executor = ToolExecutor(defects, plans, workflow_step)
        tool_calls_made = []
        
        # Try OpenAI if available
        if self.client:
            try:
                response = self._call_openai(system_prompt, executor)
                return response
            except Exception as e:
                print(f"OpenAI error: {e}")
                # Fall through to fallback
        
        # Fallback to heuristic response
        return self._fallback_response(message, defects, plans, workflow_step, executor)
    
    def _call_openai(self, system_prompt: str, executor: ToolExecutor) -> Dict[str, Any]:
        """Make OpenAI API call with function calling."""
        messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
        
        # First API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_message = response.choices[0].message
        tool_calls_made = []
        
        # Handle tool calls
        if assistant_message.tool_calls:
            # Execute each tool
            tool_results = []
            for tool_call in assistant_message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                
                result = executor.execute_tool(fn_name, fn_args)
                tool_calls_made.append({"name": fn_name, "args": fn_args, "result": result})
                
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            
            # Second API call with tool results
            messages.append(assistant_message.model_dump())
            messages.extend(tool_results)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            final_content = response.choices[0].message.content
        else:
            final_content = assistant_message.content
        
        # Update history
        self.conversation_history.append({"role": "assistant", "content": final_content})
        
        # Convert UI commands to dicts
        ui_commands = []
        for cmd in executor.ui_commands:
            ui_commands.append({
                "type": cmd.type,
                "position": cmd.position,
                "defect_index": cmd.defect_index,
                "data": cmd.data
            })
        
        return {
            "content": final_content,
            "ui_commands": ui_commands,
            "tool_calls": tool_calls_made,
            "agent": "supervisor",
            "avatar": "🤖"
        }
    
    def _fallback_response(
        self,
        message: str,
        defects: List,
        plans: List,
        workflow_step: int,
        executor: ToolExecutor
    ) -> Dict[str, Any]:
        """Generate heuristic response when OpenAI unavailable."""
        msg_lower = message.lower()
        response_text = ""
        
        # Greeting
        if any(word in msg_lower for word in ["hello", "hi", "hey"]):
            if defects:
                response_text = f"👋 Hello! I'm AARR, your repair assistant. We have **{len(defects)} defects** detected. Would you like me to show you the most critical one?"
            else:
                response_text = "👋 Hello! I'm AARR, your repair assistant. Load a part and scan it to get started. I'll guide you through the inspection and repair process."
        
        # Show/focus/zoom commands
        elif any(word in msg_lower for word in ["show", "zoom", "focus", "see", "look"]):
            if "worst" in msg_lower or "critical" in msg_lower or "high" in msg_lower:
                result = executor.focus_camera_on_defect(defect_type="any", severity="high")
                response_text = f"🔍 {result}\n\nThis is the highest priority issue. Want me to generate a repair plan?"
            elif any(dtype in msg_lower for dtype in ["crack", "rust", "corrosion", "wear"]):
                for dtype in ["crack", "rust", "corrosion", "wear"]:
                    if dtype in msg_lower:
                        result = executor.focus_camera_on_defect(defect_type=dtype)
                        response_text = f"🔍 {result}"
                        break
            else:
                result = executor.focus_camera_on_defect(defect_type="any")
                response_text = f"🔍 {result}"
        
        # Scan commands
        elif any(word in msg_lower for word in ["scan", "detect", "find"]):
            result = executor.trigger_scan()
            response_text = f"🔍 {result}"
        
        # Plan commands
        elif any(word in msg_lower for word in ["plan", "repair", "fix"]):
            result = executor.trigger_repair_plan()
            response_text = f"🔧 {result}"
        
        # Reset commands
        elif any(word in msg_lower for word in ["reset", "overview", "all"]):
            result = executor.reset_camera_view()
            response_text = f"🔍 {result}"
        
        # Status query
        elif any(word in msg_lower for word in ["status", "what", "how many"]):
            if defects:
                high = sum(1 for d in defects if d.get("severity") == "high")
                response_text = f"📊 **Current Status**\n\n- **Defects**: {len(defects)} ({high} high priority)\n- **Plans**: {'Ready' if plans else 'Not generated'}\n- **Step**: {workflow_step}/5\n\n"
                if high > 0:
                    response_text += "⚠️ I recommend addressing the high-priority defects first. Say 'show worst' to focus on them."
            else:
                response_text = "📊 No part scanned yet. Load a part and click 'Scan' to begin inspection."
        
        # Help
        elif "help" in msg_lower:
            response_text = (
                "🤖 **I'm AARR, your repair assistant!**\n\n"
                "Try saying:\n"
                "- \"Show me the defects\"\n"
                "- \"Focus on the worst crack\"\n"
                "- \"Generate a repair plan\"\n"
                "- \"What's the status?\"\n"
                "- \"Scan the part\"\n\n"
                "I'll control the 3D viewer and guide you through repairs."
            )
        
        # Default
        else:
            if defects:
                response_text = f"I'm here to help with the inspection. We have {len(defects)} defects detected. Would you like me to show them or generate a repair plan?"
            else:
                response_text = "I'm AARR, your repair assistant. Load a part from the sidebar to get started!"
        
        # Update history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Convert commands
        ui_commands = []
        for cmd in executor.ui_commands:
            ui_commands.append({
                "type": cmd.type,
                "position": cmd.position,
                "defect_index": cmd.defect_index
            })
        
        return {
            "content": response_text,
            "ui_commands": ui_commands,
            "tool_calls": [],
            "agent": "supervisor",
            "avatar": "🤖"
        }
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# ============ CONVENIENCE WRAPPER ============
class ConversationalTeam:
    """
    Wrapper that provides the same interface as MultiAgentTeam
    but uses the GPT-4o powered SupervisorAgent.
    """
    
    def __init__(self):
        self.agent = SupervisorAgent()
        self.defects: List[Dict] = []
        self.plans: List[Dict] = []
        self.workflow_step: int = 0
        self.mesh_name: str = "No Part Loaded"
    
    def update_state(
        self,
        defects: List[Dict] = None,
        plans: List[Dict] = None,
        workflow_step: int = None,
        mesh_name: str = None
    ):
        """Update the team's state."""
        if defects is not None:
            self.defects = defects
        if plans is not None:
            self.plans = plans
        if workflow_step is not None:
            self.workflow_step = workflow_step
        if mesh_name is not None:
            self.mesh_name = mesh_name
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message."""
        return self.agent.process_message(
            message=message,
            defects=self.defects,
            plans=self.plans,
            workflow_step=self.workflow_step,
            mesh_name=self.mesh_name
        )
    
    def clear_history(self):
        """Clear chat history."""
        self.agent.clear_history()


# Factory function (backwards compatible)
def create_conversational_agent() -> ConversationalTeam:
    """Create a new conversational agent instance."""
    return ConversationalTeam()
