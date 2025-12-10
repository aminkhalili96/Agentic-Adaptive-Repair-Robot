"""
AARR Operator Control Station - Industrial Grade UI

Features:
- Split-view layout (70% 3D Viewer / 30% AI Agent Panel)
- Dark Industrial Theme (Fusion 360 inspired)
- Real-time status metrics bar
- Safety Orange accent for critical actions
"""

import streamlit as st
import numpy as np
import sys
import base64
import tempfile
from pathlib import Path
import plotly.graph_objects as go
from openai import OpenAI

# Page config - must be first
st.set_page_config(
    page_title="AARR Control Station",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports from project modules
from src.visualization.mesh_loader import load_mesh, load_mesh_from_bytes, MeshData, sample_surface_points
from src.visualization.plotly_viewer import Mesh3DViewer
from src.visualization.premium_meshes import generate_premium_meshes, get_premium_defects
from src.visualization.demo_part_generator import generate_demo_part
from src.visualization.gemini_models import (
    generate_aircraft_fuselage,
    generate_complex_pipe_bend,
    generate_saddle_shape,
)
from src.visualization.premium_procedural_models import (
    generate_premium_pipe,
    generate_turbine_blade_v2,
    generate_car_hood_v2,
)
from src.agent.supervisor_agent import ConversationalTeam
from src.config import config

# Import ML predictor for chart
try:
    from src.ml import get_predictor
    HAS_ML_PREDICTOR = True
except ImportError:
    HAS_ML_PREDICTOR = False

# Import SAM segmentor for interactive segmentation
try:
    from src.vision.sam_segmentor import get_segmentor, SAMSegmentor
    HAS_SAM = True
except ImportError:
    HAS_SAM = False

# ============ DARK INDUSTRIAL THEME ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    /* === GLOBAL RESET === */
    .stApp {
        background: #121212 !important;
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header, .stDeployButton {display: none !important; visibility: hidden !important;}
    
    /* Remove default padding */
    .main .block-container {
        padding: 0.5rem 1rem !important;
        max-width: 100% !important;
    }
    
    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: #1E1E1E !important;
        border-right: 1px solid #2D2D2D !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #E0E0E0 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    /* === TYPOGRAPHY === */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    p, span, label, .stMarkdown, div {
        color: #E0E0E0 !important;
    }
    
    /* === BUTTONS === */
    /* Primary Action (Safety Orange) */
    .stButton > button[kind="primary"], 
    .stButton > button {
        background: linear-gradient(135deg, #FF5722 0%, #E64A19 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(255, 87, 34, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #FF7043 0%, #FF5722 100%) !important;
        box-shadow: 0 4px 16px rgba(255, 87, 34, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    .stButton > button:disabled {
        background: #2D2D2D !important;
        color: #666666 !important;
        box-shadow: none !important;
    }
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background: #2D2D2D !important;
        color: #E0E0E0 !important;
        box-shadow: none !important;
    }
    
    /* === TOP METRICS BAR === */
    .metrics-bar {
        background: linear-gradient(180deg, #1E1E1E 0%, #181818 100%);
        border: 1px solid #2D2D2D;
        border-radius: 8px;
        padding: 12px 20px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 32px;
    }
    
    .metric-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .metric-label {
        color: #808080 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .metric-value {
        color: #FFFFFF !important;
        font-size: 14px !important;
        font-weight: 600;
    }
    
    .status-online {
        color: #4CAF50 !important;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #4CAF50;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* === AGENT PANEL === */
    .agent-panel {
        background: #1A1A1A;
        border: 1px solid #2D2D2D;
        border-radius: 12px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .agent-header {
        background: linear-gradient(135deg, #1E1E1E 0%, #252525 100%);
        padding: 16px 20px;
        border-bottom: 1px solid #2D2D2D;
        border-radius: 12px 12px 0 0;
    }
    
    .agent-title {
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin: 0 !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .agent-subtitle {
        color: #808080 !important;
        font-size: 12px !important;
        margin-top: 4px !important;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 12px 16px;
        margin: 8px 12px;
        border-radius: 12px;
        max-width: 85%;
    }
    
    .chat-user {
        background: #2D2D2D;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .chat-agent {
        background: transparent;
        border: 1px solid #2D2D2D;
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    .chat-agent-icon {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #FF5722;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        margin-right: 8px;
    }
    
    /* === 3D VIEWER CONTAINER === */
    .viewer-container {
        background: #0D0D0D;
        border: 1px solid #2D2D2D;
        border-radius: 12px;
        padding: 0;
        overflow: hidden;
    }
    
    .viewer-toolbar {
        background: #1E1E1E;
        padding: 12px 16px;
        border-bottom: 1px solid #2D2D2D;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* === WORKFLOW STEPS === */
    .workflow-step {
        display: flex;
        align-items: center;
        padding: 8px 0;
        gap: 12px;
    }
    
    .step-icon {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 600;
    }
    
    .step-complete {
        background: #1B5E20;
        color: #4CAF50;
    }
    
    .step-active {
        background: #FF5722;
        color: #FFFFFF;
    }
    
    .step-pending {
        background: #2D2D2D;
        color: #666666;
    }
    
    /* === DEFECT CARDS === */
    .defect-card {
        background: #1E1E1E;
        border: 1px solid #2D2D2D;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .defect-card:hover {
        border-color: #FF5722;
        background: #252525;
    }
    
    .severity-high { border-left: 3px solid #F44336 !important; }
    .severity-medium { border-left: 3px solid #FF9800 !important; }
    .severity-low { border-left: 3px solid #4CAF50 !important; }
    
    /* === SCROLLBAR === */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #121212;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3D3D3D;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4D4D4D;
    }
    
    /* === SELECT BOX === */
    .stSelectbox > div > div {
        background: #1E1E1E !important;
        border: 1px solid #2D2D2D !important;
        border-radius: 6px !important;
        color: #E0E0E0 !important;
    }
    
    /* === FILE UPLOADER === */
    [data-testid="stFileUploader"] {
        background: #1E1E1E !important;
        border: 1px dashed #3D3D3D !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #FF5722 !important;
    }
    
    /* === CHAT INPUT === */
    .stChatInput > div {
        background: #1E1E1E !important;
        border: 1px solid #2D2D2D !important;
        border-radius: 8px !important;
    }
    
    .stChatInput input {
        color: #E0E0E0 !important;
    }
    
    /* === RADIO BUTTONS === */
    .stRadio > div {
        gap: 8px !important;
    }
    
    .stRadio label {
        background: #1E1E1E !important;
        border: 1px solid #2D2D2D !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
        color: #E0E0E0 !important;
    }
    
    .stRadio label[data-checked="true"] {
        background: #FF5722 !important;
        border-color: #FF5722 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# ============ SAMPLE PARTS REGISTRY ============
SAMPLE_PARTS = {
    # Premium STL Parts (from file)
    "turbine_blade": {"type": "premium", "name": "Turbine Blade", "desc": "Industrial airfoil"},
    "pipe_assembly": {"type": "premium", "name": "Pipe Assembly", "desc": "Flanged pipe"},
    "precision_gear": {"type": "premium", "name": "Precision Gear", "desc": "Involute gear"},
    "aerospace_bracket": {"type": "premium", "name": "Aerospace Bracket", "desc": "Lightened mount"},
    "robotic_gripper": {"type": "premium", "name": "Robotic Gripper", "desc": "Parallel gripper"},
    # High-Fidelity Procedural Parts (new!)
    "flanged_pipe": {"type": "procedural", "name": "Flanged Pipe", "desc": "Industrial pipe with flanges"},
    "naca_blade": {"type": "procedural", "name": "NACA Turbine Blade", "desc": "Twisted airfoil with heat stress"},
    "car_hood": {"type": "procedural", "name": "Car Hood Panel", "desc": "Solid panel with rust"},
    # Legacy Procedural Parts
    "aircraft_fuselage": {"type": "procedural", "name": "Aircraft Fuselage", "desc": "Cylindrical section"},
    "pipe_bend": {"type": "procedural", "name": "Complex Pipe Bend", "desc": "Pipe elbow"},
    "saddle_shape": {"type": "procedural", "name": "Saddle Shape", "desc": "Hyperbolic surface"},
}


# ============ SESSION STATE ============
def init_state():
    defaults = {
        "mesh_data": None,
        "mesh_display_trace": None,
        "mesh_name": "No Part Loaded",
        "mesh_source": "none",
        "defects": [],
        "defect_normals": [],
        "plans": [],
        "toolpath": [],
        "chat_history": [],
        "highlight_position": None,
        "camera_target": None,
        "camera_eye": dict(x=1.5, y=1.5, z=1.5),  # Direct camera eye control
        "workflow_step": 0,
        "approved": False,
        "agent_team": None,
        "current_part_key": None,
        "pending_visual_request": False,  # Flag for visual analysis pending
        "current_figure": None,  # Store current Plotly figure for snapshot
        # SAM Interactive Segmentation
        "sam_enabled": False,
        "sam_snapshot": None,  # Captured image for segmentation
        "sam_result": None,  # SegmentationResult from SAM
        "sam_click_x": 400,  # Default click X coordinate
        "sam_click_y": 300,  # Default click Y coordinate
        # Custom Path (Code Interpreter)
        "custom_path_points": None,  # Generated path points from LLM code
        "custom_path_info": None,  # Pattern description and metadata
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    
    if st.session_state.agent_team is None:
        st.session_state.agent_team = ConversationalTeam()

init_state()


# ============ HELPER FUNCTIONS ============
def reset_state():
    st.session_state.mesh_data = None
    st.session_state.mesh_display_trace = None
    st.session_state.mesh_name = "No Part Loaded"
    st.session_state.mesh_source = "none"
    st.session_state.defects = []
    st.session_state.defect_normals = []
    st.session_state.plans = []
    st.session_state.toolpath = []
    st.session_state.highlight_position = None
    st.session_state.camera_target = None
    st.session_state.workflow_step = 0
    st.session_state.approved = False
    st.session_state.current_part_key = None
    st.session_state.pending_visual_request = False
    st.session_state.current_figure = None
    st.session_state.sam_enabled = False
    st.session_state.sam_snapshot = None
    st.session_state.sam_result = None
    st.session_state.sam_click_x = 400
    st.session_state.sam_click_y = 300
    st.session_state.custom_path_points = None
    st.session_state.custom_path_info = None


def capture_figure_as_base64(fig: go.Figure) -> str:
    """Capture a Plotly figure as a base64-encoded PNG using Kaleido."""
    try:
        img_bytes = fig.to_image(format="png", width=800, height=600)
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error capturing figure: {e}")
        return None


def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    try:
        client = OpenAI()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            f.flush()
            with open(f.name, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
        return transcript.text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""


def load_sample_part(part_key: str):
    if part_key not in SAMPLE_PARTS:
        return
    
    part_info = SAMPLE_PARTS[part_key]
    reset_state()
    st.session_state.current_part_key = part_key
    st.session_state.mesh_name = part_info["name"]
    
    if part_info["type"] == "premium":
        mesh_dir = Path("assets/premium_meshes")
        mesh_path = mesh_dir / f"{part_key}.stl"
        
        if not mesh_path.exists():
            generate_premium_meshes(str(mesh_dir))
        
        st.session_state.mesh_data = load_mesh(str(mesh_path))
        st.session_state.mesh_source = "premium"
        st.session_state.workflow_step = 1
        
        defects = get_premium_defects(part_key)
        if defects:
            st.session_state.defects = defects
            st.session_state.defect_normals = [d.get('normal', (0, 0, 1)) for d in defects]
            st.session_state.workflow_step = 2
    else:
        generators = {
            # High-Fidelity Models (new)
            "flanged_pipe": generate_premium_pipe,
            "naca_blade": generate_turbine_blade_v2,
            "car_hood": generate_car_hood_v2,
            # Legacy Models
            "auto_body_panel": generate_demo_part,
            "aircraft_fuselage": generate_aircraft_fuselage,
            "pipe_bend": generate_complex_pipe_bend,
            "saddle_shape": generate_saddle_shape,
        }
        if part_key in generators:
            st.session_state.mesh_display_trace = generators[part_key]()
            st.session_state.mesh_source = "procedural"
            st.session_state.workflow_step = 2


def load_uploaded_mesh(uploaded_file):
    reset_state()
    file_type = uploaded_file.name.split('.')[-1]
    mesh_data = load_mesh_from_bytes(
        uploaded_file.getvalue(),
        file_type,
        name=uploaded_file.name.rsplit('.', 1)[0]
    )
    st.session_state.mesh_data = mesh_data
    st.session_state.mesh_name = mesh_data.name
    st.session_state.mesh_source = "upload"
    st.session_state.workflow_step = 1


def perform_scan():
    if not st.session_state.mesh_data:
        return
    positions, normals = sample_surface_points(st.session_state.mesh_data, n_points=3)
    defect_types = ['crack', 'corrosion', 'wear', 'pitting']
    severities = ['high', 'medium', 'low']
    st.session_state.defects = [
        {
            'position': tuple(positions[i]),
            'type': np.random.choice(defect_types),
            'severity': np.random.choice(severities),
            'confidence': np.random.uniform(0.75, 0.98),
            'normal': tuple(normals[i]) if i < len(normals) else (0, 0, 1)
        }
        for i in range(len(positions))
    ]
    st.session_state.defect_normals = [d['normal'] for d in st.session_state.defects]
    st.session_state.workflow_step = 2


def generate_plans():
    from src.agent.tools import get_fallback_plan
    plans = []
    for i, defect in enumerate(st.session_state.defects):
        plan = get_fallback_plan(defect['type'])
        plans.append({"index": i, "defect_type": defect['type'], **plan})
    st.session_state.plans = plans
    st.session_state.workflow_step = 3


# ============ SIDEBAR (Compact Part Selection) ============
with st.sidebar:
    st.markdown("### AARR")
    st.caption("Agentic Adaptive Repair Robot")
    st.markdown("---")
    
    st.markdown("##### Part Selection")
    
    part_mode = st.radio("Mode", ["Sample Parts", "Upload"], horizontal=True, key="part_mode", label_visibility="collapsed")
    
    if part_mode == "Sample Parts":
        options = [""] + list(SAMPLE_PARTS.keys())
        selection = st.selectbox(
            "Select Part",
            options=options,
            format_func=lambda x: "Select..." if x == "" else SAMPLE_PARTS.get(x, {}).get("name", x),
            key="sample_selector",
            label_visibility="collapsed"
        )
        if selection and selection != st.session_state.current_part_key:
            load_sample_part(selection)
            st.rerun()
    else:
        uploaded_file = st.file_uploader("Upload", type=["obj", "stl"], label_visibility="collapsed")
        if uploaded_file and st.session_state.mesh_source != "upload":
            load_uploaded_mesh(uploaded_file)
            st.rerun()
    
    st.markdown("---")
    
    # Workflow Actions
    st.markdown("##### Workflow")
    step = st.session_state.workflow_step
    
    can_scan = st.session_state.mesh_source in ['upload', 'premium'] and step == 1
    if st.button("SCAN PART", disabled=not can_scan, use_container_width=True):
        perform_scan()
        st.rerun()
    
    if st.button("GENERATE PLAN", disabled=step != 2, use_container_width=True):
        generate_plans()
        st.rerun()
    
    if step == 3 and st.checkbox("Approve Plan", value=st.session_state.approved):
        st.session_state.approved = True
        st.session_state.workflow_step = 4
        st.rerun()
    
    # Show ML Prediction chart when plans exist
    if step >= 3 and st.session_state.plans and st.session_state.defects:
        if HAS_ML_PREDICTOR:
            st.markdown("---")
            st.markdown("##### ML Predictions")
            
            # Get predictor and compute chart data
            predictor = get_predictor()
            chart_data = predictor.get_actual_vs_predicted_data(
                st.session_state.defects,
                st.session_state.plans
            )
            
            # Create bar chart comparing predicted vs estimated
            fig = go.Figure()
            
            # Predicted times (ML)
            fig.add_trace(go.Bar(
                name='ML Predicted',
                x=chart_data['labels'],
                y=chart_data['predicted_times'],
                marker_color='#FF5722',
                text=[f"{t:.0f}s" for t in chart_data['predicted_times']],
                textposition='outside',
            ))
            
            # Estimated times (rule-based)
            fig.add_trace(go.Bar(
                name='Rule Estimated',
                x=chart_data['labels'],
                y=chart_data['estimated_times'],
                marker_color='#4CAF50',
                text=[f"{t:.0f}s" for t in chart_data['estimated_times']],
                textposition='outside',
            ))
            
            fig.update_layout(
                title=dict(
                    text='Predicted vs Estimated Time',
                    font=dict(size=12, color='#E0E0E0')
                ),
                barmode='group',
                height=220,
                margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor='#1E1E1E',
                plot_bgcolor='#1E1E1E',
                font=dict(size=9, color='#808080'),
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                ),
                yaxis=dict(
                    title='Time (s)',
                    gridcolor='#2D2D2D',
                    showgrid=True,
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5,
                    font=dict(size=9)
                ),
                showlegend=True,
            )
            
            st.plotly_chart(fig, use_container_width=True, key="ml_prediction_chart")
            
            # Show totals
            total_predicted = sum(chart_data['predicted_times'])
            total_estimated = sum(chart_data['estimated_times'])
            st.caption(f"📊 ML Total: **{total_predicted:.0f}s** | Rule Total: **{total_estimated:.0f}s**")
    
    if st.button("EXECUTE REPAIR", disabled=step != 4, use_container_width=True):
        st.session_state.workflow_step = 5
        st.rerun()
    
    if step >= 5:
        st.success("REPAIR COMPLETE")
    
    # ============ INTERACTIVE SEGMENTATION (SAM) ============
    st.markdown("---")
    st.markdown("##### Interactive Segmentation")
    
    sam_enabled = st.checkbox(
        "Enable Zero-Shot Segmentation",
        value=st.session_state.sam_enabled,
        key="sam_toggle",
        help="Click on the captured view to segment defects using SAM"
    )
    st.session_state.sam_enabled = sam_enabled
    
    if sam_enabled:
        if HAS_SAM:
            segmentor = get_segmentor()
            status = segmentor.get_status()
            if status["model_loaded"]:
                st.caption("MobileSAM loaded")
            else:
                st.caption("Using fallback (OpenCV)")
        else:
            st.caption("SAM not installed")
        
        # Capture snapshot button
        if st.button("Capture View", use_container_width=True):
            if st.session_state.current_figure:
                try:
                    img_bytes = st.session_state.current_figure.to_image(format="png", width=800, height=600)
                    st.session_state.sam_snapshot = img_bytes
                    st.session_state.sam_result = None
                except Exception as e:
                    st.error(f"Capture failed: {e}")
        
        # Show click coordinate inputs when snapshot exists
        if st.session_state.sam_snapshot:
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.sam_click_x = st.number_input(
                    "Click X", min_value=0, max_value=800,
                    value=st.session_state.sam_click_x, step=10
                )
            with col2:
                st.session_state.sam_click_y = st.number_input(
                    "Click Y", min_value=0, max_value=600,
                    value=st.session_state.sam_click_y, step=10
                )
            
            if st.button("SEGMENT", use_container_width=True, type="primary"):
                if HAS_SAM:
                    import cv2
                    # Decode snapshot to numpy
                    nparr = np.frombuffer(st.session_state.sam_snapshot, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Run SAM segmentation
                    segmentor = get_segmentor()
                    result = segmentor.segment_at_point(
                        image,
                        st.session_state.sam_click_x,
                        st.session_state.sam_click_y
                    )
                    st.session_state.sam_result = result
                    st.rerun()
    
    # ============ SYNTHETIC DATA GENERATION ============
    st.markdown("---")
    st.markdown("##### 🧪 Synthetic Data Pipeline")
    st.caption("Generate ML training datasets")
    
    # Only show if mesh is loaded
    can_generate = st.session_state.mesh_data is not None
    
    # Number of samples slider
    num_samples = st.slider(
        "Samples to generate",
        min_value=5,
        max_value=100,
        value=50,
        step=5,
        disabled=not can_generate,
        help="Number of training samples with randomized camera/lighting/defects"
    )
    
    if st.button("🏭 Generate Training Data", use_container_width=True, disabled=not can_generate):
        try:
            from src.simulation.synthetic_data_gen import generate_dataset
            
            output_dir = "synthetic_data"
            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()
            
            def update_progress(current, total):
                progress = current / total
                progress_bar.progress(progress, text=f"Generating sample {current}/{total}")
                status_text.caption(f"🔄 Rendering scene with randomized camera/lighting...")
            
            # Generate dataset
            results = generate_dataset(
                mesh_data=st.session_state.mesh_data,
                num_samples=num_samples,
                output_dir=output_dir,
                progress_callback=update_progress
            )
            
            progress_bar.progress(1.0, text="Complete!")
            status_text.empty()
            st.success(f"✅ Generated {len(results)} samples in `{output_dir}/`")
            st.caption(f"📁 Images: `image_*.png` | Masks: `mask_*.png` | Meta: `metadata_*.json`")
            
        except Exception as e:
            st.error(f"Generation failed: {e}")
    
    if not can_generate:
        st.caption("⚠️ Load a mesh first (STL/OBJ upload)")
    
    if step > 0:
        st.markdown("---")
        if st.button("↻ Reset", use_container_width=True):
            reset_state()
            st.session_state.chat_history = []
            st.rerun()


# ============ MAIN LAYOUT (70/30 Split) ============

# Top Metrics Bar
st.markdown(f"""
<div class="metrics-bar">
    <div class="metric-item">
        <span class="status-dot"></span>
        <span class="metric-label">System Status</span>
        <span class="metric-value status-online">ONLINE</span>
    </div>
    <div class="metric-item">
        <span class="metric-label">Connected Robot</span>
        <span class="metric-value">UR5e</span>
    </div>
    <div class="metric-item">
        <span class="metric-label">Active Part</span>
        <span class="metric-value">{st.session_state.mesh_name}</span>
    </div>
    <div class="metric-item">
        <span class="metric-label">Defects</span>
        <span class="metric-value">{len(st.session_state.defects)}</span>
    </div>
    <div class="metric-item">
        <span class="metric-label">Workflow</span>
        <span class="metric-value">Step {st.session_state.workflow_step + 1}/5</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main 2-Column Layout
col_3d, col_chat = st.columns([0.7, 0.3])

# ============ LEFT COLUMN: 3D VIEWER ============
with col_3d:
    # Procedural mesh
    if st.session_state.mesh_display_trace is not None:
        fig = go.Figure(data=[st.session_state.mesh_display_trace])
        
        # Read camera state from session (connects chatbot brain to viewer eyes)
        camera_eye = st.session_state.get('camera_eye', dict(x=1.5, y=1.5, z=1.5))
        
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                bgcolor='#0D0D0D',
                xaxis=dict(visible=False, showbackground=False),
                yaxis=dict(visible=False, showbackground=False),
                zaxis=dict(visible=False, showbackground=False),
                camera=dict(eye=camera_eye),  # Apply camera from session state
            ),
            paper_bgcolor='#0D0D0D',
            plot_bgcolor='#0D0D0D',
            height=600,
            margin=dict(l=0, r=0, b=0, t=0),
        )
        
        # Add custom path visualization if exists
        if st.session_state.custom_path_points:
            points = st.session_state.custom_path_points
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                z=[p[2] for p in points],
                mode='lines+markers',
                line=dict(color='#FF00FF', width=4),
                marker=dict(size=4, color='#FF00FF'),
                name=st.session_state.custom_path_info.get('pattern', 'Custom Path') if st.session_state.custom_path_info else 'Custom Path'
            ))
        
        st.plotly_chart(fig, use_container_width=True, key="viewer_3d")
        # Store figure for visual inspection
        st.session_state.current_figure = fig
    
    # Standard mesh
    elif st.session_state.mesh_data is not None:
        viewer = Mesh3DViewer(st.session_state.mesh_data)
        
        if st.session_state.defects:
            positions = [d['position'] for d in st.session_state.defects]
            labels = [d['type'].capitalize() for d in st.session_state.defects]
            severities = [d['severity'] for d in st.session_state.defects]
            viewer.add_defect_markers(positions, labels, severities, st.session_state.defect_normals)
        
        if st.session_state.highlight_position:
            viewer.highlight_region(st.session_state.highlight_position, radius=0.015)
        
        fig = viewer.create_figure()
        fig.update_layout(
            scene=dict(bgcolor='#0D0D0D'),
            paper_bgcolor='#0D0D0D',
            height=600,
        )
        
        if st.session_state.camera_target:
            camera = viewer.set_camera_view(st.session_state.camera_target)
            fig.update_layout(scene_camera=camera)
            # Also update camera_eye for consistency
            st.session_state.camera_eye = camera.get('eye', dict(x=1.5, y=1.5, z=1.5))
        else:
            # Default camera from session state
            camera_eye = st.session_state.get('camera_eye', dict(x=1.5, y=1.5, z=1.5))
            fig.update_layout(scene_camera=dict(eye=camera_eye))
        
        # Add custom path visualization if exists
        if st.session_state.custom_path_points:
            points = st.session_state.custom_path_points
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                z=[p[2] for p in points],
                mode='lines+markers',
                line=dict(color='#FF00FF', width=4),
                marker=dict(size=4, color='#FF00FF'),
                name=st.session_state.custom_path_info.get('pattern', 'Custom Path') if st.session_state.custom_path_info else 'Custom Path'
            ))
        
        st.plotly_chart(fig, use_container_width=True, key="viewer_mesh")
        # Store figure for visual inspection
        st.session_state.current_figure = fig
        
        # Part stats
        mesh = st.session_state.mesh_data
        dims = mesh.bounds[1] - mesh.bounds[0]
        st.markdown(f"""
        <div style="display: flex; gap: 24px; padding: 8px 0; color: #808080; font-size: 11px; font-family: 'IBM Plex Mono', monospace;">
            <span>VERTICES: {len(mesh.vertices):,}</span>
            <span>FACES: {len(mesh.faces):,}</span>
            <span>SIZE: {dims[0]*1000:.1f} × {dims[1]*1000:.1f} × {dims[2]*1000:.1f} mm</span>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div style="background: #0D0D0D; border: 1px solid #2D2D2D; border-radius: 12px; 
             height: 600px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <span style="font-size: 64px; margin-bottom: 16px; opacity: 0.3;">🔧</span>
            <span style="color: #808080; font-size: 18px; font-weight: 500;">No Part Loaded</span>
            <span style="color: #4D4D4D; font-size: 13px; margin-top: 8px;">Select a sample part or upload a 3D model</span>
        </div>
        """, unsafe_allow_html=True)
    
    # ============ SEGMENTATION RESULT DISPLAY ============
    if st.session_state.sam_enabled and (st.session_state.sam_snapshot or st.session_state.sam_result):
        st.markdown("---")
        st.markdown("##### ✂️ Interactive Segmentation View")
        
        seg_col1, seg_col2 = st.columns(2)
        
        with seg_col1:
            st.markdown("**Captured Snapshot**")
            if st.session_state.sam_snapshot:
                st.image(
                    st.session_state.sam_snapshot,
                    caption=f"Click point: ({st.session_state.sam_click_x}, {st.session_state.sam_click_y})",
                    use_container_width=True
                )
        
        with seg_col2:
            st.markdown("**Segmentation Result**")
            if st.session_state.sam_result:
                result = st.session_state.sam_result
                st.image(
                    result.overlay,
                    caption=f"Coverage: {result.coverage_percent:.2f}% | Confidence: {result.confidence:.1%}",
                    use_container_width=True
                )
                
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Coverage", f"{result.coverage_percent:.2f}%")
                col_m2.metric("Confidence", f"{result.confidence:.1%}")
                col_m3.metric("Click", f"({result.click_point[0]}, {result.click_point[1]})")
            else:
                st.info("👆 Enter coordinates in sidebar and click SEGMENT")


# ============ RIGHT COLUMN: AI AGENT PANEL ============
with col_chat:
    st.markdown("""
    <div class="agent-header">
        <div class="agent-title">
            <span style="font-size: 20px;">🤖</span>
            Factory Intelligence Unit
        </div>
        <div class="agent-subtitle">Supervisor · Inspector · Engineer</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container(height=480)
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align: center; padding: 40px 20px; color: #4D4D4D;">
                <div style="font-size: 32px; margin-bottom: 12px;">💬</div>
                <div>Ask the AI team about defects, repairs, or inspections.</div>
                <div style="margin-top: 16px; font-size: 12px; color: #3D3D3D;">
                    Try: "Show me the defects" or "What repairs are needed?"
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message chat-user">
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    avatar = msg.get("avatar", "🤖")
                    st.markdown(f"""
                    <div class="chat-message chat-agent">
                        <span class="chat-agent-icon">{avatar}</span>
                        {msg["content"]}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Voice Input (for glove-wearing operators)
    audio_data = st.audio_input("Push to Speak Command", key="voice_input")
    
    if st.button("Send Voice Command", disabled=audio_data is None, use_container_width=True):
        if audio_data:
            # Transcribe audio using Whisper API
            transcribed_text = transcribe_audio(audio_data.getvalue())
            
            if transcribed_text.strip():
                # Add to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": transcribed_text,
                    "avatar": "User"
                })
                
                # Update agent state and process message
                st.session_state.agent_team.update_state(
                    defects=st.session_state.defects,
                    plans=st.session_state.plans,
                    workflow_step=st.session_state.workflow_step,
                    mesh_name=st.session_state.mesh_name
                )
                
                response = st.session_state.agent_team.process_message(transcribed_text)
                
                # Remove emoji from avatar if present
                avatar_text = response["avatar"]
                if avatar_text == "🤖": avatar_text = "AI"
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response["content"],
                    "avatar": avatar_text,
                    "agent": response["agent"]
                })
            
            # Handle UI commands (same as text input)
            for cmd in response.get("ui_commands", []):
                cmd_type = cmd.get("type", "")
                
                if cmd_type in ["FOCUS_CAMERA", "ZOOM_TO", "HIGHLIGHT_DEFECT"] and cmd.get("position"):
                    st.session_state.highlight_position = cmd["position"]
                    st.session_state.camera_target = cmd["position"]
                
                if cmd_type == "HIGHLIGHT" and cmd.get("defect_index") is not None:
                    idx = cmd["defect_index"]
                    if idx < len(st.session_state.defects):
                        st.session_state.highlight_position = st.session_state.defects[idx]["position"]
                        st.session_state.camera_target = st.session_state.defects[idx]["position"]
                
                if cmd_type == "RESET_VIEW":
                    st.session_state.highlight_position = None
                    st.session_state.camera_target = None
                
                if cmd_type == "TRIGGER_SCAN":
                    if st.session_state.mesh_source in ['upload', 'premium']:
                        perform_scan()
                
                if cmd_type == "TRIGGER_PLAN":
                    if st.session_state.defects:
                        generate_plans()
                
                if cmd_type == "EXECUTE":
                    if st.session_state.plans and st.session_state.approved:
                        st.session_state.workflow_step = 5
                
                # Custom path generated by Code Interpreter
                if cmd_type == "CUSTOM_PATH_GENERATED":
                    data = cmd.get("data", {})
                    if data.get("points"):
                        st.session_state.custom_path_points = data["points"]
                        st.session_state.custom_path_info = {
                            "pattern": data.get("pattern", "Custom"),
                            "num_points": data.get("num_points", len(data["points"]))
                        }
            
            st.rerun()
    
    # Chat input (text fallback)
    user_input = st.chat_input("Ask the team...", key="agent_chat")
    
    if user_input:
        st.session_state.chat_history.append({
            "role": "user", 
            "content": user_input,
            "avatar": "User"
        })
        
        st.session_state.agent_team.update_state(
            defects=st.session_state.defects,
            plans=st.session_state.plans,
            workflow_step=st.session_state.workflow_step,
            mesh_name=st.session_state.mesh_name
        )
        
        response = st.session_state.agent_team.process_message(user_input)
        
        # Remove emoji from avatar if present
        avatar_text = response["avatar"]
        if avatar_text == "🤖": avatar_text = "AI"

        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response["content"],
            "avatar": avatar_text,
            "agent": response["agent"]
        })
        
        # Handle UI commands from supervisor
        for cmd in response.get("ui_commands", []):
            cmd_type = cmd.get("type", "")
            
            # Camera focus/zoom
            if cmd_type in ["FOCUS_CAMERA", "ZOOM_TO", "HIGHLIGHT_DEFECT"] and cmd.get("position"):
                st.session_state.highlight_position = cmd["position"]
                st.session_state.camera_target = cmd["position"]
            
            # Highlight specific defect
            if cmd_type == "HIGHLIGHT" and cmd.get("defect_index") is not None:
                idx = cmd["defect_index"]
                if idx < len(st.session_state.defects):
                    st.session_state.highlight_position = st.session_state.defects[idx]["position"]
                    st.session_state.camera_target = st.session_state.defects[idx]["position"]
            
            # Reset view
            if cmd_type == "RESET_VIEW":
                st.session_state.highlight_position = None
                st.session_state.camera_target = None
            
            # Trigger scan
            if cmd_type == "TRIGGER_SCAN":
                if st.session_state.mesh_source in ['upload', 'premium']:
                    perform_scan()
            
            # Trigger plan
            if cmd_type == "TRIGGER_PLAN":
                if st.session_state.defects:
                    generate_plans()
            
            # Execute repair
            if cmd_type == "EXECUTE":
                if st.session_state.plans and st.session_state.approved:
                    st.session_state.workflow_step = 5
            
            # Custom path generated by Code Interpreter
            if cmd_type == "CUSTOM_PATH_GENERATED":
                data = cmd.get("data", {})
                if data.get("points"):
                    st.session_state.custom_path_points = data["points"]
                    st.session_state.custom_path_info = {
                        "pattern": data.get("pattern", "Custom"),
                        "num_points": data.get("num_points", len(data["points"]))
                    }
            
            # Visual inspection - capture snapshot and analyze
            if cmd_type == "CAPTURE_SNAPSHOT":
                if st.session_state.current_figure is not None:
                    # Capture the current figure as base64 image
                    image_base64 = capture_figure_as_base64(st.session_state.current_figure)
                    if image_base64:
                        # Call agent again with the image for visual analysis
                        st.session_state.agent_team.update_state(
                            defects=st.session_state.defects,
                            plans=st.session_state.plans,
                            workflow_step=st.session_state.workflow_step,
                            mesh_name=st.session_state.mesh_name
                        )
                        visual_response = st.session_state.agent_team.agent.process_message(
                            message="Analyze this image",
                            defects=st.session_state.defects,
                            plans=st.session_state.plans,
                            workflow_step=st.session_state.workflow_step,
                            mesh_name=st.session_state.mesh_name,
                            image_base64=image_base64
                        )
                        # Replace the "capturing..." message with actual analysis
                        if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                            st.session_state.chat_history[-1] = {
                                "role": "assistant",
                                "content": visual_response["content"],
                                "avatar": visual_response["avatar"],
                                "agent": visual_response["agent"]
                            }
                    else:
                        # Update last message with error
                        if st.session_state.chat_history:
                            st.session_state.chat_history[-1]["content"] = "❌ Could not capture the 3D view. Please ensure a part is loaded."
                else:
                    # No figure available
                    if st.session_state.chat_history:
                        st.session_state.chat_history[-1]["content"] = "❌ No part loaded to analyze. Please load a part first."
        
        st.rerun()