# Voice Control Feature

> 🎤 Hands-free control for industrial operators wearing gloves

---

## Overview

The Voice Control feature enables industrial operators to interact with the AARR system using voice commands. This is critical for MRO environments where operators wear protective gloves and cannot type.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VOICE CONTROL PIPELINE                        │
│                                                                  │
│   🎤 Audio Input    →    🔊 Whisper API    →    💬 Agent Chat   │
│   (Browser Mic)          (Transcription)        (Auto-submit)   │
│                                                                  │
│   st.audio_input()  →    whisper-1         →    process_message │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. UI Widget (Streamlit)

```python
audio_data = st.audio_input("🎤 Push to Speak Command", key="voice_input")
```

- **Location**: AI Agent Panel (right column)
- **Behavior**: Click to record, click again to stop
- **Format**: WAV audio bytes

### 2. Transcription Function

```python
def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using OpenAI Whisper API."""
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
```

- **Model**: `whisper-1` (OpenAI's speech-to-text model)
- **Accuracy**: Excellent for English industrial commands
- **Error handling**: Returns empty string on failure

### 3. Auto-Submit Logic

When audio is transcribed:
1. Text appended to chat history with 🎤 avatar
2. Agent state updated with current defects/plans
3. Message sent to Supervisor Agent
4. Response displayed and UI commands executed
5. Page reruns to show updates

---

## Usage

### Prerequisites

1. **OpenAI API Key** in `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

2. **Dependencies**:
   ```bash
   pip install openai>=1.0.0
   ```

### Example Commands

| Voice Command | Agent Action |
|--------------|--------------|
| "Inspect the top corner" | Camera focuses on defect area |
| "Show me the defects" | Lists all detected defects |
| "What repairs are needed?" | Engineer provides repair strategy |
| "Scan the part" | Triggers defect detection |
| "Execute the repair" | Starts repair sequence (if approved) |

---

## Configuration

No additional configuration required. Uses:
- `OPENAI_API_KEY` from environment
- Default `whisper-1` model

---

## Limitations

1. **Requires OpenAI API Key** - Cloud-based transcription
2. **Network dependency** - Needs internet for Whisper API
3. **Browser microphone access** - User must grant permission
4. **English optimized** - Best accuracy with English commands

---

## Future Improvements

- [ ] Local transcription with Whisper.cpp for offline use
- [ ] Wake word detection ("Hey AARR, ...")
- [ ] Continuous listening mode
- [ ] Multi-language support
