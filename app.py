import os
import re
import json
import base64
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from groq import Groq
import librosa
import numpy as np


# ============================================================
# ENVIRONMENT SETUP
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing in .env")


# ============================================================
# GROQ CLIENTS & MODELS
# ============================================================

client = Groq(api_key=GROQ_API_KEY)

STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "qwen/qwen3.8-27b"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "austin"  # Options include austin, troy, etc.


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Empathetic Voice Companion with Groq TTS"
)


# ============================================================
# SYSTEM PROMPT (EMPATHETIC & AUDIO-AWARE)
# ============================================================

SYSTEM_PROMPT = """
You are an emotionally intelligent, empathetic, and supportive AI companion for general users. 

You will receive both acoustic vocal metrics (pitch, energy, and pace extracted from the user's actual voice) and the text transcript of their words.

Your goals:
1. **Analyze Emotional State:** Combine the vocal delivery data (e.g., slow pace, quiet energy, monotone vs. expressive pitch) with their literal words to accurately sense their true emotional state (e.g., stressed, joyful, anxious, reflective, lonely, grieving, exhausted, or neutral).
2. **Respond with Empathy:** Validate their feelings first. Match their emotional wavelength—be gentle, grounding, and soothing when they are distressed or low-energy; warm and cheerful when they are happy.
3. **Conversational Flow:** Act as a thoughtful, active listener. Keep responses natural, conversational, and meaningful. Do not sound robotic, formal, or overly clinical.
4. **Safety & Boundaries:** 
   - Never diagnose medical or psychological conditions.
   - If the user expresses severe distress, self-harm, or crisis, gently provide a supportive grounding message and suggest reaching out to professional help or trusted loved ones.
   - Never ask for sensitive private information (passwords, financial records, exact locations).

============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON with no markdown formatting around it.

Exactly:

{
    "detected_emotion": "...",
    "vocal_direction": "...",
    "reply": "...",
    "speech_text": "..."
}

- `detected_emotion`: A short tag describing the user's perceived emotional state (e.g., "stressed", "joyful", "anxious", "exhausted", "neutral").
- `vocal_direction`: A 1-2 word expressive vocal tag for the TTS model matching your response tone (e.g., "[gentle]", "[cheerful]", "[soothing]", "[warm]").
- `reply`: Displayed on screen. Can include subtle, comforting emojis.
- `speech_text`: The spoken text. Do not include brackets here, just the pure sentence to be voiced.
"""


# ============================================================
# AUDIO TONE FEATURE EXTRACTOR
# ============================================================

def analyze_vocal_tone(audio_path: str) -> dict:
    """Extract pitch, speed, and energy metrics from raw user audio."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        # 1. Pitch / Frequency (F0) Analysis
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        avg_pitch = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0
        
        # 2. Energy / Volume (Root Mean Square)
        rms = librosa.feature.rms(y=y)
        avg_energy = float(np.mean(rms))
        
        # 3. Speech Rate / Tempo Estimation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0])

        # Categorize Acoustic Properties
        pitch_level = "high" if avg_pitch > 220 else ("low" if avg_pitch < 120 else "moderate")
        energy_level = "quiet/low-energy" if avg_energy < 0.02 else ("loud/high-energy" if avg_energy > 0.08 else "moderate")
        speech_pace = "fast/agitated" if tempo > 130 else ("slow/hesitant" if tempo < 85 else "steady")

        return {
            "pitch_level": pitch_level,
            "pitch_variability": "monotone" if pitch_std < 20 else "expressive",
            "energy_level": energy_level,
            "speech_pace": speech_pace
        }
    except Exception as e:
        print(f"Acoustic extraction warning: {e}")
        return {
            "pitch_level": "moderate",
            "pitch_variability": "expressive",
            "energy_level": "moderate",
            "speech_pace": "steady"
        }


# ============================================================
# HTML (FRONTEND INTERFACE)
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Empathetic AI Companion</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    min-height: 100vh;
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 20px;
}
.container {
    width: 100%;
    max-width: 650px;
    background: white;
    border-radius: 25px;
    padding: 30px;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.10);
    text-align: center;
}
h1 { margin-top: 0; font-size: 28px; color: #333; }
.subtitle { color: #666; margin-bottom: 25px; }
.mic-button {
    width: 130px;
    height: 130px;
    border-radius: 50%;
    border: none;
    background: #6c5ce7;
    color: white;
    font-size: 45px;
    cursor: pointer;
    transition: 0.2s;
    box-shadow: 0 8px 20px rgba(108, 92, 231, 0.3);
}
.mic-button.recording {
    background: #ff7675;
    transform: scale(1.08);
}
.status { margin-top: 20px; color: #666; min-height: 25px; font-weight: bold; }
.card { margin-top: 20px; padding: 18px; border-radius: 18px; background: #f8f9fa; text-align: left; }
.card-title { font-weight: bold; margin-bottom: 8px; color: #444; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }
.reply { font-size: 16px; line-height: 1.5; color: #2d3436; }
.emotion-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 12px;
    background: #e3f2fd;
    color: #1976d2;
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 10px;
}
.error { color: #d93025; margin-top: 15px; }
</style>
</head>
<body>

<div class="container">
    <h1>💙 Empathetic Companion</h1>
    <div class="subtitle">Press and hold the microphone to share how you're feeling</div>

    <button id="micButton" class="mic-button">🎤</button>
    <div id="status" class="status">Hold the mic and speak</div>

    <div class="card">
        <div class="card-title">You said</div>
        <div id="transcript">--</div>
    </div>

    <div class="card">
        <div class="card-title">AI Companion</div>
        <div id="emotionBadge" class="emotion-badge" style="display:none;"></div>
        <div id="reply" class="reply">--</div>
    </div>

    <div id="error" class="error"></div>
</div>

<script>
let mediaRecorder = null;
let audioChunks = [];

const micButton = document.getElementById("micButton");
const statusElement = document.getElementById("status");
const transcriptElement = document.getElementById("transcript");
const replyElement = document.getElementById("reply");
const emotionBadge = document.getElementById("emotionBadge");
const errorElement = document.getElementById("error");

function playAudioResponse(base64Audio) {
    if (!base64Audio) return;
    const audio = new Audio("data:audio/wav;base64," + base64Audio);
    audio.play().catch(e => console.error("Audio playback error:", e));
}

async function startRecording() {
    try {
        errorElement.textContent = "";
        audioChunks = [];
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        let options = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? { mimeType: "audio/webm;codecs=opus" } : {};
        mediaRecorder = new MediaRecorder(stream, options);

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(track => track.stop());
            const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
            await sendAudio(audioBlob);
        };

        mediaRecorder.start();
        micButton.classList.add("recording");
        statusElement.textContent = "Listening...";
    } catch (error) {
        console.error(error);
        errorElement.textContent = "Microphone permission is required.";
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        micButton.classList.remove("recording");
        statusElement.textContent = "Sensing tone & reflecting...";
    }
}

async function sendAudio(audioBlob) {
    try {
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");

        const response = await fetch("/voice", { method: "POST", body: formData });
        const result = await response.json();

        if (!result.success) throw new Error(result.error || "Something went wrong.");

        transcriptElement.textContent = result.transcript;
        replyElement.textContent = result.reply;
        
        if (result.detected_emotion) {
            emotionBadge.textContent = "Detected State: " + result.detected_emotion;
            emotionBadge.style.display = "inline-block";
        }

        statusElement.textContent = "Ready";
        playAudioResponse(result.audio_base64);
    } catch (error) {
        console.error(error);
        statusElement.textContent = "Something went wrong.";
        errorElement.textContent = error.message;
    }
}

micButton.addEventListener("mousedown", startRecording);
micButton.addEventListener("mouseup", stopRecording);
micButton.addEventListener("mouseleave", stopRecording);
micButton.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
micButton.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });
</script>
</body>
</html>
"""


# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE


# ============================================================
# VOICE API ROUTE
# ============================================================

@app.post("/voice")
async def voice(audio: UploadFile = File(...)):
    temp_path = None
    tts_temp_path = None
    try:
        audio_data = await audio.read()
        if not audio_data:
            return JSONResponse({"success": False, "error": "Audio is empty."}, status_code=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        # 1. Extract Acoustic Tone Metrics from Audio
        vocal_metrics = analyze_vocal_tone(temp_path)

        # 2. Whisper Transcription
        with open(temp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=STT_MODEL,
                response_format="json",
                temperature=0
            )

        transcript = transcription.text.strip()
        if not transcript:
            return JSONResponse({"success": False, "error": "Could not understand audio."}, status_code=400)

        # 3. Build Contextual Payload for Qwen (Audio Metrics + Text)
        user_context_payload = f"""
[USER AUDIO TONE ANALYSIS]
- Pitch Level: {vocal_metrics['pitch_level']}
- Pitch Delivery: {vocal_metrics['pitch_variability']}
- Audio Volume/Energy: {vocal_metrics['energy_level']}
- Speech Pace: {vocal_metrics['speech_pace']}

[TRANSCRIPTION OF WORDS]
"{transcript}"

Instructions: Analyze both the acoustic vocal metrics and the words spoken to determine the user's true emotional state and respond empathetically.
"""

        # 4. Qwen Completion
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_context_payload}
            ],
            temperature=0.5,
            max_completion_tokens=300,
            response_format={"type": "json_object"}
        )

        raw_reply = completion.choices[0].message.content.strip()

        # 5. JSON Parsing
        try:
            ai_data = json.loads(raw_reply)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_reply, re.DOTALL)
            if match:
                try:
                    ai_data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    ai_data = {"detected_emotion": "neutral", "vocal_direction": "[gentle]", "reply": raw_reply, "speech_text": raw_reply}
            else:
                ai_data = {"detected_emotion": "neutral", "vocal_direction": "[gentle]", "reply": raw_reply, "speech_text": raw_reply}

        detected_emotion = str(ai_data.get("detected_emotion", "neutral")).strip()
        vocal_direction = str(ai_data.get("vocal_direction", "[gentle]")).strip()
        reply = str(ai_data.get("reply", "")).strip()
        speech_text = str(ai_data.get("speech_text", "")).strip()

        if not reply:
            reply = "I'm right here with you. How are you feeling right now? 💙"
        if not speech_text:
            speech_text = reply

        # Format input with vocal direction tags for Groq Orpheus TTS model
        tts_input_text = f"{vocal_direction} {speech_text}"

        # 6. Groq Orpheus TTS Generation
        tts_response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=tts_input_text,
            response_format="wav"
        )

        # Save to temp file and encode as Base64 for web delivery
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tts_file:
            tts_temp_path = tts_file.name
        
        tts_response.write_to_file(tts_temp_path)

        with open(tts_temp_path, "rb") as f:
            audio_bytes = f.read()
        
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "success": True,
            "transcript": transcript,
            "detected_emotion": detected_emotion,
            "reply": reply,
            "audio_base64": audio_base64
        }

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        if tts_temp_path and os.path.exists(tts_temp_path):
            try:
                os.unlink(tts_temp_path)
            except Exception:
                pass


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)