"""
Interview Copilot — Real-Time Interview Assistant
Captures internal meeting audio (PipeWire loopback) -> Groq Whisper -> Groq Llama 3.1 -> Elaborated Phone UI
"""

import asyncio
import io
import json
import os
import subprocess
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

load_dotenv()

# ==============================================================================
# Configuration
# ==============================================================================

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
CHUNK_DURATION = 0.5  # seconds
CHUNK_SIZE = int(AUDIO_RATE * AUDIO_CHANNELS * 2 * CHUNK_DURATION)

SILENCE_THRESHOLD = int(os.getenv("SILENCE_THRESHOLD", "300"))
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "1.8"))
MIN_SPEECH_DURATION = 1.0

PW_TARGET = os.getenv("PW_TARGET", "58")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

BASE_DIR = Path(__file__).resolve().parent

def load_context_data() -> tuple[str, str]:
    """Dynamically loads resume and job description files so changes take effect immediately."""
    resume_path = BASE_DIR / "resume.txt"
    jd_path = BASE_DIR / "job_description.txt"

    resume_text = ""
    if resume_path.exists():
        try:
            resume_text = resume_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    if not resume_text:
        resume_text = os.getenv(
            "RESUME_CONTEXT",
            "Software Engineer with experience in Java, Python, Multi-threading, System Design, Microservices, and Linux."
        )

    jd_text = ""
    if jd_path.exists():
        try:
            jd_text = jd_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    return resume_text, jd_text


def build_system_prompt() -> str:
    resume, jd = load_context_data()
    return f"""You are a real-time interview copilot. The candidate reads your answers on a PHONE SCREEN during a live interview — every word must earn its place.

### Candidate:
{resume}

### Role:
{jd if jd else "Senior Software Engineer (Backend & Distributed Systems)"}

### ANSWER FORMAT — Use exactly these sections:

**🎯 SAY THIS FIRST** (1-2 sentences — the confident opening the candidate speaks aloud)

**📌 KEY POINTS** (3-5 bullet points max)
- Each bullet: **Bold keyword** → one sentence explanation
- Use the candidate's real stack (Spring Boot, SQS, Redis, LiveKit, DynamoDB, etc.)
- Mention Samsung/Mastercard experience ONLY when directly relevant

**💻 CODE / DIAGRAM** — Smart trigger rules:
- **LIFECYCLE / FLOW / PIPELINE questions** (e.g., "bean lifecycle", "request lifecycle", "message flow", "how does X work step by step"):
  → ALWAYS include an ASCII diagram showing the sequential steps inside ```text block
  → PLUS 1-line explanation per step below the diagram (bold step name → what happens)
  → Diagram: max 10 lines, under 35 chars wide, use arrows (──►, ▼) and boxes [Component]
- **System design / architecture questions**:
  → ALWAYS include an ASCII component diagram inside ```text block
  → Max 8 lines, under 35 chars wide
- **Code-specific questions** (e.g., "how to implement X", "write code for"):
  → Include code snippet, max 8 lines, production-grade, no boilerplate
- **Simple definition / comparison questions**:
  → SKIP this section entirely
- NEVER include both code AND diagram — pick the one that adds more value

**⚡ GOTCHA** (1-2 one-liner traps or tips — things that impress interviewers)

### STRICT RULES:
1. TOTAL answer must be under 250 words (allow room for diagrams when needed).
2. NO tables, NO trade-off matrices, NO numbered sub-sections.
3. NO filler ("Sure!", "Great question!", "Let me explain...").
4. Write in SPEAKABLE language — what to SAY, not what to READ.
5. Bold the keyword in each bullet so the candidate can scan instantly.
6. Prefer concrete numbers and examples over abstract explanations.
"""

# ==============================================================================
# Phone UI (Dark, responsive, styled headers, auto-scroll, keep-alive)
# ==============================================================================

PHONE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Interview Copilot</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #0a0a0c;
            color: #e4e4e7;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            padding: 10px;
            padding-top: 48px;
            min-height: 100vh;
        }
        #status-bar {
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 40px;
            background: #141418;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 14px;
            font-size: 12px;
            z-index: 100;
            border-bottom: 1px solid #27272a;
        }
        #status-dot {
            width: 9px; height: 9px;
            border-radius: 50%;
            background: #ef4444;
            margin-right: 6px;
            display: inline-block;
        }
        #status-dot.connected { background: #22c55e; }
        #status-dot.listening { background: #22c55e; animation: pulse 1.5s infinite; }
        #status-text { color: #a1a1aa; font-weight: 500; }
        #clock { color: #71717a; font-family: monospace; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        #banner {
            text-align: center;
            padding: 40px 16px;
            color: #71717a;
            font-size: 14px;
        }
        #banner .mic { font-size: 34px; margin-bottom: 12px; }
        .card {
            background: #141418;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 14px;
            border-left: 4px solid #22c55e;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            animation: slide 0.25s ease-out;
        }
        .card.loading { border-left-color: #f59e0b; }
        .card.error { border-left-color: #ef4444; }
        @keyframes slide {
            from { opacity: 0; transform: translateY(-8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .card-meta {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #71717a;
            margin-bottom: 8px;
        }
        .card-q {
            font-size: 14px;
            color: #86efac;
            margin-bottom: 12px;
            line-height: 1.4;
            font-weight: 600;
            background: #18221b;
            padding: 8px 10px;
            border-radius: 6px;
            border: 1px solid #233827;
        }
        .card-a {
            font-size: 14px;
            line-height: 1.6;
            color: #f4f4f5;
        }
        .section-header {
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-top: 14px;
            margin-bottom: 6px;
            color: #f59e0b;
            display: flex;
            align-items: center;
            border-bottom: 1px solid #27272a;
            padding-bottom: 4px;
        }
        .card-a ul { padding-left: 20px; margin: 6px 0; }
        .card-a li { margin-bottom: 8px; }
        .card-a code {
            background: #27272a;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 13px;
            color: #fcd34d;
            font-family: monospace;
        }
        .card-a pre {
            background: #09090b;
            border: 1px solid #27272a;
            padding: 10px 12px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 11px;
            margin: 10px 0;
            color: #38bdf8;
            font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', 'Courier New', monospace;
            line-height: 1.25;
            white-space: pre;
            letter-spacing: -0.2px;
            -webkit-overflow-scrolling: touch;
        }
        .spinner {
            color: #f59e0b;
            font-size: 13px;
            padding: 8px 0;
            animation: pulse 1s infinite;
        }
        #clear-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: #27272a;
            border: 1px solid #3f3f46;
            color: #d4d4d8;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 100;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
        }
        #clear-btn:active { background: #ef4444; color: #fff; }
        #btn-group {
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            z-index: 100;
        }
        #paste-btn {
            height: 48px;
            padding: 0 16px;
            border-radius: 24px;
            background: #2563eb;
            border: 1px solid #3b82f6;
            color: #fff;
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(37,99,235,0.4);
        }
        #paste-btn:active { background: #1d4ed8; }
        #ask-modal {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 200;
            display: flex;
            align-items: flex-end;
        }
        #modal-backdrop {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(2px);
        }
        #modal-content {
            position: relative;
            background: #18181b;
            border-top: 1px solid #3f3f46;
            width: 100%;
            padding: 16px;
            border-radius: 16px 16px 0 0;
            box-shadow: 0 -4px 20px rgba(0,0,0,0.6);
            animation: slideUp 0.2s ease-out;
        }
        @keyframes slideUp {
            from { transform: translateY(100%); }
            to { transform: translateY(0); }
        }
        #modal-input {
            width: 100%;
            height: 120px;
            background: #09090b;
            border: 1px solid #3f3f46;
            border-radius: 8px;
            color: #f4f4f5;
            font-size: 14px;
            padding: 10px;
            resize: none;
            outline: none;
            font-family: inherit;
        }
    </style>
</head>
<body>
    <div id="status-bar">
        <div>
            <span id="status-dot"></span>
            <span id="status-text">Connecting...</span>
        </div>
        <span id="clock"></span>
    </div>

    <div id="content">
        <div id="banner">
            <div class="mic">🎤</div>
            Listening to interviewer audio...
            <div style="font-size: 11px; margin-top: 6px; color: #52525b;">Audio loopback & text/chat ingestion active</div>
        </div>
    </div>

    <div id="btn-group">
        <button id="paste-btn" onclick="openModal()" title="Paste / Ask Question">💬 Paste</button>
        <button id="clear-btn" onclick="clearCards()" title="Clear">🗑</button>
    </div>

    <div id="ask-modal" style="display:none;">
        <div id="modal-backdrop" onclick="closeModal()"></div>
        <div id="modal-content">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="font-weight:700; font-size:14px; color:#f59e0b;">📝 Paste Interview / Chat Question</span>
                <span onclick="closeModal()" style="color:#71717a; font-size:20px; cursor:pointer;">&times;</span>
            </div>
            <textarea id="modal-input" placeholder="Paste coding problem, SQL query, or prompt from Zoom/Meet chat..."></textarea>
            <div style="display:flex; justify-content:space-between; margin-top:10px;">
                <button onclick="closeModal()" style="background:#27272a; color:#a1a1aa; border:none; padding:8px 16px; border-radius:6px; cursor:pointer;">Cancel</button>
                <button onclick="submitModalQuestion()" style="background:#22c55e; color:#000; font-weight:700; border:none; padding:8px 20px; border-radius:6px; cursor:pointer;">Solve ⚡</button>
            </div>
        </div>
    </div>

    <script>
        const content = document.getElementById('content');
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        const clockEl = document.getElementById('clock');
        let ws;

        async function keepAwake() {
            try {
                if ('wakeLock' in navigator) await navigator.wakeLock.request('screen');
            } catch(e){}
        }
        keepAwake();
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') keepAwake();
        });

        setInterval(() => {
            clockEl.textContent = new Date().toLocaleTimeString();
        }, 1000);

        function formatMarkdown(text) {
            let h = text
                .replace(/```(\\w*)\\n([\\s\\S]*?)```/g, '<pre><code>$2</code></pre>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
                .replace(/^### (.*$)/gm, '<div class="section-header">$1</div>')
                .replace(/^## (.*$)/gm, '<div class="section-header" style="color:#60a5fa;">$1</div>')
                .replace(/^[\\-\\*•]\\s+(.+)$/gm, '<li>$1</li>');
            h = h.replace(/((?:<li>.*<\\/li>\\s*)+)/g, '<ul>$1</ul>');
            h = h.replace(/\\n/g, '<br>');
            return h;
        }

        function removeBanner() {
            const b = document.getElementById('banner');
            if (b) b.remove();
        }

        function clearCards() {
            content.innerHTML = `
                <div id="banner">
                    <div class="mic">🎤</div>
                    Listening to interviewer audio...
                </div>
            `;
        }

        function openModal() {
            document.getElementById('ask-modal').style.display = 'flex';
            setTimeout(() => document.getElementById('modal-input').focus(), 100);
        }

        function closeModal() {
            document.getElementById('ask-modal').style.display = 'none';
        }

        function submitModalQuestion() {
            const val = document.getElementById('modal-input').value.trim();
            if (!val) return;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ask', question: val, source: 'phone' }));
                document.getElementById('modal-input').value = '';
                closeModal();
            } else {
                alert('Connecting to server...');
            }
        }

        function connect() {
            const proto = location.protocol === 'https:' ? 'wss' : 'ws';
            ws = new WebSocket(`${proto}://${location.host}/ws`);

            ws.onopen = () => {
                statusDot.className = 'listening';
                statusText.textContent = 'Listening';
            };

            ws.onclose = () => {
                statusDot.className = '';
                statusText.textContent = 'Reconnecting...';
                setTimeout(connect, 2000);
            };

            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);

                if (data.type === 'transcription') {
                    removeBanner();
                    const existing = document.getElementById('loading-card');
                    if (existing) existing.remove();

                    const card = document.createElement('div');
                    card.className = 'card loading';
                    card.id = 'loading-card';
                    card.innerHTML = `
                        <div class="card-q">🎤 "${data.text}"</div>
                        <div class="spinner">⏳ Generating elaborated answer...</div>
                    `;
                    content.prepend(card);
                }

                if (data.type === 'qa') {
                    removeBanner();
                    const existing = document.getElementById('loading-card');
                    if (existing) existing.remove();

                    const card = document.createElement('div');
                    card.className = 'card';
                    card.innerHTML = `
                        <div class="card-meta"><span>${data.time}</span><span>${data.model || ''}</span></div>
                        <div class="card-q">🎤 "${data.question}"</div>
                        <div class="card-a">${formatMarkdown(data.answer)}</div>
                    `;
                    content.prepend(card);

                    if (navigator.vibrate) navigator.vibrate([100, 60, 100]);
                }

                if (data.type === 'error') {
                    const card = document.createElement('div');
                    card.className = 'card error';
                    card.innerHTML = `<div style="color:#ef4444;font-size:13px;">❌ ${data.text}</div>`;
                    content.prepend(card);
                }

                if (data.type === 'status') {
                    statusText.textContent = data.text;
                }
            };
        }

        connect();
    </script>
</body>
</html>"""

# ==============================================================================
# Audio Processing Utilities
# ==============================================================================


def calculate_rms(audio_bytes: bytes) -> float:
    if len(audio_bytes) < 2:
        return 0.0
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def pcm_to_wav(pcm_data: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def transcribe_audio(wav_data: bytes) -> str:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    audio_file = io.BytesIO(wav_data)
    audio_file.name = "audio.wav"

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3-turbo",
        language="en",
        response_format="text",
    )
    return transcription.strip()


# ==============================================================================
# LLM Response Generation (Elaborated + Context Grounded)
# ==============================================================================

transcript_history: deque = deque(maxlen=15)
FALLBACK_MODELS = [
    GROQ_MODEL,
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


def generate_response(question: str) -> tuple[str, str]:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    system_prompt = build_system_prompt()

    history = ""
    if transcript_history:
        recent = list(transcript_history)[-3:]
        history = "\n\nRecent context:\n" + "\n".join(
            f'- Interviewer asked: "{h["question"]}"' for h in recent
        )

    user_message = (
        f"{history}\n\n"
        f'Interviewer just said:\n"{question}"\n\n'
        f"Generate the comprehensive, elaborated 4-part response:"
    )

    last_error = None
    for model_name in dict.fromkeys(FALLBACK_MODELS):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=800,  # Room for diagrams on lifecycle/flow questions
            )
            return response.choices[0].message.content, model_name
        except Exception as err:
            print(f"⚠️ Model '{model_name}' failed: {err}. Trying fallback...")
            last_error = err

    raise last_error or Exception("All LLM models failed.")


# ==============================================================================
# FastAPI App & WebSocket
# ==============================================================================

app = FastAPI(title="Interview Copilot")
connected_clients: list[WebSocket] = []


class AskPayload(BaseModel):
    question: str
    source: str = "chat"


async def process_and_broadcast_question(question: str, source_label: str):
    """Processes a text/chat question and broadcasts progress + answer to phone."""
    time_str = time.strftime("%H:%M:%S")
    await broadcast({
        "type": "transcription",
        "text": f"[{source_label}]: {question}"
    })
    loop = asyncio.get_running_loop()
    try:
        answer, model_used = await loop.run_in_executor(None, generate_response, question)
        transcript_history.append({
            "question": f"[{source_label}] {question}",
            "answer": answer,
            "time": time_str
        })
        await broadcast({
            "type": "qa",
            "question": f"[{source_label}] {question}",
            "answer": answer,
            "time": time_str,
            "model": model_used
        })
        return answer, model_used
    except Exception as e:
        await broadcast({"type": "error", "text": str(e)})
        raise e


@app.get("/")
async def root():
    return HTMLResponse(PHONE_HTML)


@app.get("/health")
async def health():
    return {"status": "ok", "connected_clients": len(connected_clients)}


@app.post("/ask")
async def ask_endpoint(payload: AskPayload):
    q = payload.question.strip()
    if not q:
        return {"status": "error", "message": "Empty question"}
    source_label = "Clipboard" if payload.source == "clipboard" else "Chat Question"
    asyncio.create_task(process_and_broadcast_question(q, source_label))
    return {"status": "processing", "question": q}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"📱 Phone connected! (Total clients: {len(connected_clients)})")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                if data.get("type") == "ask":
                    q = data.get("question", "").strip()
                    if q:
                        asyncio.create_task(process_and_broadcast_question(q, "Phone Input"))
            except Exception:
                pass
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print(f"📱 Phone disconnected. (Total clients: {len(connected_clients)})")


async def broadcast(message: dict):
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in connected_clients:
            connected_clients.remove(ws)


# ==============================================================================
# Audio Capture Loop
# ==============================================================================


def audio_processing_loop(loop: asyncio.AbstractEventLoop):
    cmd = [
        "pw-record",
        "--target", PW_TARGET,
        "--rate", str(AUDIO_RATE),
        "--channels", str(AUDIO_CHANNELS),
        "--format", "s16",
        "--raw",
        "-",
    ]

    print(f"🎧 Starting PipeWire capture on sink {PW_TARGET}...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    audio_buffer = bytearray()
    silence_count = 0
    speech_detected = False
    speech_chunks = 0

    silence_chunks_needed = int(SILENCE_DURATION / CHUNK_DURATION)
    min_speech_chunks = int(MIN_SPEECH_DURATION / CHUNK_DURATION)

    print(f"🎤 Listening on sink {PW_TARGET} (threshold={SILENCE_THRESHOLD})...")

    try:
        while True:
            chunk = process.stdout.read(CHUNK_SIZE)
            if not chunk:
                break

            rms = calculate_rms(chunk)

            if rms > SILENCE_THRESHOLD:
                silence_count = 0
                if not speech_detected:
                    speech_detected = True
                    speech_chunks = 0
                    audio_buffer = bytearray()
                    print("🗣️ Speech started...")
                    asyncio.run_coroutine_threadsafe(
                        broadcast({"type": "status", "text": "🗣️ Listening..."}), loop
                    )

                audio_buffer.extend(chunk)
                speech_chunks += 1
            else:
                if speech_detected:
                    audio_buffer.extend(chunk)
                    silence_count += 1

                    if silence_count >= silence_chunks_needed:
                        if speech_chunks >= min_speech_chunks:
                            duration = speech_chunks * CHUNK_DURATION
                            print(f"⏸️ Speech ended ({duration:.1f}s). Processing...")

                            wav_data = pcm_to_wav(bytes(audio_buffer))

                            try:
                                # 1. Transcribe
                                text = transcribe_audio(wav_data)

                                if text and len(text.strip()) > 5:
                                    print(f"📝 Transcribed: \"{text}\"")
                                    asyncio.run_coroutine_threadsafe(
                                        broadcast({"type": "transcription", "text": text}),
                                        loop,
                                    )

                                    # 2. Generate Elaborated Answer
                                    answer, model_used = generate_response(text)
                                    print(f"✅ Generated answer via {model_used}\n")

                                    time_str = time.strftime("%H:%M:%S")
                                    transcript_history.append({
                                        "question": text,
                                        "answer": answer,
                                        "time": time_str,
                                    })

                                    asyncio.run_coroutine_threadsafe(
                                        broadcast({
                                            "type": "qa",
                                            "question": text,
                                            "answer": answer,
                                            "time": time_str,
                                            "model": model_used,
                                        }),
                                        loop,
                                    )
                                else:
                                    print("⏭️ Transcription empty or too short.")
                            except Exception as e:
                                print(f"❌ Error: {e}")
                                asyncio.run_coroutine_threadsafe(
                                    broadcast({"type": "error", "text": str(e)}), loop
                                )
                        else:
                            print("⏭️ Speech too brief, skipped.")

                        speech_detected = False
                        silence_count = 0
                        speech_chunks = 0
                        audio_buffer = bytearray()

                        asyncio.run_coroutine_threadsafe(
                            broadcast({"type": "status", "text": "Listening"}), loop
                        )
    except Exception as e:
        print(f"Audio loop exception: {e}")
    finally:
        process.terminate()
        process.wait()


@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    threading.Thread(target=audio_processing_loop, args=(loop,), daemon=True).start()


if __name__ == "__main__":
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("gsk_your"):
        print("\n⚠️ Please set your GROQ_API_KEY in the .env file before running!\n")
        exit(1)

    print("=" * 60)
    print("  🎯 INTERVIEW COPILOT (ELABORATED CONTEXT MODE)")
    print(f"  Model       : {GROQ_MODEL}")
    print(f"  PW Target   : {PW_TARGET}")
    print("=" * 60)

    uvicorn.run("server:app", host="0.0.0.0", port=8765, log_level="warning")
