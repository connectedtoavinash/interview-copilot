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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

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
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
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
- **Code-specific questions** (e.g., "how to implement X", "write code for", LeetCode/algorithm/SQL):
  → ALWAYS wrap inside markdown code block with language (e.g. ```java, ```sql, ```python)
  → Use 100% syntactically correct code with complete types and generics (e.g. List<List<Integer>>, Map<Integer, Integer>)
  → Clean, production-grade, no boilerplate
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


def build_detailed_system_prompt() -> str:
    resume, jd = load_context_data()
    return f"""You are an elite, principal-level technical interview copilot assisting a candidate in a senior engineering interview.
The candidate selected DETAILED DEEP-DIVE MODE to deliver an authoritative, masterclass response with architectural depth and senior engineering nuance.

### Candidate Background:
{resume}

### Target Role Context:
{jd if jd else "Senior Software Engineer (Backend, Concurrency & Distributed Systems)"}

### DETAILED ANSWER STRUCTURE — Format in these 4 comprehensive sections:

### 🎯 1. EXECUTIVE SUMMARY & THESIS
- Direct, confident, senior-level opening answering the question with precise architectural terminology.
- State the industry best practice approach immediately.

### 🧠 2. DEEP-DIVE & INTERNAL ARCHITECTURE
- Explain the underlying engineering mechanics (e.g. JVM memory layout, garbage collection impact, kernel scheduling, lock primitives & atomic CAS, network protocols, distributed consensus, CAP/PACELC trade-offs).
- Detail latency vs throughput, consistency vs availability, CPU cache friendliness, and database indexing implications.
- Contrast naive approaches with production-grade architectures.

### 💻 3. PRODUCTION IMPLEMENTATION & ARCHITECTURE PATTERN
- If architectural or flow-based: provide a clean plain-text ASCII Diagram / Flowchart (under 40 chars wide) inside ```text block, tagging actual technologies (Spring Boot, SQS, Redis, LiveKit, DynamoDB, PostgreSQL, etc.).
- If code-based: provide clean, robust, production-grade code wrapped inside markdown code blocks (```java, ```sql, ```python) with complete syntax and generic types (e.g. List<List<Integer>>, Map<Integer, Integer>).
- Explicitly connect to how the candidate solved similar challenges at Samsung or Mastercard.

### 🛡️ 4. TRADE-OFFS, FAILURE MODES & SENIOR GOTCHAS
- 3-4 real-world production traps (e.g., thread starvation, deadlocks, connection pool exhaustion, unhandled rejected execution, cache stampede, split-brain).
- Mitigation strategies, resiliency patterns (circuit breaker, rate limiter, backpressure), and observability metrics.

### GUIDELINES:
- Deliver deep technical substance, highly organized and professional.
- Do NOT use filler greetings ("Sure!", "Great question!"). Begin immediately with Section 1.
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
        .card-actions {
            display: flex;
            gap: 8px;
            margin-top: 6px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }
        .action-pill {
            background: #18181b;
            border: 1px solid #3f3f46;
            color: #d4d4d8;
            font-size: 11px;
            font-weight: 600;
            padding: 5px 12px;
            border-radius: 14px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            transition: all 0.15s ease;
            user-select: none;
            -webkit-tap-highlight-color: transparent;
        }
        .action-pill:active { transform: scale(0.96); }
        .action-pill.edit {
            color: #f59e0b;
            border-color: rgba(245, 158, 11, 0.4);
            background: rgba(245, 158, 11, 0.08);
        }
        .action-pill.edit:active { background: rgba(245, 158, 11, 0.25); }
        .action-pill.alt {
            color: #38bdf8;
            border-color: rgba(56, 189, 248, 0.4);
            background: rgba(56, 189, 248, 0.08);
        }
        .action-pill.alt:active { background: rgba(56, 189, 248, 0.25); }
        .action-pill.deep {
            color: #c084fc;
            border-color: rgba(192, 132, 252, 0.4);
            background: rgba(192, 132, 252, 0.08);
        }
        .action-pill.deep:active { background: rgba(192, 132, 252, 0.25); }
        .mode-badge {
            font-size: 10px;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 10px;
            margin-left: 6px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        .mode-badge.normal {
            background: rgba(34, 197, 94, 0.15);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.3);
        }
        .mode-badge.detailed {
            background: rgba(192, 132, 252, 0.15);
            color: #c084fc;
            border: 1px solid rgba(192, 132, 252, 0.3);
        }
        #mode-toggle-group {
            display: inline-flex;
            background: #09090b;
            padding: 2px;
            border-radius: 8px;
            border: 1px solid #27272a;
        }
        .mode-toggle-btn {
            background: transparent;
            border: none;
            color: #71717a;
            font-size: 11px;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
            -webkit-tap-highlight-color: transparent;
        }
        .mode-toggle-btn#mode-btn-normal.active {
            color: #22c55e;
            background: #142918;
            border: 1px solid rgba(34, 197, 94, 0.4);
        }
        .mode-toggle-btn#mode-btn-detailed.active {
            color: #c084fc;
            background: #28143d;
            border: 1px solid rgba(192, 132, 252, 0.4);
        }
        .inline-alt-btn {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid #0284c7;
            color: #38bdf8;
            font-size: 11px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 12px;
            cursor: pointer;
            margin-left: auto;
            user-select: none;
            -webkit-tap-highlight-color: transparent;
        }
        .inline-alt-btn:active {
            background: rgba(56, 189, 248, 0.3);
            transform: scale(0.96);
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
        <div style="display:flex; align-items:center;">
            <span id="status-dot"></span>
            <span id="status-text">Connecting...</span>
        </div>
        <div id="mode-toggle-group">
            <button id="mode-btn-normal" class="mode-toggle-btn active" onclick="switchGlobalMode('normal')">⚡ Normal</button>
            <button id="mode-btn-detailed" class="mode-toggle-btn" onclick="switchGlobalMode('detailed')">🧠 Detailed</button>
        </div>
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
                <span id="modal-title" style="font-weight:700; font-size:14px; color:#f59e0b;">📝 Paste Interview / Chat Question</span>
                <span onclick="closeModal()" style="color:#71717a; font-size:20px; cursor:pointer;">&times;</span>
            </div>
            <textarea id="modal-input" placeholder="Paste or edit coding problem, SQL query, or prompt from Zoom/Meet chat..."></textarea>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; gap:8px;">
                <button onclick="closeModal()" style="background:#27272a; color:#a1a1aa; border:none; padding:8px 14px; border-radius:6px; cursor:pointer;">Cancel</button>
                <div style="display:flex; gap:8px;">
                    <button id="modal-alt-btn" onclick="submitModalQuestion(true)" style="background:#0284c7; color:#fff; font-weight:700; border:none; padding:8px 14px; border-radius:6px; cursor:pointer;">Alt Way 🔄</button>
                    <button id="modal-submit-btn" onclick="submitModalQuestion(false)" style="background:#22c55e; color:#000; font-weight:700; border:none; padding:8px 18px; border-radius:6px; cursor:pointer;">Solve ⚡</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const content = document.getElementById('content');
        const statusDot = document.getElementById('status-dot');
        const statusText = document.getElementById('status-text');
        const cardQuestions = {};
        let currentAppMode = 'normal';
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

        function switchGlobalMode(mode) {
            currentAppMode = mode;
            updateModeButtons(mode);
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'set_mode', mode: mode }));
            } else {
                fetch('/mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: mode })
                }).catch(e => console.error(e));
            }
        }

        function updateModeButtons(mode) {
            const btnNorm = document.getElementById('mode-btn-normal');
            const btnDet = document.getElementById('mode-btn-detailed');
            if (btnNorm) btnNorm.className = 'mode-toggle-btn' + (mode === 'normal' ? ' active' : '');
            if (btnDet) btnDet.className = 'mode-toggle-btn' + (mode === 'detailed' ? ' active' : '');
        }

        function escapeHtml(str) {
            if (!str) return '';
            return str
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        function cleanQuestionText(q) {
            if (!q) return '';
            return q
                .replace(/^\\[.*?\\]\\s*/, '')
                .replace(/^🎤\\s*"?/, '')
                .replace(/"?\\s*$/, '')
                .trim();
        }

        function formatMarkdown(text, cardId) {
            if (!text) return '';

            // 1. Extract and escape fenced code blocks
            const codeBlocks = [];
            let h = text.replace(/```(\\w*)\\n([\\s\\S]*?)```/g, (match, lang, code) => {
                const id = '___CODE_BLOCK_' + codeBlocks.length + '___';
                const escaped = code
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');
                codeBlocks.push('<pre><code>' + escaped + '</code></pre>');
                return id;
            });

            // 2. Extract and escape inline code
            const inlineCodes = [];
            h = h.replace(/`([^`]+)`/g, (match, code) => {
                const id = '___INLINE_CODE_' + inlineCodes.length + '___';
                const escaped = code
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');
                inlineCodes.push('<code>' + escaped + '</code>');
                return id;
            });

            // 3. Escape raw < and > so generic types (e.g. List<Integer>) are not swallowed by the browser as HTML tags
            h = h.replace(/</g, '&lt;').replace(/>/g, '&gt;');

            // 4. Format section headers (Normal & Detailed modes)
            h = h
                .replace(/(?:###\\s*|\\*\\*)?(🎯 SAY THIS FIRST[^*<\\n]*)(?:\\*\\*)?/gi, '<div class="section-header" style="color:#f59e0b;">$1</div>')
                .replace(/(?:###\\s*|\\*\\*)?(📌 KEY POINTS[^*<\\n]*)(?:\\*\\*)?/gi, '<div class="section-header" style="color:#38bdf8;">$1</div>')
                .replace(/(?:###\\s*|\\*\\*)?(💻 CODE \\/ DIAGRAM[^*<\\n]*)(?:\\*\\*)?/gi, (match, p1) => {
                    return `<div class="section-header" style="color:#a78bfa; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;"><span>${p1}</span>${cardId ? `<button class="inline-alt-btn" onclick="triggerAlt('${cardId}')">🔄 Rewrite another way</button>` : ''}</div>`;
                })
                .replace(/(?:###\\s*|\\*\\*)?(⚡ GOTCHA[^*<\\n]*)(?:\\*\\*)?/gi, '<div class="section-header" style="color:#fbbf24;">$1</div>')
                // Detailed Mode headers
                .replace(/(?:###\\s*|\\*\\*)?(🎯 1\\.\\s*EXECUTIVE SUMMARY[^*<\\n]*)(?:\\*\\*)?/gi, '<div class="section-header" style="color:#f59e0b;">$1</div>')
                .replace(/(?:###\\s*|\\*\\*)?(🧠 2\\.\\s*DEEP-DIVE[^*<\\n]*)(?:\\*\\*)?/gi, '<div class="section-header" style="color:#38bdf8;">$1</div>')
                .replace(/(?:###\\s*|\\*\\*)?(💻 3\\.\\s*PRODUCTION IMPLEMENTATION[^*<\\n]*)(?:\\*\\*)?/gi, (match, p1) => {
                    return `<div class="section-header" style="color:#a78bfa; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;"><span>${p1}</span>${cardId ? `<button class="inline-alt-btn" onclick="triggerAlt('${cardId}')">🔄 Rewrite another way</button>` : ''}</div>`;
                })
                .replace(/(?:###\\s*|\\*\\*)?(🛡️ 4\\.\\s*TRADE-OFFS[^*<\\n]*)(?:\\*\\*)?/gi, '<div class="section-header" style="color:#ef4444;">$1</div>')
                // General bold and markdown headers
                .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
                .replace(/^### (.*$)/gm, '<div class="section-header">$1</div>')
                .replace(/^## (.*$)/gm, '<div class="section-header" style="color:#60a5fa;">$1</div>')
                .replace(/^[\\-\\*•]\\s+(.+)$/gm, '<li>$1</li>');

            h = h.replace(/((?:<li>.*<\\/li>\\s*)+)/g, '<ul>$1</ul>');
            h = h.replace(/\\n/g, '<br>');

            // 5. Restore escaped code blocks and inline code
            for (let i = 0; i < codeBlocks.length; i++) {
                h = h.replace('___CODE_BLOCK_' + i + '___', codeBlocks[i]);
            }
            for (let i = 0; i < inlineCodes.length; i++) {
                h = h.replace('___INLINE_CODE_' + i + '___', inlineCodes[i]);
            }

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
            document.getElementById('modal-title').textContent = '📝 Paste Interview / Chat Question';
            document.getElementById('modal-input').value = '';
            document.getElementById('ask-modal').style.display = 'flex';
            setTimeout(() => document.getElementById('modal-input').focus(), 100);
        }

        function closeModal() {
            document.getElementById('ask-modal').style.display = 'none';
        }

        function editCardQuestion(cardId) {
            const q = cardQuestions[cardId] || '';
            document.getElementById('modal-title').textContent = '✏️ Edit Question';
            document.getElementById('modal-input').value = q;
            document.getElementById('ask-modal').style.display = 'flex';
            setTimeout(() => {
                const input = document.getElementById('modal-input');
                input.focus();
                input.setSelectionRange(input.value.length, input.value.length);
            }, 100);
        }

        function toggleCardDepth(cardId, targetMode) {
            const q = cardQuestions[cardId];
            if (!q) return;
            const label = targetMode === 'detailed' ? 'Deep Dive' : 'Concise View';
            sendQuestion(q, label, targetMode, false);
        }

        function triggerAlt(cardId) {
            const q = cardQuestions[cardId];
            if (!q) return;
            sendQuestion(q, 'Alt Solution', currentAppMode, true);
        }

        function sendQuestion(q, source, mode, isAlt) {
            const trimmed = q.trim();
            if (!trimmed) return;

            const useMode = mode || currentAppMode;
            removeBanner();
            const existing = document.getElementById('loading-card');
            if (existing) existing.remove();

            const loadCard = document.createElement('div');
            loadCard.className = 'card loading';
            loadCard.id = 'loading-card';
            const altTag = isAlt ? '🔄 Alt Solution for' : '🎤';
            const depthTag = useMode === 'detailed' ? ' [🧠 Detailed]' : '';
            loadCard.innerHTML = `
                <div class="card-q">${altTag}: "${escapeHtml(trimmed)}"${depthTag}</div>
                <div class="spinner">⏳ Generating ${useMode === 'detailed' ? 'detailed architectural' : 'concise'} answer...</div>
            `;
            content.prepend(loadCard);

            const payload = {
                type: 'ask',
                question: trimmed,
                source: source || 'Phone',
                mode: useMode,
                is_alt: Boolean(isAlt)
            };

            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify(payload));
            } else {
                fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                }).catch(err => console.error('Fetch fallback error:', err));
            }
        }

        function submitModalQuestion(isAlt) {
            const val = document.getElementById('modal-input').value.trim();
            if (!val) return;
            const isEdit = document.getElementById('modal-title').textContent.includes('Edit');
            const source = isEdit
                ? (isAlt ? 'Edited Alt' : 'Edited')
                : (isAlt ? 'Alt Solution' : 'Phone Input');
            sendQuestion(val, source, currentAppMode, Boolean(isAlt));
            closeModal();
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

                if (data.type === 'mode_changed') {
                    currentAppMode = data.mode;
                    updateModeButtons(data.mode);
                }

                if (data.type === 'transcription') {
                    removeBanner();
                    const existing = document.getElementById('loading-card');
                    if (existing) existing.remove();

                    const card = document.createElement('div');
                    card.className = 'card loading';
                    card.id = 'loading-card';
                    const depthTag = data.mode === 'detailed' ? ' [🧠 Detailed]' : '';
                    card.innerHTML = `
                        <div class="card-q">🎤 "${escapeHtml(data.text)}"${depthTag}</div>
                        <div class="spinner">⏳ Generating answer...</div>
                    `;
                    content.prepend(card);
                }

                if (data.type === 'qa') {
                    removeBanner();
                    const existing = document.getElementById('loading-card');
                    if (existing) existing.remove();

                    const cardId = 'card_' + Date.now() + '_' + Math.floor(Math.random() * 1000);
                    cardQuestions[cardId] = cleanQuestionText(data.question);
                    const cardMode = data.mode || currentAppMode;
                    const isDet = cardMode === 'detailed';

                    const card = document.createElement('div');
                    card.className = 'card';
                    card.id = cardId;
                    card.innerHTML = `
                        <div class="card-meta">
                            <span>${data.time} <span class="mode-badge ${cardMode}">${isDet ? '🧠 Detailed' : '⚡ Normal'}</span></span>
                            <span>${data.model || ''}</span>
                        </div>
                        <div class="card-q">🎤 "${escapeHtml(data.question)}"</div>
                        <div class="card-actions">
                            <button class="action-pill edit" onclick="editCardQuestion('${cardId}')">✏️ Edit</button>
                            <button class="action-pill alt" onclick="triggerAlt('${cardId}')">🔄 Alt Solution</button>
                            <button class="action-pill deep" onclick="toggleCardDepth('${cardId}', '${isDet ? 'normal' : 'detailed'}')">
                                ${isDet ? '⚡ Make Concise' : '🧠 Deep Dive'}
                            </button>
                        </div>
                        <div class="card-a">${formatMarkdown(data.answer, cardId)}</div>
                    `;
                    content.prepend(card);

                    if (navigator.vibrate) navigator.vibrate([100, 60, 100]);
                }

                if (data.type === 'error') {
                    const card = document.createElement('div');
                    card.className = 'card error';
                    card.innerHTML = `<div style="color:#ef4444;font-size:13px;">❌ ${escapeHtml(data.text)}</div>`;
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
current_answering_mode: str = "normal"  # "normal" or "detailed"


def generate_response(question: str, mode: str = "normal", is_alt: bool = False) -> tuple[str, str]:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    if mode == "detailed":
        system_prompt = build_detailed_system_prompt()
        max_tokens_val = 1600
    else:
        system_prompt = build_system_prompt()
        max_tokens_val = 800

    history = ""
    if transcript_history:
        recent = list(transcript_history)[-3:]
        history = "\n\nRecent context:\n" + "\n".join(
            f'- Interviewer asked: "{h["question"]}"' for h in recent
        )

    if is_alt:
        if mode == "detailed":
            user_message = (
                f"{history}\n\n"
                f'Interviewer requested a DETAILED ALTERNATIVE APPROACH / DIFFERENT ARCHITECTURAL IMPLEMENTATION for:\n"{question}"\n\n'
                f"STRICT INSTRUCTIONS FOR DETAILED ALTERNATIVE:\n"
                f"1. Provide a completely DIFFERENT way to solve this query / code / design problem than the standard approach (e.g. if previous used Window Function, use Subquery/JOIN/MAX; if iterative, use recursive/Stream API; if SQL, provide a distinct query strategy; if design, provide an alternate architecture pattern).\n"
                f"2. In **💻 3. PRODUCTION IMPLEMENTATION & ARCHITECTURE PATTERN**, provide this working alternative code or diagram.\n"
                f"3. In **🧠 2. DEEP-DIVE & INTERNAL ARCHITECTURE**, contrast both paradigms deeply: algorithmic complexity, memory layout, indexing, concurrency behavior, and production scale trade-offs.\n"
                f"4. Follow the comprehensive 4-part structure (EXECUTIVE SUMMARY, DEEP-DIVE, PRODUCTION IMPLEMENTATION, TRADE-OFFS & GOTCHAS)."
            )
        else:
            user_message = (
                f"{history}\n\n"
                f'Interviewer requested an ALTERNATIVE APPROACH / DIFFERENT IMPLEMENTATION for:\n"{question}"\n\n'
                f"STRICT INSTRUCTIONS FOR ALTERNATIVE / REWRITE:\n"
                f"1. Provide a completely DIFFERENT way to solve this query / code / design problem than the standard approach (e.g. if previous used Window Function RANK/DENSE_RANK, use Subquery/JOIN/MAX; if iterative, use recursive/Stream API; if SQL, provide a distinct query strategy; if design, provide an alternate architecture pattern).\n"
                f"2. In **💻 CODE / DIAGRAM**, provide this working alternative code or diagram.\n"
                f"3. In **📌 KEY POINTS**, explain why this alternative exists, its performance/readability trade-offs, and when to choose it over the primary approach.\n"
                f"4. Keep the exact 4-part structure (SAY THIS FIRST, KEY POINTS, CODE / DIAGRAM, GOTCHA) under 250 words."
            )
    else:
        if mode == "detailed":
            user_message = (
                f"{history}\n\n"
                f'Interviewer asked:\n"{question}"\n\n'
                f"Generate the comprehensive, deep-dive 4-part technical response:"
            )
        else:
            user_message = (
                f"{history}\n\n"
                f'Interviewer just said:\n"{question}"\n\n'
                f"Generate the concise, scannable 4-part response:"
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
                max_tokens=max_tokens_val,
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
    mode: str = ""
    is_alt: bool = False


class ModePayload(BaseModel):
    mode: str


AskPayload.model_rebuild()
ModePayload.model_rebuild()


async def process_and_broadcast_question(
    question: str,
    source_label: str,
    mode: str = "",
    is_alt: bool = False
):
    """Processes a text/chat question and broadcasts progress + answer to phone."""
    time_str = time.strftime("%H:%M:%S")
    prefix = f"[{source_label}]" if source_label else ""
    display_q = f"{prefix} {question}".strip()
    effective_mode = mode if mode in ("normal", "detailed") else current_answering_mode

    await broadcast({
        "type": "transcription",
        "text": display_q,
        "mode": effective_mode,
    })
    loop = asyncio.get_running_loop()
    try:
        answer, model_used = await loop.run_in_executor(
            None, generate_response, question, effective_mode, is_alt
        )
        transcript_history.append({
            "question": display_q,
            "answer": answer,
            "time": time_str,
            "mode": effective_mode,
        })
        await broadcast({
            "type": "qa",
            "question": display_q,
            "answer": answer,
            "time": time_str,
            "model": model_used,
            "mode": effective_mode,
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
    return {
        "status": "ok",
        "connected_clients": len(connected_clients),
        "mode": current_answering_mode,
    }


@app.get("/mode")
async def get_mode_endpoint():
    return {"mode": current_answering_mode}


@app.post("/mode")
async def set_mode_endpoint(payload: ModePayload):
    global current_answering_mode
    if payload.mode in ("normal", "detailed"):
        current_answering_mode = payload.mode
        print(f"🔄 Mode set to: {current_answering_mode.upper()}")
        await broadcast({"type": "mode_changed", "mode": current_answering_mode})
        return {"status": "ok", "mode": current_answering_mode}
    return {"status": "error", "message": "Invalid mode. Choose 'normal' or 'detailed'"}


@app.post("/ask")
async def ask_endpoint(payload: AskPayload):
    q = payload.question.strip()
    if not q:
        return {"status": "error", "message": "Empty question"}
    source_label = "Clipboard" if payload.source == "clipboard" else payload.source
    effective_mode = payload.mode if payload.mode in ("normal", "detailed") else current_answering_mode
    asyncio.create_task(process_and_broadcast_question(
        q, source_label, mode=effective_mode, is_alt=payload.is_alt
    ))
    return {"status": "processing", "question": q, "mode": effective_mode}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_answering_mode
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"📱 Phone connected! (Total clients: {len(connected_clients)})")
    # Sync active mode immediately to phone
    await websocket.send_text(json.dumps({"type": "mode_changed", "mode": current_answering_mode}))
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
                msg_type = data.get("type")
                if msg_type == "ask":
                    q = data.get("question", "").strip()
                    source = data.get("source", "Phone Input")
                    mode = data.get("mode")
                    is_alt = bool(data.get("is_alt", False))
                    if q:
                        asyncio.create_task(process_and_broadcast_question(
                            q, source, mode=mode or "", is_alt=is_alt
                        ))
                elif msg_type == "set_mode":
                    new_mode = data.get("mode", "normal")
                    if new_mode in ("normal", "detailed"):
                        current_answering_mode = new_mode
                        print(f"🔄 Switched answering mode to: {current_answering_mode.upper()}")
                        await broadcast({"type": "mode_changed", "mode": current_answering_mode})
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

                                    active_mode = current_answering_mode
                                    # 2. Generate Answer
                                    answer, model_used = generate_response(text, mode=active_mode)
                                    print(f"✅ Generated answer via {model_used} [{active_mode.upper()} MODE]\n")

                                    time_str = time.strftime("%H:%M:%S")
                                    transcript_history.append({
                                        "question": text,
                                        "answer": answer,
                                        "time": time_str,
                                        "mode": active_mode,
                                    })

                                    asyncio.run_coroutine_threadsafe(
                                        broadcast({
                                            "type": "qa",
                                            "question": text,
                                            "answer": answer,
                                            "time": time_str,
                                            "model": model_used,
                                            "mode": active_mode,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    threading.Thread(target=audio_processing_loop, args=(loop,), daemon=True).start()
    yield

app.router.lifespan_context = lifespan


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
