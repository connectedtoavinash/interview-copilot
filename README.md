# 🎯 Interview Copilot

An undetectable, real-time technical interview assistant that captures interviewer audio via PipeWire loopback or clipboard shortcuts, transcribes questions with Groq Whisper, and delivers concise, phone-scannable answers grounded in your candidate resume directly to your mobile browser.

---

## ⚡ Features

- **Invisible & Screen-Share Safe**: Zero UI on your desktop, no popups, no system notifications (`send_clipboard.sh` is 100% silent).
- **Mobile-First UI**: View answers in dark mode on your phone screen with auto-scroll and wake lock.
- **Dual Input Modes**:
  - **Audio Loopback**: Automatically captures interviewer's voice via Linux PipeWire (`pw-record`) and transcribes using Groq Whisper.
  - **Clipboard Shortcut (`Ctrl+Alt+C`)**: Instantly send copied questions or code snippets directly to the server.
  - **Phone Paste Modal**: Manually paste text or code directly from your phone.
- **Dual Answering Modes (Dynamic Toggle)**:
  - **⚡ Normal Mode (Concise)**: Built for rapid live scanning (<250 words) — confident opening thesis, 3-5 bold keyword bullets, compact ASCII diagram/code snippet, gotcha.
  - **🧠 Detailed Mode (Deep Dive)**: Masterclass architectural answer (~1600 tokens) — Executive Summary, JVM/kernel mechanics & internal architecture, production code/distributed diagrams, trade-offs, failure modes & mitigations.
  - **Instant Switching**: Switch anytime via the pinned top bar toggle `[⚡ Normal | 🧠 Detailed]`, or tap `🧠 Deep Dive` / `⚡ Make Concise` on any card to re-run that specific question!
- **Interactive Phone Controls**:
  - **✏️ Edit Question**: Modify any transcription or pasted question right on your phone and re-solve instantly.
  - **🔄 Alt Solution / Rewrite**: One-tap trigger on the question card or right above the code/diagram to rewrite with an alternative implementation (e.g. Subquery vs Window Function, Iterative vs Recursive, different SQL dialect or design pattern).
- **Engineered for Live Answers**:
  - Structured into **SAY THIS FIRST**, **KEY POINTS**, **CODE / ASCII DIAGRAM**, and **GOTCHAS**.
  - Smart triggers: generates compact ASCII diagrams for architecture/lifecycle questions, concise code snippets for implementation questions, or skips straight to bullets for quick definitions.
  - Grounded in your actual resume and target job description.

---

## 🚀 Quick Setup

### 1. Prerequisites
- Linux (Ubuntu / Debian with PipeWire)
- Python 3.10+
- `wl-clipboard` and `xclip` (for clipboard shortcut):
  ```bash
  sudo apt install -y wl-clipboard xclip
  ```

### 2. Configuration
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Put your Groq API key in `.env`:
   ```bash
   GROQ_API_KEY=gsk_your_groq_api_key
   GROQ_MODEL=qwen/qwen3.8-27b
   ```
3. Update `resume.txt` and `job_description.txt` with your details.

---

## 🎮 Running the Server

Start the copilot:
```bash
cd ~/Downloads/interview-copilot && bash start.sh
```

`start.sh` will automatically detect your network IP (Wi-Fi, Mobile Hotspot, or Ethernet) and display your phone URL:
```
┌──────────────────────────────────────────┐
│  📱 Open on your PHONE:                  │
│  http://192.168.1.X:8765                 │
│                                          │
│  💻 Localhost:                            │
│  http://127.0.0.1:8765                   │
└──────────────────────────────────────────┘
```

---

## ⌨️ Silent Clipboard Shortcut (Ubuntu)

Set up a global keybinding so you can highlight a question during a screen-share and send it silently to your phone:

1. Open **Ubuntu Settings** → **Keyboard** → **Keyboard Shortcuts** → **Custom Shortcuts**.
2. Click **+** (Add Shortcut):
   - **Name**: `Send to Copilot`
   - **Command**: `/bin/bash /home/avinash/Downloads/interview-copilot/send_clipboard.sh`
   - **Shortcut**: `Ctrl + Alt + C`
3. Now whenever you copy any text and press `Ctrl + Alt + C`, the answer will silently show up on your phone!

---

## 🛠️ Management & Debug Commands

```bash
# Start server
cd ~/Downloads/interview-copilot && bash start.sh

# Stop server
pkill -f "python3 server.py"

# Restart
pkill -f "python3 server.py"; sleep 1; bash ~/Downloads/interview-copilot/start.sh

# Check if server is running
ss -tlnp | grep 8765

# Test health check
curl http://127.0.0.1:8765/health

# Manually test clipboard send
echo "Explain Bean Lifecycle in Spring Boot" | wl-copy
bash ~/Downloads/interview-copilot/send_clipboard.sh
```

