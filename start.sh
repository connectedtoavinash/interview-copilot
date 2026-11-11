#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================"
echo "  🎯 Interview Copilot (Downloads Environment)"
echo "======================================================"

# 1. Setup venv if missing
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "📦 Installing dependencies (FastAPI, Groq, etc.)..."
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo "✅ Dependencies installed!"
else
    source venv/bin/activate
fi

# 2. Check .env
if [ ! -f ".env" ]; then
    cp .env.example .env
fi

# 3. Check API key
if grep -q "gsk_your_key_here" .env; then
    echo ""
    echo "⚠️  Wait! You need to put your Groq API key into .env"
    echo "   Opening .env in nano for you now..."
    echo "   (Paste your key next to GROQ_API_KEY=, press Ctrl+O, Enter, Ctrl+X)"
    echo ""
    read -p "Press Enter to open .env..."
    nano .env
fi

# 4. Detect LAN IP (works on WiFi, hotspot, ethernet)
LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$LAN_IP" ]; then
    LAN_IP=$(ip -4 route get 1.0.0.0 2>/dev/null | grep -oP 'src \K\S+')
fi
LAN_IP="${LAN_IP:-127.0.0.1}"

# 5. Launch server
echo ""
echo "🚀 Starting server..."
echo ""
echo "┌──────────────────────────────────────────┐"
echo "│  📱 Open on your PHONE:                  │"
printf "│  http://%-32s│\n" "${LAN_IP}:8765"
echo "│                                          │"
echo "│  💻 Localhost:                           │"
echo "│  http://127.0.0.1:8765                   │"
echo "└──────────────────────────────────────────┘"
echo ""
python3 server.py

