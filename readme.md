🎯 Interview Copilot — Command Cheat Sheet

Start/Stop:

bash
# Start server
cd ~/Downloads/interview-copilot && bash start.sh
# Kill server
pkill -f "python3 server.py"
# Restart
pkill -f "python3 server.py"; sleep 1; bash ~/Downloads/interview-copilot/start.sh

Send questions:

bash
# Copy text then send clipboard (or use Ctrl+Alt+C shortcut)
echo "What is JVM?" | wl-copy
bash ~/Downloads/interview-copilot/send_clipboard.sh

Debug:

bash
# Check if server is running
ss -tlnp | grep 8765
# Health check
curl http://127.0.0.1:8765/health
# Find your phone URL IP
hostname -I | awk '{print $1}'
# Kill stuck port
fuser -k 8765/tcp

Phone:

Open in phone browser → http://<YOUR_IP>:8765
(start.sh prints the exact URL for you)