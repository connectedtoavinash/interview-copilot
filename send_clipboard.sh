#!/usr/bin/env bash
# ==============================================================================
# Interview Copilot — SILENT clipboard sender (no visible notifications)
# ==============================================================================

PORT="8765"
SERVER_URL="http://127.0.0.1:${PORT}/ask"
TEXT=""

# 1. Read clipboard silently
if command -v wl-paste >/dev/null 2>&1; then
    TEXT="$(wl-paste 2>/dev/null)"
    [ -z "$TEXT" ] && TEXT="$(wl-paste -p 2>/dev/null)"
fi

if [ -z "$TEXT" ] && command -v xclip >/dev/null 2>&1; then
    TEXT="$(xclip -selection clipboard -o 2>/dev/null)"
    [ -z "$TEXT" ] && TEXT="$(xclip -o 2>/dev/null)"
fi

if [ -z "$TEXT" ]; then
    TEXT=$(python3 -c "
try:
    import gi
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk, Gdk
    cb = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
    t = cb.wait_for_text()
    if t: print(t)
except Exception:
    pass
" 2>/dev/null)
fi

TEXT="$(echo "$TEXT" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
[ -z "$TEXT" ] && exit 1

# 2. Build JSON and send silently
JSON_PAYLOAD=$(python3 -c "import json, sys; print(json.dumps({'question': sys.argv[1], 'source': 'clipboard'}))" "$TEXT")

curl -s -m 5 -X POST "$SERVER_URL" \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD" >/dev/null 2>&1 &

exit 0
