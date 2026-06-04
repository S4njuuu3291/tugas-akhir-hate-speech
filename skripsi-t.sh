#!/bin/bash

# Tmux session setup untuk HATE-SPEECH project
# Membuat 1 window dengan 2 pane vertikal

SESSION_NAME="ta"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kill existing session jika ada
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Buat session baru
tmux new-session -d -s $SESSION_NAME -c "$PROJECT_DIR"

# Split window secara vertikal (kanan-kiri)
tmux split-window -h -t $SESSION_NAME -c "$PROJECT_DIR"

# Konfigurasi pane kiri (index 0)
tmux select-pane -t $SESSION_NAME:0
tmux send-keys -t $SESSION_NAME:0 "# Pane kiri - Ready" Enter

# Konfigurasi pane kanan (index 1)
tmux select-pane -t $SESSION_NAME:1
tmux send-keys -t $SESSION_NAME:1 "# Pane kanan - Ready" Enter

# Select pane kiri sebagai pane aktif
tmux select-pane -t $SESSION_NAME:0

# Attach ke session
tmux attach-session -t $SESSION_NAME
