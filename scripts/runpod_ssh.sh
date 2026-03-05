#!/bin/bash
# RunPod SSH Helper
# Automatically discovers the running pod and connects via SSH.
#
# Usage:
#   bash scripts/runpod_ssh.sh              # Interactive SSH session
#   bash scripts/runpod_ssh.sh "ls /workspace/data"  # Run a command
#   bash scripts/runpod_ssh.sh --sync       # Git pull on RunPod
#   bash scripts/runpod_ssh.sh --info       # Show pod info + disk usage
#
# Requirements:
#   - runpodctl at /opt/homebrew/bin/runpodctl (or in PATH)
#   - SSH key at ~/.ssh/id_ed25519 (ed25519 key registered with RunPod)
#
# Note: The RunPod-Key-Go RSA key in ~/.runpod/ssh/ does NOT work.
# The registered key is the ed25519 key from ~/.ssh/id_ed25519.

set -euo pipefail

# --- Configuration ---
RUNPODCTL="${RUNPODCTL:-/opt/homebrew/bin/runpodctl}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ServerAliveInterval=30"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- Functions ---
get_pod_info() {
    local pod_line
    pod_line=$("$RUNPODCTL" get pod 2>/dev/null | grep -i "RUNNING" | head -1)
    if [ -z "$pod_line" ]; then
        echo -e "${RED}No running pod found.${NC}" >&2
        echo -e "Start a pod first at https://www.runpod.io/console/pods" >&2
        exit 1
    fi
    echo "$pod_line"
}

get_pod_id() {
    get_pod_info | awk '{print $1}'
}

get_ssh_connection() {
    local pod_id="$1"
    local pod_detail
    pod_detail=$("$RUNPODCTL" get pod "$pod_id" -a 2>/dev/null)

    # Extract SSH port: look for pattern like IP:PORT->22 (pub,tcp)
    local ssh_info
    ssh_info=$(echo "$pod_detail" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+->22' | head -1)

    if [ -z "$ssh_info" ]; then
        echo -e "${RED}Could not find SSH port for pod $pod_id${NC}" >&2
        echo -e "Pod details:" >&2
        echo "$pod_detail" >&2
        exit 1
    fi

    local ip port
    ip=$(echo "$ssh_info" | cut -d: -f1)
    port=$(echo "$ssh_info" | cut -d: -f2 | cut -d- -f1)

    echo "$ip $port"
}

run_ssh() {
    local ip="$1"
    local port="$2"
    shift 2

    if [ $# -eq 0 ]; then
        # Interactive session
        ssh -i "$SSH_KEY" -p "$port" $SSH_OPTS "root@$ip"
    else
        # Run command
        ssh -i "$SSH_KEY" -p "$port" $SSH_OPTS "root@$ip" "$@"
    fi
}

# --- Main ---
# Check runpodctl exists
if [ ! -x "$RUNPODCTL" ]; then
    # Try PATH
    if command -v runpodctl &>/dev/null; then
        RUNPODCTL="runpodctl"
    else
        echo -e "${RED}runpodctl not found at $RUNPODCTL${NC}" >&2
        echo "Install: brew install runpod/runpodctl/runpodctl" >&2
        exit 1
    fi
fi

# Discover pod
echo -e "${CYAN}Discovering running pod...${NC}" >&2
POD_ID=$(get_pod_id)
read -r IP PORT <<< "$(get_ssh_connection "$POD_ID")"
echo -e "${GREEN}Pod: $POD_ID → $IP:$PORT${NC}" >&2

# Handle special flags
case "${1:-}" in
    --sync)
        echo -e "${YELLOW}Syncing eeg-state repo on RunPod...${NC}" >&2
        run_ssh "$IP" "$PORT" "cd /workspace/eeg-state && git stash && git pull origin main && git stash pop 2>/dev/null; git log --oneline -3"
        ;;
    --info)
        echo -e "${YELLOW}Pod info:${NC}" >&2
        "$RUNPODCTL" get pod "$POD_ID" -a
        echo ""
        echo -e "${YELLOW}Disk usage:${NC}" >&2
        run_ssh "$IP" "$PORT" "echo '=== /workspace ==='; du -sh /workspace/*/; echo ''; echo '=== /workspace/data ==='; du -sh /workspace/data/*/; echo ''; echo '=== GPU ==='; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo 'No GPU info'"
        ;;
    --port)
        # Just print connection details (for scripting)
        echo "$IP $PORT"
        ;;
    *)
        run_ssh "$IP" "$PORT" "$@"
        ;;
esac
