#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_DIR="$ROOT_DIR/rrt_project"
VENV="$ROOT_DIR/.venv_rrt_clean"
LOCK_FILE="$PROJECT_DIR/results/checkpoints/night_run.lock"
LOG_DIR="$PROJECT_DIR/results/reports"
LOG_FILE="$LOG_DIR/night_run.log"
LIMIT_PER_MIN="40"
RESUME="--resume"
MODE="foreground"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)
      RESUME=""
      shift
      ;;
    --limit)
      LIMIT_PER_MIN="$2"
      shift 2
      ;;
    --daemon)
      MODE="daemon"
      shift
      ;;
    --tmux)
      MODE="tmux"
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR" "$PROJECT_DIR/results/checkpoints"

if [[ "$MODE" == "daemon" ]]; then
  nohup bash "$0" $RESUME --limit "$LIMIT_PER_MIN" > "$LOG_FILE" 2>&1 &
  echo "Started daemon PID=$! log=$LOG_FILE"
  exit 0
fi

if [[ "$MODE" == "tmux" ]]; then
  SESSION="rrt_night_run"
  tmux new-session -d -s "$SESSION" "bash '$0' $RESUME --limit '$LIMIT_PER_MIN'"
  echo "Started tmux session=$SESSION"
  exit 0
fi

exec 200>"$LOCK_FILE"
flock -n 200 || { echo "Another run is active."; exit 1; }

cd "$ROOT_DIR"
source "$VENV/bin/activate"

python -m rrt_project.benchmark.runner --pipeline full $RESUME --limit-per-minute "$LIMIT_PER_MIN"

echo "Night run finished at $(date -Is)" | tee -a "$LOG_FILE"
