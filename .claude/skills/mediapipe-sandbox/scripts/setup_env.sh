#!/usr/bin/env bash
# Installation idempotente et non interactive des environnements MediaPipe (sandbox Claude.ai).
# Relançable à volonté : un environnement déjà complet est ignoré, un environnement à moitié installé est refait.
#   setup_env.sh legacy  -> mediapipe 0.10.18 + Python 3.12 : mp.solutions + modèles EMBARQUÉS (hors-ligne)
#   setup_env.sh tasks   -> dernier mediapipe (API Tasks) : nécessite des modèles fournis par l'utilisateur
#   setup_env.sh both    -> les deux
# Variable : MP_VENV_BASE (défaut /tmp/mp) -> /tmp/mp-legacy, /tmp/mp-tasks
set -euo pipefail
BASE="${MP_VENV_BASE:-/tmp/mp}"
mode="${1:-legacy}"
case "$mode" in legacy|tasks|both) ;; *) echo "usage: $0 legacy|tasks|both" >&2; exit 2 ;; esac
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1

mk() { # nom python paquets...
  local name="$1" py="$2"; shift 2
  local dir="$BASE-$name"
  if [ -f "$dir/.ready" ]; then echo "[ok] $dir déjà prêt"; return; fi
  command -v "$py" >/dev/null || { echo "[erreur] $py introuvable" >&2; exit 1; }
  rm -rf "$dir"
  "$py" -m venv "$dir"
  "$dir/bin/pip" install -q "$@"
  touch "$dir/.ready"
  echo "[ok] $dir prêt"
}

case "$mode" in legacy|both) mk legacy python3.12 "mediapipe==0.10.18" opencv-python-headless "numpy<2" ;; esac
case "$mode" in tasks|both)  mk tasks python3 mediapipe opencv-python-headless ;; esac
