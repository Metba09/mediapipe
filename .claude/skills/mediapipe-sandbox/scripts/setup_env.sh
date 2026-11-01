#!/usr/bin/env bash
# Crée les environnements MediaPipe du sandbox Claude.ai (pip fonctionne, le CDN de modèles est bloqué).
#   setup_env.sh legacy  -> mediapipe 0.10.18 + Python 3.12 : mp.solutions + modèles EMBARQUÉS (hors-ligne)
#   setup_env.sh tasks   -> dernier mediapipe (API Tasks) : nécessite des modèles .task/.tflite fournis par l'utilisateur
#   setup_env.sh both    -> les deux
set -euo pipefail
BASE="${MP_VENV_BASE:-/tmp/mp}"
mode="${1:-legacy}"
mk() { # nom python paquets...
  local name="$1" py="$2"; shift 2
  if [ -x "$BASE-$name/bin/python" ]; then echo "[ok] $BASE-$name déjà prêt"; return; fi
  "$py" -m venv "$BASE-$name"
  "$BASE-$name/bin/pip" install -q "$@"
  echo "[ok] $BASE-$name prêt"
}
case "$mode" in
  legacy|both) mk legacy python3.12 "mediapipe==0.10.18" opencv-python-headless "numpy<2" ;;
esac
case "$mode" in
  tasks|both)  mk tasks python3 mediapipe opencv-python-headless ;;
esac
[ "$mode" = legacy ] || [ "$mode" = tasks ] || [ "$mode" = both ] || { echo "usage: $0 legacy|tasks|both"; exit 2; }
