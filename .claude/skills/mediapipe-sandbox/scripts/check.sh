#!/usr/bin/env bash
# Vérifications explicites à lancer à la main avant de conclure (remplace les hooks) : syntaxe + test de fumée.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="${MP_VENV_BASE:-/tmp/mp}"
echo "== syntaxe"
for f in scripts/*.py tests/*.py; do python3 -m py_compile "$f"; done
bash -n scripts/setup_env.sh; bash -n scripts/check.sh
find . -name __pycache__ -prune -exec rm -rf {} +
echo "== frontmatter SKILL.md"
python3 - <<'PY'
import re
s = open("SKILL.md").read()
m = re.match(r"---\nname: ([a-z0-9-]+)\ndescription: (.+)\n---", s)
assert m and len(m.group(2)) < 1024, "frontmatter invalide"
print("ok :", m.group(1))
PY
echo "== secrets"
if grep -rEn --exclude=check.sh "ghp_|github_pat_|AKIA[0-9A-Z]{16}|BEGIN (RSA |EC )?PRIVATE KEY" . ; then echo "secret détecté" >&2; exit 1; else echo "aucun"; fi
echo "== test de fumée (voie legacy)"
[ -f "$BASE-legacy/.ready" ] || { echo "environnement absent : lancer 'make setup' d'abord" >&2; exit 1; }
"$BASE-legacy/bin/python" tests/test_smoke.py 2>&1 | grep -v "^W0000\|^I0000\|INFO\|WARNING"
echo "== test du suivi de texte (voie OCR, facultatif)"
if [ -f "${MP_VENV_BASE:-/tmp/mp}-ocr/.ready" ]; then
  "${MP_VENV_BASE:-/tmp/mp}-ocr/bin/python" tests/test_text.py 2>&1 | grep -v "^W0000\|^I0000\|INFO\|WARNING"
else
  echo "ignoré : environnement OCR absent (make setup-ocr)"
fi
