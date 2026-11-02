# Commandes du sandbox Claude.ai (voir CLAUDE.md, section « Session Chat »).
# Toute la logique est dans .claude/skills/mediapipe-sandbox/scripts/ : ce Makefile n'est qu'un point d'entrée.
SK := .claude/skills/mediapipe-sandbox
MP_VENV_BASE ?= /tmp/mp
export MP_VENV_BASE

.PHONY: setup setup-tasks sample test check track clean help
help:
	@echo "make setup        installe l'environnement hors-ligne (idempotent)"
	@echo "make setup-tasks  installe aussi l'API Tasks (modèles fournis par l'utilisateur)"
	@echo "make sample       génère des données d'exemple dans /tmp/mp-sample"
	@echo "make track VIDEO=f.mp4  suivi de mouvement -> /tmp/mp-out/track (OUT=dossier pour changer)"
	@echo "make test         test de fumée des solutions embarquées"
	@echo "make check        syntaxe + frontmatter + secrets + test (à lancer avant de conclure)"
setup:
	bash $(SK)/scripts/setup_env.sh legacy
setup-tasks:
	bash $(SK)/scripts/setup_env.sh tasks
sample:
	python3 $(SK)/scripts/make_sample_data.py /tmp/mp-sample
track: setup
	@test -n "$(VIDEO)" || { echo "usage : make track VIDEO=fichier.mp4 [OUT=dossier]"; exit 2; }
	$(MP_VENV_BASE)-legacy/bin/python $(SK)/scripts/track_motion.py "$(VIDEO)" --out $(or $(OUT),/tmp/mp-out/track)
test: setup
	$(MP_VENV_BASE)-legacy/bin/python $(SK)/tests/test_smoke.py 2>&1 | grep -v "^W0000\|^I0000\|INFO\|WARNING"
check: setup
	bash $(SK)/scripts/check.sh
clean:
	rm -rf $(MP_VENV_BASE)-legacy $(MP_VENV_BASE)-tasks /tmp/mp-sample
