# Commandes du sandbox Claude.ai (voir CLAUDE.md, section « Session Chat »).
# Toute la logique est dans .claude/skills/mediapipe-sandbox/scripts/ : ce Makefile n'est qu'un point d'entrée.
SK := .claude/skills/mediapipe-sandbox
MP_VENV_BASE ?= /tmp/mp
export MP_VENV_BASE

.PHONY: gsap setup setup-tasks setup-ocr sample test check track text clean help
help:
	@echo "make setup        installe l'environnement hors-ligne (idempotent)"
	@echo "make setup-tasks  installe aussi l'API Tasks (modèles fournis par l'utilisateur)"
	@echo "make sample       génère des données d'exemple dans /tmp/mp-sample"
	@echo "make track VIDEO=f.mp4  suivi de mouvement -> /tmp/mp-out/track (OUT=dossier pour changer)"
	@echo "make setup-ocr    installe l'OCR hors-ligne (/tmp/mp-ocr)"
	@echo "make text VIDEO=f.mp4 [SHOTS=tracks.json] [REVIEW=revue.json]  textes, trajectoires, styles -> /tmp/mp-out/text"
	@echo "make test         test de fumée des solutions embarquées"
	@echo "make check        syntaxe + frontmatter + secrets + test (à lancer avant de conclure)"
setup:
	bash $(SK)/scripts/setup_env.sh legacy
setup-tasks:
	bash $(SK)/scripts/setup_env.sh tasks
sample:
	python3 $(SK)/scripts/make_sample_data.py /tmp/mp-sample
setup-ocr:
	bash $(SK)/scripts/setup_env.sh ocr
text: setup-ocr
	@test -n "$(VIDEO)" || { echo "usage : make text VIDEO=fichier.mp4 [SHOTS=tracks.json] [REVIEW=revue.json] [OUT=dossier]"; exit 2; }
	$(MP_VENV_BASE)-ocr/bin/python $(SK)/scripts/text_track.py "$(VIDEO)" --out $(or $(OUT),/tmp/mp-out/text) $(if $(SHOTS),--shots $(SHOTS)) $(if $(REVIEW),--review $(REVIEW))
gsap:
	@test -n "$(CORPUS)" && test -n "$(TRACKING)" || { echo "usage : make gsap CORPUS=texte.md TRACKING=text_tracking.json [OUT=page.html]  (npm i gsap @fontsource/arimo @fontsource/playfair-display @fontsource/tinos dans /tmp/mp-work/gsap-assets)"; exit 2; }
	python3 $(SK)/scripts/extract_motion_presets.py "$(TRACKING)" --out /tmp/mp-out/gsap/motion_presets.json
	python3 $(SK)/scripts/build_gsap_corpus.py "$(CORPUS)" --presets /tmp/mp-out/gsap/motion_presets.json --out $(or $(OUT),/tmp/mp-out/gsap/page.html)
track: setup
	@test -n "$(VIDEO)" || { echo "usage : make track VIDEO=fichier.mp4 [OUT=dossier]"; exit 2; }
	$(MP_VENV_BASE)-legacy/bin/python $(SK)/scripts/track_motion.py "$(VIDEO)" --out $(or $(OUT),/tmp/mp-out/track)
test: setup
	$(MP_VENV_BASE)-legacy/bin/python $(SK)/tests/test_smoke.py 2>&1 | grep -v "^W0000\|^I0000\|INFO\|WARNING"
check: setup
	bash $(SK)/scripts/check.sh
clean:
	rm -rf $(MP_VENV_BASE)-legacy $(MP_VENV_BASE)-tasks $(MP_VENV_BASE)-ocr /tmp/mp-sample
