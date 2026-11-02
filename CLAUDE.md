# Fork MediaPipe (Metba09/mediapipe)

Ce dépôt est le code source de MediaPipe (C++/Bazel). Il ne se compile pas dans le sandbox Claude.ai.
Pour exécuter MediaPipe (mains, pose, visage, segmentation, objets), utiliser le skill
`.claude/skills/mediapipe-sandbox/` ; utiliser le reste du dépôt comme documentation et référence d'API.

## Session Chat (Claude.ai, sandbox éphémère)

Le sandbox repart de zéro à chaque session. Toute la logique est dans des scripts ordinaires ; rien ne dépend d'un hook, d'une slash command ou d'un MCP.

**Installer et tester (aucune saisie, relançable sans risque)**
```bash
make setup     # environnement hors-ligne : /tmp/mp-legacy (Python 3.12 + mediapipe 0.10.18, modèles embarqués)
make test      # test de fumée des 6 solutions sur données synthétiques locales
make track VIDEO=f.mp4   # suivi de mouvement d'une vidéo -> /tmp/mp-out/track
make check     # syntaxe + frontmatter + secrets + test : à lancer avant de conclure
make setup-tasks   # optionnel : API Tasks récente (exige un modèle .task/.tflite fourni par l'utilisateur)
```
Équivalent sans make : `bash .claude/skills/mediapipe-sandbox/scripts/setup_env.sh legacy`.
Variables : voir `.env.example` (aucun secret requis).

**Inutilisable dans le sandbox**
- Compilation du dépôt (Bazel, `setup.py`, `build_*_examples.sh`), Docker, GPU, webcam, SDK Android/iOS.
- Réseau hors GitHub et registres de paquets : les modèles officiels (`storage.googleapis.com`) sont inaccessibles ; `Pose(model_complexity=0 ou 2)` échoue pour cette raison (utiliser 1).
- Services externes et secrets : aucun n'est nécessaire ; les tests utilisent des données d'exemple générées localement (`make sample`).

**À lire à la demande**
- `.claude/skills/mediapipe-sandbox/SKILL.md` : usage, choix de la voie, règles de travail.
- `.claude/skills/mediapipe-sandbox/references/legacy-solutions.md`, `tasks-api.md`, `repo-map.md`.
- Il n'y a aucune commande ni agent `.claude/commands` ou `.claude/agents` à charger.

**Livrables** : les fichiers générés sont déposés dans `Metba09/claude-memory` (dossier `livrables/mediapipe/AAAA-MM-JJ-sujet/`), commités et poussés avant la fin de la session. Ce fork ne contient que le code et le skill ; les scripts écrivent par défaut dans `/tmp/mp-out`.
