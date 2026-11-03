---
name: mediapipe-sandbox
description: Utiliser MediaPipe (Google) dans le sandbox Claude.ai - détection de mains, pose du corps, maillage du visage, détection de visage, segmentation de silhouette (selfie), plus l'API Tasks (objets, gestes, classification, segmentation) avec un modèle fourni par l'utilisateur. À utiliser dès que l'utilisateur mentionne MediaPipe, landmarks, squelette/pose, suivi des mains, face mesh, segmentation de personne, détection d'objets sur image ou vidéo, ou veut exploiter le dépôt Metba09/mediapipe (fork) depuis une session Claude.
---

# MediaPipe dans le sandbox Claude.ai

Le dépôt `Metba09/mediapipe` est un fork du code source MediaPipe (C++/Bazel, ~88 Mo). Il ne se **compile pas** dans le sandbox : ce skill utilise les paquets pip officiels et traite le dépôt comme une **bibliothèque de référence** (docs, API Python, graphes, model maker).

## Contraintes du sandbox (vérifiées)

| Fait | Conséquence |
|---|---|
| `pip` fonctionne | installer `mediapipe` depuis PyPI, ne pas compiler le dépôt |
| `storage.googleapis.com` est **bloqué** | impossible de télécharger les modèles `.task` officiels |
| `mediapipe==0.10.18` (cp312) **embarque** les modèles (mains, pose, visage, iris, segmentation selfie) | voie hors-ligne : `mp.solutions` |
| `mediapipe` ≥ 1.0 (API Tasks) n'embarque aucun modèle et n'a plus `mp.solutions` | modèle fourni par l'utilisateur obligatoire |
| Pas de GPU, pas de webcam | traiter des fichiers image/vidéo uniquement |

## Choisir la voie

1. **Mains, pose, visage, segmentation de personne** → voie « legacy » (hors-ligne, rien à demander à l'utilisateur).
2. **Objets, gestes, classification, embeddings, segmentation générique** → voie « tasks » : demander à l'utilisateur de joindre le modèle (`.task` ou `.tflite`, ex. EfficientDet, gesture_recognizer.task). Sans modèle, le dire clairement et ne pas inventer de résultat.

## Mise en route

```bash
bash .claude/skills/mediapipe-sandbox/scripts/setup_env.sh legacy   # ~1 min, crée /tmp/mp-legacy (Python 3.12)
bash .claude/skills/mediapipe-sandbox/scripts/setup_env.sh tasks    # crée /tmp/mp-tasks (dernier mediapipe)
```
Depuis la racine du fork, `make setup` fait la même chose ; `make test` lance le test de fumée et `make check` (syntaxe, secrets, test) se lance avant de conclure. Le script est idempotent et sans saisie. Si le skill est installé ailleurs, adapter le chemin (`/mnt/skills/...`). Les environnements sont perdus à la fin de la session : relancer le script.

## Utilisation

Suivi des TEXTES animés (motion typographique) : expressions, trajectoires, styles en JSON (OCR hors-ligne : rapidocr + tesseract) :
```bash
make setup-ocr                                   # crée /tmp/mp-ocr (≈ 1 min)
make text VIDEO=video.mp4 [SHOTS=/tmp/mp-out/track/tracks.json] [REVIEW=revue.json]
```
Sortie `text_tracking.json` (schéma `text-motion-tracking/1`) : `expressions` (lignes regroupées, ordre de lecture, alignement), `elements` (texte, variantes OCR, qualité fiable/moyen/fragment, niveau principal/secondaire/overlay, `trajectory.per_frame` [image, t, cx, cy, w, h, angle, conf, opacité estimée, couleur], `keyframes` simplifiées, `style` mesuré : couleur + candidats, fond, taille de police estimée, graisse, italique, casse, halo, contraste des traits), plus `text_preview.mp4`. Les polices ne sont PAS identifiées : `font_class_guess` est une heuristique faible (elle classe mal certains serifs) ; pour une classe de police fiable, regarder l'image en gros plan et l'ajouter via `--review` (champ `visual_review`, séparé des mesures). Vérifier toujours les textes principaux à l'œil : l'OCR se trompe quand un objet masque une lettre (ex. « VOGUE » lu « VOUE ») et les petits textes restent approximatifs.

Suivi de mouvement d'une vidéo (motion design, plans caméra : aucun modèle, aucun réseau) :
```bash
make track VIDEO=video.mp4            # ou : /tmp/mp-legacy/bin/python scripts/track_motion.py video.mp4 --out /tmp/mp-out/track
```
Sorties : `tracks.json` (plans, mouvement dominant par image : dx, dy, échelle, rotation, cumul par plan ; pistes de points), `motion_curve.csv`, `shots.csv`, `annotated.mp4` (traînées), `motion_timeline.png`. Les points immobiles (filigrane, fond fixe) sont marqués `static`. Les coupes sont détectées par histogramme **et** par perte brutale de pistes (transitions sur fond blanc) ; régler `--cut-threshold` et `--lost-cut` si un plan est manqué ou coupé à tort. Pour une vidéo de motion design (affiches, typographies), c'est cette voie qui s'applique, pas la pose ni les mains. L'énergie est moyennée sur tout l'écran : un petit élément qui bouge pèse peu.

Voie legacy (image ou vidéo ; `--stride N` pour sous-échantillonner, `--max-frames` borne les vidéos) :
```bash
/tmp/mp-legacy/bin/python scripts/legacy_landmarks.py hands photo.jpg --out /tmp/mp-out/sujet
/tmp/mp-legacy/bin/python scripts/legacy_landmarks.py pose video.mp4 --out /tmp/mp-out/sujet --stride 2
```
Solutions : `hands | pose | face_mesh | face_detection | holistic | selfie_segmentation`.
Sorties : `annotated.png|mp4`, `landmarks.json` (coordonnées normalisées 0-1 ; `pose_world` en mètres), et `mask.png` pour la segmentation.

Voie tasks :
```bash
/tmp/mp-tasks/bin/python scripts/tasks_run.py object --model efficientdet.tflite --image photo.jpg --out /tmp/mp-out/sujet
```
Tâches : `pose | hands | face | face_detect | gesture | object | classify | segment`.

## Règles de travail

- **Dépôt des fichiers générés (règle de l'utilisateur)** : les livrables vont dans le dépôt `Metba09/claude-memory`, dossier `livrables/mediapipe/AAAA-MM-JJ-sujet/` ; le fork ne contient que le code et le skill. Travailler dans `/tmp/mp-out` (défaut des scripts), puis copier, commiter et pousser dans `claude-memory` avant la fin de la session (le sandbox est éphémère). Ne pas y mettre d'images de personnes sans l'accord de l'utilisateur ; vérifier qu'aucun secret n'y figure.
- Livrer aussi le fichier annoté (image/mp4) avec `SendUserFile` et résumer les résultats en une ou deux phrases (nombre de mains/personnes, scores).
- Un résultat vide (`None`, 0 détection) est normal si le sujet est absent, trop petit ou flou : le signaler plutôt que d'insister ; proposer une autre image.
- Les landmarks sont normalisés : multiplier x par la largeur et y par la hauteur pour obtenir des pixels.
- Vidéo : traiter les images dans l'ordre (le suivi temporel dépend de la séquence) et borner la durée (`--max-frames`).
- Pose : garder `model_complexity=1` (0 et 2 exigent un téléchargement bloqué dans le sandbox).
- Ne jamais présenter la voie legacy comme l'API actuelle : 0.10.18 est figée, utile ici parce que les modèles y sont inclus.
- Visage/personne : traiter uniquement les images que l'utilisateur fournit ; pas d'identification de personnes.

## Limites connues (honnêteté sur les tests)

- Les scripts ont été exécutés avec succès sur une image et une vidéo **synthétiques** (aucune détection attendue) : le chargement des modèles et la chaîne de sortie fonctionnent, mais la qualité de détection n'a pas été vérifiée sur de vraies photos.
- `tasks_run.py` n'a pas été testé avec un vrai modèle (aucun n'est téléchargeable ici) ; seuls l'import de l'API et la gestion d'erreur sont vérifiés.

## Références (charger à la demande)

- `references/legacy-solutions.md` : options et sorties de `mp.solutions`, indices des points clés.
- `references/tasks-api.md` : classes, options et résultats de l'API Tasks, relevés dans les sources du dépôt.
- `references/repo-map.md` : carte du dépôt (docs, tasks, model_maker, graphes) et où chercher quoi.
