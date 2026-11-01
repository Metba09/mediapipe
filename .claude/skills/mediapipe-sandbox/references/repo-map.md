# Carte du dépôt Metba09/mediapipe (fork, commit 68392c4, 2026-10-30)

| Chemin | Contenu | Utilité dans le sandbox |
|---|---|---|
| `mediapipe/tasks/python/` | API Python Tasks (vision, text, audio, core, components) | lire les signatures exactes avant d'écrire du code |
| `mediapipe/python/` | bindings du framework (`solution_base.py`, `packet_creator.py`...) et anciennes solutions | référence ; exécution via pip, pas via ce code |
| `mediapipe/model_maker/python/` | entraînement : gesture_recognizer, image_classifier, object_detector, text_classifier, llm | à proposer pour des modèles personnalisés (TensorFlow requis) |
| `mediapipe/graphs/` | graphes (.pbtxt) : hand_tracking, pose_tracking, face_mesh, holistic, selfie_segmentation, iris, objectron, tracking... | comprendre le pipeline d'une solution |
| `mediapipe/calculators/`, `framework/`, `gpu/` | noyau C++ (Bazel) | lecture seule : non compilable ici |
| `mediapipe/tasks/{cc,c,java,ios,web}` | autres liaisons (C++, C, Android, iOS, JavaScript) | exemples multiplateformes |
| `mediapipe/examples/` | exemples android, ios, desktop, coral | référence |
| `docs/solutions/` | docs détaillées des solutions (mains, pose, face_mesh, holistic, iris, selfie...) + `models.md` | fiches modèle, formats de sortie |
| `docs/getting_started/` | installation, Python, Android, iOS, JavaScript, FAQ | procédures officielles |
| `docs/framework_concepts/` | graphes, calculators, packets, synchronisation | concepts du framework |
| `mediapipe/tasks/testdata`, `**/testdata` | petits modèles et données de test (14 fichiers .tflite, aucun modèle de production) | tests uniquement |

Compiler le dépôt (Bazel, `setup.py`, `build_*_examples.sh`) demande Bazel, des SDK et beaucoup de temps : hors périmètre du sandbox. Pour modifier le code C++, passer par CI ou une machine locale.

Recherche rapide : `grep -rn "NomDeClasse" mediapipe/tasks/python` ; `ls docs/solutions`.
