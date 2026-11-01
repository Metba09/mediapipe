# mp.solutions (mediapipe==0.10.18) — aide-mémoire

Modèles embarqués (dossier `mediapipe/modules/` du paquet) : face_detection (courte portée, portée complète), face_landmark (+ attention), iris_landmark, palm_detection + hand_landmark (lite/full), pose_detection + pose_landmark_full, selfie_segmentation (carré, paysage).

| Solution | Constructeur (options utiles) | Sortie de `.process(rgb)` |
|---|---|---|
| `solutions.hands.Hands` | `static_image_mode, max_num_hands=2, model_complexity (0/1), min_detection_confidence, min_tracking_confidence` | `multi_hand_landmarks` (21 pts), `multi_handedness`, `multi_hand_world_landmarks` |
| `solutions.pose.Pose` | `static_image_mode, model_complexity (0/1/2), smooth_landmarks, enable_segmentation, min_detection_confidence` | `pose_landmarks` (33 pts), `pose_world_landmarks`, `segmentation_mask` |
| `solutions.face_mesh.FaceMesh` | `static_image_mode, max_num_faces, refine_landmarks (iris, 478 pts), min_detection_confidence` | `multi_face_landmarks` (468 ou 478 pts) |
| `solutions.face_detection.FaceDetection` | `model_selection (0 proche, 1 lointain), min_detection_confidence` | `detections` (bbox relative + 6 points clés) |
| `solutions.holistic.Holistic` | `static_image_mode, model_complexity, refine_face_landmarks` | `pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks` |
| `solutions.selfie_segmentation.SelfieSegmentation` | `model_selection (0 carré, 1 paysage)` | `segmentation_mask` (float HxW, 0-1) |

**Piège hors-ligne (vérifié)** : `Pose(model_complexity=0 ou 2)` tente de télécharger le modèle lite/heavy et échoue (403) ; seul `model_complexity=1` (modèle « full » embarqué) fonctionne. Les mains fonctionnent en complexité 0 et 1. Holistic utilise la pose « full » (complexité 1 par défaut).

Entrée : image **RGB** (OpenCV lit en BGR : convertir). Ressources de dessin : `mp.solutions.drawing_utils`, `drawing_styles`.
Coordonnées : `x, y` normalisés (0-1), `z` relatif (échelle de x) ; les `*_world_*` sont en mètres, origine au centre des hanches (pose).

Points clés fréquents : main — 0 poignet, 4 bout du pouce, 8 bout de l'index, 12 majeur, 16 annulaire, 20 auriculaire. Pose — 0 nez, 11/12 épaules, 13/14 coudes, 15/16 poignets, 23/24 hanches, 25/26 genoux, 27/28 chevilles.

Documentation détaillée dans le dépôt : `docs/solutions/{hands,pose,face_mesh,face_detection,holistic,selfie_segmentation,iris}.md` (documentation de l'ancienne API, utile pour les modèles et les cartes de modèles).
