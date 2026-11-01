# API Tasks (mediapipe ≥ 1.0) — relevé dans les sources du dépôt

Source : `mediapipe/tasks/python/{vision,text,audio}/*.py`. Modèle de base commun :
```python
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
base = mpt.BaseOptions(model_asset_path="modele.task")   # ou model_asset_buffer=bytes
```
Modes d'exécution (`running_mode`) : `IMAGE` (défaut), `VIDEO` (`*_for_video(image, timestamp_ms)`), `LIVE_STREAM` (`*_async` + `result_callback`). Image d'entrée : `mp.Image.create_from_file(path)` ou `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_ndarray)`.

## Vision
| Classe | Options principales | Méthodes |
|---|---|---|
| `PoseLandmarker` | num_poses, min_pose_detection_confidence, min_pose_presence_confidence, min_tracking_confidence, output_segmentation_masks | detect, detect_for_video, detect_async |
| `HandLandmarker` | num_hands, min_hand_detection_confidence, min_hand_presence_confidence, min_tracking_confidence | idem |
| `FaceLandmarker` | num_faces, min_face_detection_confidence, min_face_presence_confidence, min_tracking_confidence, output_face_blendshapes, output_facial_transformation_matrixes | idem |
| `FaceDetector` | min_detection_confidence, min_suppression_threshold | idem |
| `HolisticLandmarker` | seuils face/pose/main, output_face_blendshapes, output_segmentation_mask | idem |
| `GestureRecognizer` | num_hands, seuils main, canned/custom_gesture_classifier_options | recognize, recognize_for_video, recognize_async |
| `ObjectDetector` | max_results, score_threshold, category_allowlist/denylist, display_names_locale | detect, detect_for_video, detect_async |
| `ImageClassifier` | max_results, score_threshold, category_allowlist/denylist | classify, classify_for_video, classify_async |
| `ImageEmbedder` | l2_normalize, quantize | embed, embed_for_video, `cosine_similarity` |
| `ImageSegmenter` | output_confidence_masks, output_category_mask | segment, segment_for_video, `labels` |
| `InteractiveSegmenter` | (options de base) | set_image, segment |

Dessin : `mediapipe/tasks/python/vision/drawing_utils.py` (`draw_landmarks`, `draw_detection`, `draw_axis`, `plot_landmarks`) et `drawing_styles.py`. Constantes de connexions : `PoseLandmarksConnections`, `HandLandmarksConnections`, `FaceLandmarksConnections`, enums `PoseLandmark`, `HandLandmark`.

## Texte et audio
`TextClassifier`, `TextEmbedder` (`cosine_similarity`), `LanguageDetector`, `AudioClassifier`, `AudioEmbedder` ; des classes plus récentes (`TextSummarizer`, `TextProofreader`, `AudioTranscriber`, `TextToSpeech`) existent dans les sources et reposent sur de gros modèles généraux : non vérifiées dans le sandbox.

## Résultats (champs usuels)
- Landmarkers : `pose_landmarks / hand_landmarks / face_landmarks` (listes de listes de points x,y,z), `handedness`, `face_blendshapes`, `segmentation_masks`.
- Détecteurs : `detections[i].bounding_box` (origin_x, origin_y, width, height en pixels) et `.categories[0]` (category_name, score).
- Segmenteur : `category_mask.numpy_view()`, `confidence_masks[i].numpy_view()`.

## Modèles
Les modèles officiels se téléchargent depuis `storage.googleapis.com/mediapipe-models/...` : **inaccessible** dans le sandbox. Faire joindre le fichier par l'utilisateur. Entraîner ses propres modèles : `mediapipe/model_maker/python/vision/{gesture_recognizer,image_classifier,object_detector}` et `text/text_classifier` (nécessite TensorFlow, lourd : à proposer seulement si demandé).
