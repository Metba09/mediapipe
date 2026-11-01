#!/usr/bin/env python3
"""API MediaPipe Tasks sur image, avec un modèle FOURNI par l'utilisateur (.task / .tflite).

Les modèles officiels ne sont pas téléchargeables depuis le sandbox : demander à l'utilisateur de joindre le fichier.

Usage :
  /tmp/mp-tasks/bin/python tasks_run.py <tâche> --model MODELE.task --image IMG [--out DIR] [--max-results N] [--threshold X]
Tâches : pose | hands | face | face_detect | gesture | object | classify | segment
Sorties dans DIR : result.json + annotated.png (ou mask_*.png pour segment)
"""
import argparse, json, os, sys
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

TASKS = ("pose", "hands", "face", "face_detect", "gesture", "object", "classify", "segment")


def pts(lms, w, h):
    return [(int(p.x * w), int(p.y * h)) for p in lms]


def dots(img, lms_list):
    h, w = img.shape[:2]
    for lms in lms_list:
        for x, y in pts(lms, w, h):
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)


def lm_json(lms_list):
    return [[[round(p.x, 5), round(p.y, 5), round(p.z, 5)] for p in lms] for lms in lms_list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=TASKS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="mp_out")
    ap.add_argument("--max-results", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    if not os.path.isfile(a.model):
        sys.exit(f"modèle introuvable : {a.model}\n(le sandbox ne peut pas le télécharger : demander à l'utilisateur de le joindre)")
    os.makedirs(a.out, exist_ok=True)

    base = mpt.BaseOptions(model_asset_path=a.model)
    image = mp.Image.create_from_file(a.image)
    bgr = cv2.imread(a.image)
    h, w = bgr.shape[:2]
    out, res = bgr.copy(), {}

    if a.task == "pose":
        with vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=base, num_poses=2)) as m:
            r = m.detect(image)
        res["poses"] = lm_json(r.pose_landmarks); dots(out, r.pose_landmarks)
    elif a.task == "hands":
        with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=base, num_hands=2)) as m:
            r = m.detect(image)
        res["hands"] = lm_json(r.hand_landmarks)
        res["handedness"] = [[c.category_name for c in hs] for hs in r.handedness]
        dots(out, r.hand_landmarks)
    elif a.task == "face":
        with vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=base, num_faces=2, output_face_blendshapes=True)) as m:
            r = m.detect(image)
        res["faces"] = lm_json(r.face_landmarks)
        res["blendshapes"] = [{c.category_name: round(c.score, 4) for c in bs} for bs in (r.face_blendshapes or [])]
        dots(out, r.face_landmarks)
    elif a.task == "face_detect":
        with vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=base)) as m:
            r = m.detect(image)
        res["detections"] = []
        for d in r.detections:
            b = d.bounding_box
            cv2.rectangle(out, (b.origin_x, b.origin_y), (b.origin_x + b.width, b.origin_y + b.height), (0, 255, 0), 2)
            res["detections"].append({"bbox": [b.origin_x, b.origin_y, b.width, b.height],
                                      "score": round(d.categories[0].score, 3)})
    elif a.task == "gesture":
        with vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(base_options=base, num_hands=2)) as m:
            r = m.recognize(image)
        res["gestures"] = [[(c.category_name, round(c.score, 3)) for c in g] for g in r.gestures]
        res["hands"] = lm_json(r.hand_landmarks); dots(out, r.hand_landmarks)
    elif a.task == "object":
        kw = dict(base_options=base, max_results=a.max_results)
        if a.threshold is not None: kw["score_threshold"] = a.threshold
        with vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(**kw)) as m:
            r = m.detect(image)
        res["detections"] = []
        for d in r.detections:
            b, c = d.bounding_box, d.categories[0]
            cv2.rectangle(out, (b.origin_x, b.origin_y), (b.origin_x + b.width, b.origin_y + b.height), (0, 255, 0), 2)
            cv2.putText(out, f"{c.category_name} {c.score:.2f}", (b.origin_x, max(12, b.origin_y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            res["detections"].append({"label": c.category_name, "score": round(c.score, 3),
                                      "bbox": [b.origin_x, b.origin_y, b.width, b.height]})
    elif a.task == "classify":
        with vision.ImageClassifier.create_from_options(vision.ImageClassifierOptions(
                base_options=base, max_results=a.max_results)) as m:
            r = m.classify(image)
        res["classes"] = [(c.category_name, round(c.score, 4)) for c in r.classifications[0].categories]
    elif a.task == "segment":
        with vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=base, output_category_mask=True, output_confidence_masks=True)) as m:
            r = m.segment(image)
        cm = r.category_mask.numpy_view()
        cv2.imwrite(os.path.join(a.out, "mask_category.png"), (cm.astype(np.uint8) * (255 // max(1, int(cm.max()) or 1))))
        res["category_values"] = sorted(int(v) for v in np.unique(cm))
        for i, c in enumerate(r.confidence_masks or []):
            cv2.imwrite(os.path.join(a.out, f"mask_conf_{i}.png"), (c.numpy_view() * 255).astype(np.uint8))
        out = None

    if out is not None:
        cv2.imwrite(os.path.join(a.out, "annotated.png"), out)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print("ok ->", a.out, {k: (len(v) if hasattr(v, "__len__") else v) for k, v in res.items()})


if __name__ == "__main__":
    main()
