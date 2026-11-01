#!/usr/bin/env python3
"""Landmarks MediaPipe hors-ligne (mp.solutions, modèles embarqués dans mediapipe==0.10.18).

Usage :
  /tmp/mp-legacy/bin/python legacy_landmarks.py <solution> <entrée image|vidéo> [--out DIR] [--stride N] [--max-frames N]
Solutions : hands | pose | face_mesh | face_detection | holistic | selfie_segmentation
Sorties dans DIR : annotated.png (ou annotated.mp4) + landmarks.json (+ mask.png, cutout.png pour selfie_segmentation)
"""
import argparse, json, os, sys
import cv2, numpy as np
import mediapipe as mp

S = mp.solutions
DRAW = S.drawing_utils


def lm_list(lms):
    return [[round(p.x, 5), round(p.y, 5), round(p.z, 5)] for p in lms.landmark] if lms else None


def make(solution, static):
    if solution == "hands":
        return S.hands.Hands(static_image_mode=static, max_num_hands=2, min_detection_confidence=0.5)
    if solution == "pose":
        return S.pose.Pose(static_image_mode=static, model_complexity=1, min_detection_confidence=0.5)
    if solution == "face_mesh":
        return S.face_mesh.FaceMesh(static_image_mode=static, max_num_faces=2, refine_landmarks=True)
    if solution == "face_detection":
        return S.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    if solution == "holistic":
        return S.holistic.Holistic(static_image_mode=static)
    if solution == "selfie_segmentation":
        return S.selfie_segmentation.SelfieSegmentation(model_selection=1)
    sys.exit(f"solution inconnue : {solution}")


def process(solution, model, bgr):
    """Retourne (image annotée, données JSON-sérialisables)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = model.process(rgb)
    out = bgr.copy()
    h, w = bgr.shape[:2]
    data = {}
    if solution == "hands":
        data["hands"] = []
        for i, lms in enumerate(r.multi_hand_landmarks or []):
            DRAW.draw_landmarks(out, lms, S.hands.HAND_CONNECTIONS)
            label = r.multi_handedness[i].classification[0] if r.multi_handedness else None
            data["hands"].append({"label": label.label if label else None,
                                  "score": round(label.score, 3) if label else None,
                                  "landmarks": lm_list(lms)})
    elif solution == "pose":
        if r.pose_landmarks:
            DRAW.draw_landmarks(out, r.pose_landmarks, S.pose.POSE_CONNECTIONS)
        data["pose"] = lm_list(r.pose_landmarks)
        data["pose_world"] = lm_list(r.pose_world_landmarks)
    elif solution == "face_mesh":
        data["faces"] = []
        for lms in r.multi_face_landmarks or []:
            DRAW.draw_landmarks(out, lms, S.face_mesh.FACEMESH_CONTOURS)
            data["faces"].append(lm_list(lms))
    elif solution == "face_detection":
        data["faces"] = []
        for d in r.detections or []:
            DRAW.draw_detection(out, d)
            bb = d.location_data.relative_bounding_box
            data["faces"].append({"score": round(d.score[0], 3),
                                  "bbox_rel": [round(v, 4) for v in (bb.xmin, bb.ymin, bb.width, bb.height)]})
    elif solution == "holistic":
        for name, lms, conn in (("pose", r.pose_landmarks, S.holistic.POSE_CONNECTIONS),
                                ("left_hand", r.left_hand_landmarks, S.holistic.HAND_CONNECTIONS),
                                ("right_hand", r.right_hand_landmarks, S.holistic.HAND_CONNECTIONS),
                                ("face", r.face_landmarks, S.holistic.FACEMESH_CONTOURS)):
            if lms:
                DRAW.draw_landmarks(out, lms, conn)
            data[name] = lm_list(lms)
    elif solution == "selfie_segmentation":
        mask = r.segmentation_mask  # float32 [0,1], HxW
        data["mask_mean"] = round(float(mask.mean()), 4)
        data["_mask"] = mask
        keep = (mask > 0.5)[..., None]
        out = np.where(keep, bgr, np.full_like(bgr, 255))  # fond blanc
    return out, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution")
    ap.add_argument("input")
    ap.add_argument("--out", default="/tmp/mp-out")
    ap.add_argument("--stride", type=int, default=1, help="traiter 1 image sur N (vidéo)")
    ap.add_argument("--max-frames", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    img = cv2.imread(a.input)
    if img is not None:  # image fixe
        with make(a.solution, True) as m:
            out, data = process(a.solution, m, img)
        mask = data.pop("_mask", None)
        cv2.imwrite(os.path.join(a.out, "annotated.png"), out)
        if mask is not None:
            cv2.imwrite(os.path.join(a.out, "mask.png"), (mask * 255).astype(np.uint8))
        json.dump(data, open(os.path.join(a.out, "landmarks.json"), "w"), indent=1)
        print("image traitée ->", a.out, {k: (len(v) if isinstance(v, list) else v) for k, v in data.items()})
        return

    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        sys.exit(f"entrée illisible : {a.input}")
    fps = (cap.get(cv2.CAP_PROP_FPS) or 25) / a.stride
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(os.path.join(a.out, "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    frames, i, n = [], 0, 0
    with make(a.solution, False) as m:
        while n < a.max_frames:
            ok, f = cap.read()
            if not ok:
                break
            if i % a.stride == 0:
                out, data = process(a.solution, m, f)
                data.pop("_mask", None)
                vw.write(out)
                frames.append({"frame": i, **data})
                n += 1
            i += 1
    vw.release()
    json.dump(frames, open(os.path.join(a.out, "landmarks.json"), "w"))
    print(f"vidéo traitée : {n} images ->", a.out)


if __name__ == "__main__":
    main()
