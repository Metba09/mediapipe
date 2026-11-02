#!/usr/bin/env python3
"""Suivi de mouvement d'une vidéo (motion design, plans de caméra...) avec OpenCV : aucun modèle, aucun réseau.

Ce que fait le script
  - détecte les COUPES (plans) par comparaison d'histogrammes couleur ;
  - suit des POINTS CARACTÉRISTIQUES (Lucas-Kanade pyramidal + contrôle aller-retour) et les ré-alimente au fil de l'eau ;
  - estime le MOUVEMENT DOMINANT image par image (translation, échelle, rotation) et son cumul depuis le début du plan ;
  - mesure l'ÉNERGIE DE MOUVEMENT (flux optique dense) ;
  - repère les points IMMOBILES (filigrane, fond fixe) pour ne pas les confondre avec du mouvement.

Usage :
  /tmp/mp-legacy/bin/python track_motion.py VIDEO [--out DIR] [--max-points N] [--cut-threshold X] [--max-frames N]
Sorties dans DIR : tracks.json, motion_curve.csv, shots.csv, annotated.mp4, motion_timeline.png
Coordonnées en pixels de la vidéo d'origine.
"""
import argparse, csv, json, os, shutil, subprocess, sys
import cv2
import numpy as np

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def color_hist(frame):
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 160)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, h).flatten()


def detect(gray, n, existing):
    if n <= 0:
        return np.empty((0, 1, 2), np.float32)
    mask = np.full(gray.shape, 255, np.uint8)
    for x, y in existing.reshape(-1, 2):
        cv2.circle(mask, (int(x), int(y)), 10, 0, -1)
    p = cv2.goodFeaturesToTrack(gray, maxCorners=n, qualityLevel=0.01, minDistance=10, blockSize=7, mask=mask)
    return p if p is not None else np.empty((0, 1, 2), np.float32)


def timeline_png(path, frames, shots, fps):
    W, H, L, B = 1200, 320, 60, 40
    img = np.full((H, W, 3), 255, np.uint8)
    e = np.array([f["flow_energy"] for f in frames], float)
    top = max(1.0, float(np.percentile(e, 99)) * 1.1) if len(e) else 1.0
    n = max(1, len(frames) - 1)
    X = lambda i: int(L + (W - L - 10) * i / n)
    Y = lambda v: int(H - B - (H - B - 20) * min(v, top) / top)
    for s in shots[1:]:
        cv2.line(img, (X(s["start"]), 15), (X(s["start"]), H - B), (60, 60, 220), 1)
    pts = np.array([[X(i), Y(v)] for i, v in enumerate(e)], np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, (180, 90, 20), 2, cv2.LINE_AA)
    cv2.line(img, (L, H - B), (W - 10, H - B), (0, 0, 0), 1)
    cv2.line(img, (L, 15), (L, H - B), (0, 0, 0), 1)
    cv2.putText(img, "energie de mouvement (px/image), traits rouges = coupes", (L, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, f"{top:.1f}", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, "0", (40, H - B), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    for sec in range(0, int(len(frames) / fps) + 1, 1):
        x = X(int(sec * fps))
        if x < W - 10:
            cv2.line(img, (x, H - B), (x, H - B + 5), (0, 0, 0), 1)
            cv2.putText(img, f"{sec}s", (x - 8, H - B + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="/tmp/mp-out/track")
    ap.add_argument("--max-points", type=int, default=150)
    ap.add_argument("--cut-threshold", type=float, default=0.6, help="corrélation d'histogramme sous laquelle on déclare une coupe")
    ap.add_argument("--lost-cut", type=float, default=0.6, help="part de pistes perdues en une image au-delà de laquelle on déclare un changement de scène")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--trail", type=int, default=12, help="longueur des traînées dessinées (images)")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        sys.exit(f"vidéo illisible : {a.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.out, "annotated_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    tracks = {}          # id -> {"shot", "frames", "xy"}
    ids = np.empty(0, int)
    pts = np.empty((0, 1, 2), np.float32)
    next_id, shot = 0, 0
    shots = [{"index": 0, "start": 0}]
    frames = []
    cum = np.eye(3)
    prev_gray = prev_hist = None
    i = 0
    palette = [(0, 160, 255), (0, 200, 0), (255, 120, 0), (200, 0, 200), (0, 200, 200), (60, 60, 255)]

    while True:
        ok, f = cap.read()
        if not ok or (a.max_frames and i >= a.max_frames):
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        hist = color_hist(f)
        rec = {"frame": i, "t": round(i / fps, 3), "shot": shot, "cut": False, "tracked": 0,
               "dx": 0.0, "dy": 0.0, "scale": 1.0, "rot_deg": 0.0,
               "cum_dx": 0.0, "cum_dy": 0.0, "cum_scale": 1.0, "cum_rot_deg": 0.0,
               "flow_energy": 0.0, "moving_fraction": 0.0, "lost_fraction": 0.0, "cut_reason": ""}

        cut_reason, lk = None, None
        if prev_gray is not None:
            if cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < a.cut_threshold:
                cut_reason = "histogramme"
            elif len(pts):
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **LK)
                back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **LK)
                err = np.linalg.norm(pts - back, axis=2).ravel()
                x, y = nxt[:, 0, 0], nxt[:, 0, 1]
                good = (st.ravel() == 1) & (st2.ravel() == 1) & (err < 1.0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
                lost = 1.0 - float(good.sum()) / len(pts)
                rec["lost_fraction"] = round(lost, 3)
                if len(pts) >= 30 and lost > a.lost_cut:
                    cut_reason = "perte de pistes"  # changement de scène sur fond semblable
                else:
                    lk = (pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2), good, nxt)
        if cut_reason == "perte de pistes" and i - shots[-1]["start"] < 4:
            # transition en cours (flash, fondu) : pas de nouveau plan, on repart de zéro sur les points
            ids, pts = np.empty(0, int), np.empty((0, 1, 2), np.float32)
            cut_reason = None
        if cut_reason:
            shots[-1]["end"] = i - 1
            shot += 1
            shots.append({"index": shot, "start": i, "reason": cut_reason})
            ids, pts, cum = np.empty(0, int), np.empty((0, 1, 2), np.float32), np.eye(3)
            rec.update(shot=shot, cut=True, cut_reason=cut_reason)
        elif prev_gray is not None:
            if lk is not None:
                p0, p1, good, nxt = lk
                ids, pts = ids[good], nxt[good]
                rec["tracked"] = int(len(p0))
                if len(p0) >= 6:
                    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if M is not None:
                        s = float(np.hypot(M[0, 0], M[1, 0]))
                        r = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
                        rec.update(dx=round(float(M[0, 2]), 3), dy=round(float(M[1, 2]), 3), scale=round(s, 5), rot_deg=round(r, 3))
                        cum = np.vstack([M, [0, 0, 1]]) @ cum
                        c = cum @ np.array([W / 2, H / 2, 1.0])
                        rec.update(cum_dx=round(float(c[0] - W / 2), 2), cum_dy=round(float(c[1] - H / 2), 2),
                                   cum_scale=round(float(np.hypot(cum[0, 0], cum[1, 0])), 5),
                                   cum_rot_deg=round(float(np.degrees(np.arctan2(cum[1, 0], cum[0, 0]))), 3))
            # énergie de mouvement (flux dense, demi-résolution)
            sm0, sm1 = cv2.resize(prev_gray, (W // 2, H // 2)), cv2.resize(gray, (W // 2, H // 2))
            fl = cv2.calcOpticalFlowFarneback(sm0, sm1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(fl, axis=2) * 2.0
            rec["flow_energy"] = round(float(mag.mean()), 4)
            rec["moving_fraction"] = round(float((mag > 1.0).mean()), 4)

        # ré-alimentation des points
        if len(pts) < a.max_points * 0.7:
            new = detect(gray, a.max_points - len(pts), pts)
            if len(new):
                nid = np.arange(next_id, next_id + len(new))
                next_id += len(new)
                ids, pts = np.concatenate([ids, nid]), np.concatenate([pts, new]) if len(pts) else new
                for k in nid:
                    tracks[int(k)] = {"shot": shot, "frames": [], "xy": []}
        for k, p in zip(ids, pts.reshape(-1, 2)):
            t = tracks[int(k)]
            t["frames"].append(i)
            t["xy"].append([round(float(p[0]), 2), round(float(p[1]), 2)])

        # image annotée
        out = f.copy()
        col = palette[shot % len(palette)]
        for k, p in zip(ids, pts.reshape(-1, 2)):
            xy = tracks[int(k)]["xy"]
            if len(xy) >= 3:
                moved = np.hypot(xy[-1][0] - xy[0][0], xy[-1][1] - xy[0][1])
                if moved < 2.0 and len(xy) > 8:
                    continue  # immobile : non dessiné
            tr = np.array(xy[-a.trail:], np.int32)
            if len(tr) > 1:
                cv2.polylines(out, [tr.reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
            cv2.circle(out, (int(p[0]), int(p[1])), 3, col, -1)
        label = f"plan {shot}  img {i}  {rec['tracked']} pts  dx {rec['dx']:+.1f} dy {rec['dy']:+.1f}  x{rec['scale']:.3f}"
        cv2.rectangle(out, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.putText(out, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if rec["cut"]:
            cv2.rectangle(out, (0, 0), (W - 1, H - 1), (0, 0, 255), 6)
        vw.write(out)

        frames.append(rec)
        prev_gray, prev_hist = gray, hist
        i += 1

    vw.release()
    if shots:
        shots[-1]["end"] = i - 1
    for s in shots:
        s["start_t"], s["end_t"], s["n_frames"] = round(s["start"] / fps, 3), round((s["end"] + 1) / fps, 3), s["end"] - s["start"] + 1

    track_list = []
    for k, t in tracks.items():
        xy = np.array(t["xy"])
        static = bool(len(xy) >= 10 and np.hypot(*(xy - xy[0]).T).max() < 2.0)
        track_list.append({"id": k, "shot": t["shot"], "static": static, "length": len(xy), "frames": t["frames"], "xy": t["xy"]})

    meta = {"source": os.path.basename(a.input), "fps": round(fps, 3), "width": W, "height": H, "n_frames": i,
            "params": {"max_points": a.max_points, "cut_threshold": a.cut_threshold}}
    json.dump({"video": meta, "shots": shots, "frames": frames, "tracks": track_list},
              open(os.path.join(a.out, "tracks.json"), "w"), separators=(",", ":"))
    with open(os.path.join(a.out, "motion_curve.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frames[0].keys()))
        w.writeheader(); w.writerows(frames)
    with open(os.path.join(a.out, "shots.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["index", "start", "end", "start_t", "end_t", "n_frames", "reason"], restval="")
        w.writeheader(); w.writerows(shots)
    timeline_png(os.path.join(a.out, "motion_timeline.png"), frames, shots, fps)

    final = os.path.join(a.out, "annotated.mp4")
    if shutil.which("ffmpeg"):  # H.264 lisible partout
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", final])
        if r.returncode == 0:
            os.remove(raw)
        else:
            os.replace(raw, final)
    else:
        os.replace(raw, final)

    mov = [t for t in track_list if not t["static"]]
    print(f"{i} images, {len(shots)} plans, {len(track_list)} pistes ({len(mov)} en mouvement, {len(track_list) - len(mov)} immobiles) -> {a.out}")
    for s in shots:
        print(f"  plan {s['index']:>2} : images {s['start']}-{s['end']}  ({s['start_t']:.2f}s -> {s['end_t']:.2f}s)")


if __name__ == "__main__":
    main()
