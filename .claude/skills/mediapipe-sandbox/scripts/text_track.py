#!/usr/bin/env python3
"""Suivi des TEXTES animés d'une vidéo (motion typographique) -> un JSON : expressions, trajectoires, styles.

Hors-ligne (aucun modèle à télécharger) :
  - OCR : rapidocr-onnxruntime (modèles PP-OCRv4 embarqués dans le paquet pip), lecture image par image ;
  - correction : tesseract (fra+eng) sur la meilleure image de chaque élément (espaces, accents) ;
  - liaison des détections en ÉLÉMENTS suivis dans le temps ;
  - trajectoire : par image (centre, taille, angle, opacité estimée) + images clés simplifiées ;
  - style MESURÉ sur les pixels : couleur, fond, taille, graisse, italique, casse, halo, dégradé.

Les polices ne sont pas identifiées : seule une classe grossière (serif à fort contraste / sans) est estimée.

Usage :
  /tmp/mp-ocr/bin/python text_track.py VIDEO [--out DIR] [--shots tracks.json] [--phase ocr|all] [--stride 1]
Installation : bash setup_env.sh ocr   (crée /tmp/mp-ocr : Python 3.12 + rapidocr-onnxruntime + OpenCV)
Sorties dans DIR : text_tracking.json, ocr_cache.json, text_preview.mp4
"""
import argparse, difflib, json, math, os, re, shutil, subprocess, sys, tempfile
from collections import defaultdict
import cv2
import numpy as np

SCHEMA = "text-motion-tracking/1"
try:
    import wordninja  # facultatif : découpe les longues chaînes de lettres sans espaces
except Exception:  # pragma: no cover
    wordninja = None


# ----------------------------------------------------------------------------- OCR
def run_ocr(video, cache_path, stride):
    from rapidocr_onnxruntime import RapidOCR
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"vidéo illisible : {video}")
    meta = {"fps": cap.get(cv2.CAP_PROP_FPS) or 25.0, "w": int(cap.get(3)), "h": int(cap.get(4)), "n": int(cap.get(7))}
    cache = {"video": meta, "stride": stride, "complete": False, "frames": []}
    if os.path.isfile(cache_path):
        old = json.load(open(cache_path))
        if old.get("video") == meta and old.get("stride") == stride:
            cache = old
            if cache["complete"]:
                return cache
    done = len(cache["frames"])
    ocr = RapidOCR()
    cap.set(cv2.CAP_PROP_POS_FRAMES, done)
    i = done
    while True:
        ok, f = cap.read()
        if not ok:
            break
        dets = []
        if i % stride == 0:
            res, _ = ocr(f)
            for box, txt, conf in (res or []):
                dets.append({"t": txt, "c": round(float(conf), 3), "q": [[round(float(x), 1), round(float(y), 1)] for x, y in box]})
        cache["frames"].append(dets)
        i += 1
        if i % 40 == 0:
            json.dump(cache, open(cache_path, "w"))
            print(f"  OCR {i}/{meta['n']}", flush=True)
    cache["complete"] = True
    json.dump(cache, open(cache_path, "w"))
    return cache


# ----------------------------------------------------------------------------- géométrie / liaison
def box_info(q):
    q = np.array(q, float)
    tl, tr, br, bl = q
    w, h = float(np.linalg.norm(tr - tl)), float(np.linalg.norm(bl - tl))
    ang = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
    cx, cy = q[:, 0].mean(), q[:, 1].mean()
    x0, y0, x1, y1 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
    return {"cx": float(cx), "cy": float(cy), "w": w, "h": h, "angle": ang, "bbox": (float(x0), float(y0), float(x1), float(y1))}


def norm_text(t):
    return re.sub(r"[^0-9a-zà-ÿ@]", "", t.lower())


def sim(a, b):
    a, b = norm_text(a), norm_text(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def iou(b1, b2):
    ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    inter = ix * iy
    u = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter
    return inter / u if u > 0 else 0.0


def link(cache, cuts, a):
    W, H = cache["video"]["w"], cache["video"]["h"]
    tracks, active, nid = [], [], 0
    for i, dets in enumerate(cache["frames"]):
        infos = []
        for d in dets:
            if d["c"] < a.min_conf or not norm_text(d["t"]):
                continue
            b = box_info(d["q"])
            b.update(text=d["t"], conf=d["c"], quad=d["q"], frame=i)
            infos.append(b)
        cands = []
        for ti, tr in enumerate(active):
            last = tr["obs"][-1]
            gap = i - last["frame"]
            if gap > a.max_gap:
                continue
            for di, d in enumerate(infos):
                s = sim(last["text"], d["text"])
                dist = math.hypot(d["cx"] - last["cx"], d["cy"] - last["cy"])
                ov = iou(last["bbox"], d["bbox"])
                sz = min(d["h"], last["h"]) / max(d["h"], last["h"], 1e-6)
                if i in cuts and not (dist < 12 and s >= 0.9 and sz > 0.9):
                    continue
                L = max(W, H)
                # taille proche d'une image à l'autre ; petit déplacement : texte assez proche ; grand déplacement : texte quasi identique
                ok = sz >= 0.4 and ((s >= 0.6 and dist <= 0.12 * L * gap) or (s >= 0.85 and dist <= 0.55 * L * gap) or (ov >= 0.3 and s >= 0.5))
                if ok:
                    cands.append((s + ov - dist / (2.0 * max(W, H)), ti, di))
        cands.sort(reverse=True)
        used_t, used_d = set(), set()
        for sc, ti, di in cands:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            active[ti]["obs"].append(infos[di])
        for di, d in enumerate(infos):
            if di not in used_d:
                tr = {"id": nid, "obs": [d]}
                nid += 1
                tracks.append(tr); active.append(tr)
        active = [t for t in active if i - t["obs"][-1]["frame"] <= a.max_gap]
    return tracks


# ----------------------------------------------------------------------------- mesure du style
def hexc(bgr):
    b, g, r = [int(round(v)) for v in bgr]
    return f"#{r:02x}{g:02x}{b:02x}"


def lab(bgr_img):
    return cv2.cvtColor(bgr_img.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)


def asc_desc(text):
    t = re.sub(r"[^A-Za-zÀ-ÿ0-9]", "", text)
    top = any(c.isupper() or c.isdigit() or c in "bdfhklt" for c in t)
    desc = any(c in "gjpqy" for c in t)
    return top, desc


def measure(frame, obs):
    """Mesure le style d'une détection (frame BGR). Retourne un dict ou None."""
    Hh, Ww = frame.shape[:2]
    x0, y0, x1, y1 = obs["bbox"]
    pad = int(max(6, 0.2 * obs["h"]))
    X0, Y0, X1, Y1 = max(0, int(x0) - pad), max(0, int(y0) - pad), min(Ww, int(x1) + pad), min(Hh, int(y1) + pad)
    crop = frame[Y0:Y1, X0:X1]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    L = lab(crop)
    ring = np.zeros(crop.shape[:2], bool)
    r = 3
    ring[:r, :] = ring[-r:, :] = ring[:, :r] = ring[:, -r:] = True
    # le cadre de l'image n'est pas du fond : on retire les bords collés au bord de l'image
    if Y0 == 0: ring[:r, :] = False
    if Y1 == Hh: ring[-r:, :] = False
    if X0 == 0: ring[:, :r] = False
    if X1 == Ww: ring[:, -r:] = False
    if ring.sum() < 20:
        ring[:] = False; ring[:r, :] = ring[-r:, :] = ring[:, :r] = ring[:, -r:] = True
    bg_lab = np.median(L[ring], axis=0)
    bg_bgr = np.median(crop[ring], axis=0)
    bg_spread = float(np.mean(np.std(L[ring], axis=0)))
    D = np.linalg.norm(L - bg_lab, axis=2)
    D8 = np.clip(D * (255.0 / max(D.max(), 1.0)), 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(D8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    M = (D8 > thr).astype(np.uint8)
    poly = np.zeros(crop.shape[:2], np.uint8)
    q = (np.array(obs["quad"]) - [X0, Y0]).astype(np.int32)
    cv2.fillPoly(poly, [q], 1)
    poly = cv2.dilate(poly, np.ones((5, 5), np.uint8))
    M &= poly
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(M, connectivity=8)
    keep = np.zeros(n, bool)
    min_area = max(4, 0.003 * obs["w"] * obs["h"])
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    M = keep[lbl].astype(np.uint8)
    if M.sum() < 20:
        return None
    ys, xs = np.where(M > 0)
    ink_h, ink_w = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    res = {"ink_w": ink_w, "ink_h": ink_h, "bg": hexc(bg_bgr), "bg_spread": bg_spread}

    # couleur du texte : pixels d'encre les plus éloignés du fond
    dd = D[M > 0]
    core = (M > 0) & (D >= np.percentile(dd, 50))
    pix = crop[core]
    res["color"] = hexc(np.median(pix, axis=0))
    res["ink_bg_dist"] = float(np.median(dd))
    # groupes de couleurs de l'encre : couleur retenue = groupe coloré assez présent, sinon groupe dominant
    if len(pix) > 80:
        Z = lab(pix.reshape(-1, 1, 3)).reshape(-1, 3)
        K = 3
        _, labels, centers = cv2.kmeans(Z, K, None, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
        lb = labels.ravel()
        share = np.bincount(lb, minlength=K) / len(Z)
        cl = []
        for k in range(K):
            if share[k] == 0:
                continue
            col = np.median(pix[lb == k], axis=0)
            cl.append({"hex": hexc(col), "share": float(share[k]), "chroma": float(col.max() - col.min()), "lab": centers[k]})
        cl.sort(key=lambda c: -c["share"])
        res["clusters"] = [(c["hex"], round(c["share"], 3)) for c in cl]
        chrom = [c for c in cl if c["share"] >= 0.2 and c["chroma"] >= 50]
        if chrom:
            res["color"] = max(chrom, key=lambda c: c["share"])["hex"]
        if len(cl) >= 2 and cl[1]["share"] >= 0.2 and np.linalg.norm(cl[0]["lab"] - cl[1]["lab"]) > 35:
            res["multicolor"] = [cl[0]["hex"], cl[1]["hex"]]

    # taille de police estimée (empan de l'encre / proportion typique des glyphes)
    top, desc = asc_desc(obs["text"])
    prop = (0.72 if top else 0.5) + (0.2 if desc else 0.0)
    res["font_px"] = ink_h / prop

    # graisse : épaisseur de trait par transformée de distance sur le squelette approché
    dt = cv2.distanceTransform(M, cv2.DIST_L2, 3)
    mx = (dt == cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt >= 0.8)
    if mx.sum() >= 5:
        wd = 2.0 * dt[mx]
        res["stroke_px"] = float(np.median(wd))
        res["stroke_contrast"] = float(np.percentile(wd, 90) / max(np.median(wd), 1e-6))
    # inclinaison : cisaillement qui maximise la verticalité des traits
    if ink_h >= 12:
        sub = M[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
        h_, w_ = sub.shape
        best, base = (0, 0.0), None
        for th in range(-25, 26, 1):
            t = math.tan(math.radians(th))
            Mx = np.float32([[1, t, -(h_ / 2) * t + (h_ * abs(t))], [0, 1, 0]])
            wimg = cv2.warpAffine(sub, Mx, (w_ + int(h_ * abs(t)) * 2 + 2, h_), flags=cv2.INTER_NEAREST)
            sc = float(np.sum(wimg.sum(axis=0) ** 2))
            if th == 0:
                base = sc
            if sc > best[1]:
                best = (th, sc)
        gain = best[1] / base - 1.0 if base else 0.0
        res["slant_deg"] = float(best[0]) if gain >= 0.04 else 0.0
        res["slant_gain"] = float(gain)
    # halo / contour / lueur : couleur de l'anneau autour de l'encre
    ring2 = cv2.dilate(M, np.ones((7, 7), np.uint8)) & ~cv2.dilate(M, np.ones((3, 3), np.uint8)) & poly
    if ring2.sum() > 15:
        rc = np.median(crop[ring2 > 0], axis=0)
        dist_bg = float(np.linalg.norm(lab(rc.reshape(1, 1, 3))[0, 0] - bg_lab))
        res["halo_dist"] = dist_bg
        res["halo"] = hexc(rc)
    # espacement des lettres : écart médian entre composantes / taille
    n2, lbl2, st2, _ = cv2.connectedComponentsWithStats(M, connectivity=8)
    comps = sorted([(st2[k, 0], st2[k, 0] + st2[k, 2]) for k in range(1, n2) if st2[k, 4] >= min_area])
    if len(comps) >= 3:
        gaps = [comps[k + 1][0] - comps[k][1] for k in range(len(comps) - 1)]
        pos = [g for g in gaps if g > 0]
        res["gap_px"] = float(np.median(pos)) if pos else 0.0
    return res


def weight_class(ratio):
    if ratio is None:
        return None
    for lim, name in ((0.05, "light"), (0.085, "regular"), (0.12, "medium"), (0.17, "bold")):
        if ratio < lim:
            return name
    return "black"


def clean_text(t):
    """Retire les signes parasites en bordure (guillemets, barres, tiret bas) ; garde le « _ » d'un identifiant @."""
    t = t.strip()
    t = re.sub(r"^[\s‘’“”\"'|_*•»«,.;:]+", "", t)
    junk = "‘’“”\"'|*•»«" + ("" if "@" in t else "_")
    t = re.sub(r"[\s" + re.escape(junk) + r"]+$", "", t)
    if " " not in t and re.fullmatch(r"[a-zà-ÿ]+[.,;:!]+", t):  # mot isolé en minuscules : la ponctuation finale est un artefact
        t = t.rstrip(".,;:!")
    return t


def segment_text(t):
    """Insère des espaces dans les longues suites de lettres (petits textes mal espacés). Retourne (texte, modifié?)."""
    if wordninja is None:
        return t, False
    out, changed = [], False
    for tok in t.split(" "):
        if tok.startswith("@") or "http" in tok or len(re.sub(r"[^A-Za-z]", "", tok)) < 14:
            out.append(tok)
            continue
        new = re.sub(r"[A-Za-z]{10,}", lambda m: " ".join(wordninja.split(m.group(0))), tok)
        changed = changed or new != tok
        out.append(new)
    return " ".join(out), changed


def case_of(t):
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return "n/a"
    if all(c.isupper() for c in letters):
        return "MAJUSCULES"
    if all(c.islower() for c in letters):
        return "minuscules"
    words = [w for w in re.split(r"\s+", t) if any(c.isalpha() for c in w)]
    if words and all(w[0].isupper() for w in words):
        return "Capitales initiales"
    return "Mixte"


# ----------------------------------------------------------------------------- tesseract
def crop_for(frame, obs):
    x0, y0, x1, y1 = obs["bbox"]
    pad = int(max(4, 0.15 * obs["h"]))
    Hh, Ww = frame.shape[:2]
    return frame[max(0, int(y0) - pad):min(Hh, int(y1) + pad), max(0, int(x0) - pad):min(Ww, int(x1) + pad)].copy()


def tess_refine(c):
    if not shutil.which("tesseract") or c is None or c.size == 0:
        return None
    s = max(1.0, 96.0 / c.shape[0])
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    if g[:3].mean() < 110:  # fond sombre : inverser pour tesseract
        g = 255 - g
    g = cv2.copyMakeBorder(g, 16, 16, 16, 16, cv2.BORDER_REPLICATE)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.png")
        cv2.imwrite(p, g)
        r = subprocess.run(["tesseract", p, "stdout", "-l", "fra+eng", "--psm", "7", "tsv"], capture_output=True, text=True)
    words, confs = [], []
    for line in r.stdout.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 12 and cols[11].strip():
            words.append(cols[11].strip()); confs.append(float(cols[10]))
    if not words:
        return None
    return {"text": " ".join(words), "conf": round(float(np.mean(confs)) / 100.0, 3)}


def pick_candidates(obs, k=3):
    """Images les plus stables (vitesse minimale) parmi les plus sûres : meilleures pour relire le texte."""
    if len(obs) == 1:
        return [obs[0]["frame"]]
    mc = max(o["conf"] for o in obs)
    good = [j for j, o in enumerate(obs) if o["conf"] >= 0.9 * mc]

    def speed(j):
        v = []
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(obs):
                v.append(math.hypot(obs[j]["cx"] - obs[jj]["cx"], obs[j]["cy"] - obs[jj]["cy"]) / max(1, abs(obs[j]["frame"] - obs[jj]["frame"])))
        return max(v) if v else 0.0
    good.sort(key=lambda j: (round(speed(j), 1), -obs[j]["conf"], abs(j - len(obs) // 2)))
    return [obs[j]["frame"] for j in good[:k]]


# ----------------------------------------------------------------------------- simplification des images clés
def rdp_keyframes(frames, vals, tol):
    """Garde les images où l'interpolation linéaire s'écarte de plus de tol (dimensions pondérées)."""
    n = len(frames)
    if n <= 2:
        return list(range(n))
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        t = (np.array(frames[a + 1:b]) - frames[a]) / max(frames[b] - frames[a], 1)
        interp = vals[a][None, :] + t[:, None] * (vals[b] - vals[a])[None, :]
        err = np.abs(vals[a + 1:b] - interp).max(axis=1)
        k = int(np.argmax(err))
        if err[k] > tol:
            keep.add(a + 1 + k)
            stack.append((a, a + 1 + k)); stack.append((a + 1 + k, b))
    return sorted(keep)


def hexdist(h1, h2):
    a = np.array([int(h1[i:i + 2], 16) for i in (1, 3, 5)], float)
    b = np.array([int(h2[i:i + 2], 16) for i in (1, 3, 5)], float)
    return float(np.linalg.norm(a - b))


# ----------------------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="/tmp/mp-out/text")
    ap.add_argument("--shots", default=None, help="tracks.json de track_motion.py (coupes de plan)")
    ap.add_argument("--phase", choices=["ocr", "all"], default="all")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--review", default=None, help="JSON de revue visuelle : [{text, start_t, set:{...}}] ajouté sous element.visual_review")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--max-gap", type=int, default=3)
    ap.add_argument("--min-len", type=int, default=3, help="durée minimale (images) d'un élément gardé s'il est peu fiable")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache_path = os.path.join(a.out, "ocr_cache.json")
    cache = run_ocr(a.video, cache_path, a.stride)
    if a.phase == "ocr":
        print("OCR terminé ->", cache_path)
        return
    meta = cache["video"]
    fps, W, H, N = meta["fps"], meta["w"], meta["h"], len(cache["frames"])

    shots, cuts = [], set()
    if a.shots and os.path.isfile(a.shots):
        shots = json.load(open(a.shots)).get("shots", [])
        cuts = {s["start"] for s in shots if s["start"] > 0}

    tracks = link(cache, cuts, a)

    # --- consolidation du texte et filtrage du bruit
    elements = []
    for tr in tracks:
        obs = tr["obs"]
        score = defaultdict(float)
        for o in obs:
            score[o["text"]] += o["conf"]
        variants = sorted(score.items(), key=lambda kv: -kv[1])
        conf = float(np.median([o["conf"] for o in obs]))
        if len(obs) < a.min_len and conf < 0.9:
            continue
        tr.update(text=variants[0][0], variants=[{"text": t, "weight": round(s, 2)} for t, s in variants[:4]], conf=conf)
        elements.append(tr)

    # --- une seule passe sur la vidéo : mesures par image
    by_frame = defaultdict(list)
    for tr in elements:
        for o in tr["obs"]:
            by_frame[o["frame"]].append((tr, o))
    cand = {}
    for tr in elements:
        tr["cand_frames"] = pick_candidates(tr["obs"])
        for fi in tr["cand_frames"]:
            cand[(tr["id"], fi)] = None
    cap = cv2.VideoCapture(a.video)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        for tr, o in by_frame.get(i, []):
            o["m"] = measure(f, o)
            if (tr["id"], i) in cand:
                cand[(tr["id"], i)] = crop_for(f, o)
        i += 1
    # tesseract sur les images candidates : on garde la lecture la plus cohérente avec rapidocr
    for tr in elements:
        tr["rapid_text"] = tr["text"]
        best_t, best_sc = None, -1.0
        for fi in tr["cand_frames"]:
            t = tess_refine(cand.get((tr["id"], fi)))
            if not t:
                continue
            t["similarity"] = round(sim(tr["text"], t["text"]), 3)
            t["frame"] = fi
            sc = 0.6 * t["similarity"] + 0.4 * t["conf"]
            if sc > best_sc:
                best_t, best_sc = t, sc
        tr["tess"] = best_t
        if best_t and best_t["similarity"] >= 0.8 and (best_t["conf"] >= 0.5 or (" " in best_t["text"] and " " not in tr["text"])):
            tr["text"] = best_t["text"]
        tr["text"] = clean_text(tr["text"]) or tr["rapid_text"]
        tr["text_unsegmented"] = tr["text"]
        tr["text"], tr["segmented"] = segment_text(tr["text"])
    cand.clear()

    # --- éléments : trajectoire, style, résumé
    out_el = []
    for k, tr in enumerate(sorted(elements, key=lambda t: (t["obs"][0]["frame"], t["obs"][0]["cy"]))):
        obs = tr["obs"]
        ms = [o["m"] for o in obs if o.get("m")]
        # trajectoire par image, avec comblement linéaire des trous
        fr = [o["frame"] for o in obs]
        cols = ["frame", "t", "cx", "cy", "w", "h", "angle", "conf", "opacity", "color", "interpolated"]
        dmax = max([m["ink_bg_dist"] for m in ms], default=1.0)
        rows = []
        for j, o in enumerate(obs):
            if j > 0 and o["frame"] - obs[j - 1]["frame"] > 1:
                p = obs[j - 1]
                for g in range(p["frame"] + 1, o["frame"]):
                    u = (g - p["frame"]) / (o["frame"] - p["frame"])
                    rows.append([g, round(g / fps, 3)] + [round(p[key] + u * (o[key] - p[key]), 2) for key in ("cx", "cy", "w", "h", "angle")] + [0.0, None, None, 1])
            m = o.get("m")
            rows.append([o["frame"], round(o["frame"] / fps, 3), round(o["cx"], 2), round(o["cy"], 2), round(o["w"], 2), round(o["h"], 2),
                         round(o["angle"], 2), o["conf"], round(min(1.0, m["ink_bg_dist"] / dmax), 3) if m else None, m["color"] if m else None, 0])
        arr = np.array([[r[2], r[3], r[4], r[5], r[6] * 3.0, (r[8] if r[8] is not None else 1.0) * 40.0] for r in rows], float)
        kidx = rdp_keyframes([r[0] for r in rows], arr, tol=2.0)
        keyframes = [{"frame": rows[j][0], "t": rows[j][1], "x": rows[j][2], "y": rows[j][3], "nx": round(rows[j][2] / W, 4), "ny": round(rows[j][3] / H, 4),
                      "w": rows[j][4], "h": rows[j][5], "scale": round(rows[j][5] / rows[0][5], 3) if rows[0][5] else None,
                      "rotation_deg": rows[j][6], "opacity": rows[j][8], "color": rows[j][9]} for j in kidx]
        pos = np.array([[r[2], r[3]] for r in rows])
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.array([0.0])
        total_disp = float(np.hypot(*(pos[-1] - pos[0])))
        static = bool(len(rows) >= 10 and np.abs(pos - pos[0]).max() < 4.0)
        traj = {"columns": cols, "per_frame": rows, "keyframes": keyframes,
                "summary": {"start": {"x": rows[0][2], "y": rows[0][3], "w": rows[0][4], "h": rows[0][5]},
                            "end": {"x": rows[-1][2], "y": rows[-1][3], "w": rows[-1][4], "h": rows[-1][5]},
                            "displacement_px": round(total_disp, 1), "path_length_px": round(float(steps.sum()), 1),
                            "max_speed_px_per_s": round(float(steps.max() * fps), 1), "height_ratio_end_start": round(rows[-1][5] / rows[0][5], 3) if rows[0][5] else None}}
        # style : médiane des mesures (images nettes) ; couleur = valeur la plus fréquente
        style = {}
        if ms:
            colors = [m["color"] for m in ms]
            style["color"] = max(set(colors), key=lambda c: sum(1 for x in colors if hexdist(x, c) < 30))
            cand_cols = []
            for m in ms:
                for hx_, sh_ in m.get("clusters", []):
                    for cc in cand_cols:
                        if hexdist(cc["hex"], hx_) < 40:
                            cc["w"] += sh_; break
                    else:
                        cand_cols.append({"hex": hx_, "w": sh_})
            tot = sum(c["w"] for c in cand_cols) or 1.0
            style["color_candidates"] = [{"hex": c["hex"], "share": round(c["w"] / tot, 3)} for c in sorted(cand_cols, key=lambda c: -c["w"])[:3]]
            style["background"] = max(set(m["bg"] for m in ms), key=[m["bg"] for m in ms].count)
            style["background_complex"] = bool(np.median([m["bg_spread"] for m in ms]) > 18)
            mc = [m["multicolor"] for m in ms if m.get("multicolor")]
            if len(mc) >= max(2, len(ms) // 3):
                style["multicolor"] = mc[len(mc) // 2]
            fpx = float(np.median([m["font_px"] for m in ms]))
            style["font_size_px_est"] = round(fpx, 1)
            style["font_size_pct_of_height"] = round(100 * fpx / H, 2)
            style["ink_height_px"] = round(float(np.median([m["ink_h"] for m in ms])), 1)
            sp = [m["stroke_px"] for m in ms if "stroke_px" in m]
            if sp:
                spx = float(np.median(sp))
                style["stroke_px"] = round(spx, 2)
                style["stroke_to_em"] = round(spx / fpx, 3) if fpx else None
                style["weight_class"] = weight_class(spx / fpx if fpx else None)
                sc = float(np.median([m["stroke_contrast"] for m in ms if "stroke_contrast" in m]))
                style["stroke_contrast"] = round(sc, 2)
                style["font_class_guess"] = ("serif à fort contraste, type Didone (heuristique)" if sc >= 2.0 else "serif à contraste moyen (heuristique)" if sc >= 1.45 else "sans-serif / contraste faible (heuristique)")
            sl = [m["slant_deg"] for m in ms if "slant_deg" in m]
            if sl:
                sd = float(np.median(sl))
                style["slant_deg"] = round(sd, 1)
                style["italic"] = bool(sd >= 7)
            hd = [m for m in ms if "halo_dist" in m]
            if hd:
                hdist = float(np.median([m["halo_dist"] for m in hd]))
                style["halo"] = {"present": bool(hdist > 28), "color": max(set(m["halo"] for m in hd), key=[m["halo"] for m in hd].count), "distance_to_bg": round(hdist, 1)}
            gp = [m["gap_px"] for m in ms if "gap_px" in m]
            if gp and fpx:
                style["letter_gap_em"] = round(float(np.median(gp)) / fpx, 3)
            style["case"] = case_of(tr["text"])
            style["box_height_px"] = round(float(np.median([o["h"] for o in obs])), 1)
            # changements de couleur au fil du temps
            seq = [(o["frame"], o["m"]["color"]) for o in obs if o.get("m")]
            changes, ref = [], seq[0][1]
            for fi, c in seq[1:]:
                if hexdist(ref, c) > 60:
                    changes.append({"frame": fi, "t": round(fi / fps, 3), "from": ref, "to": c}); ref = c
            if changes:
                style["color_changes"] = changes[:12]
        offs = sum(1 for r in rows if r[2] - r[4] / 2 < -2 or r[2] + r[4] / 2 > W + 2 or r[3] - r[5] / 2 < -2 or r[3] + r[5] / 2 > H + 2) / len(rows)
        dur_s = (rows[-1][0] - rows[0][0] + 1) / fps
        nt = len(norm_text(tr["text"]))
        quality = "fiable" if (tr["conf"] >= 0.9 and dur_s >= 0.3 and nt >= 3) else ("moyen" if (tr["conf"] >= 0.8 and dur_s >= 0.2 and nt >= 3) else "fragment")
        rel = style.get("font_size_pct_of_height", 0)
        role = "overlay_fixe" if static and len(rows) >= 25 else ("titre" if rel >= 4.5 else ("sous-titre" if rel >= 2.5 else "texte_secondaire"))
        shot_ids = sorted({next((s["index"] for s in shots if s["start"] <= r[0] <= s["end"]), 0) for r in rows}) if shots else []
        tier = "fragment" if quality == "fragment" else ("overlay" if role == "overlay_fixe" else ("principal" if role in ("titre", "sous-titre") and dur_s >= 0.4 else "secondaire"))
        out_el.append({"id": f"t{k + 1:03d}", "text": tr["text"], "text_segmented": bool(tr.get("segmented")),
                       "text_before_segmentation": tr["text_unsegmented"] if tr.get("segmented") else None, "tier": tier,
                       "text_variants": tr["variants"],
                       "ocr": {"rapidocr": tr.get("rapid_text"), "tesseract": tr.get("tess")},
                       "confidence": round(tr["conf"], 3), "quality": quality, "role": role, "static": static, "partly_offscreen": bool(offs > 0.2),
                       "shots": shot_ids, "start_frame": rows[0][0], "end_frame": rows[-1][0],
                       "start_t": rows[0][1], "end_t": rows[-1][1], "duration_s": round(rows[-1][1] - rows[0][1] + 1 / fps, 3),
                       "trajectory": traj, "style": style})

    # --- revue visuelle (annotations manuelles explicites, jamais mélangées aux mesures)
    if a.review and os.path.isfile(a.review):
        for rv in json.load(open(a.review)):
            for e in out_el:
                if norm_text(e["text"]) == norm_text(rv["text"]) and abs(e["start_t"] - rv.get("start_t", e["start_t"])) <= 0.2:
                    e["visual_review"] = dict(rv["set"], source=rv.get("source", "revue visuelle manuelle"))

    # --- blocs de texte (lignes d'une même expression) et expressions
    n_el = len(out_el)
    parent = list(range(n_el))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def med(e, key):
        return float(np.median([r[key] for r in e["trajectory"]["per_frame"]]))
    for p in range(n_el):
        for q in range(p + 1, n_el):
            A, B = out_el[p], out_el[q]
            if A["static"] or B["static"] or A["quality"] == "fragment" or B["quality"] == "fragment":
                continue
            if not set(A["shots"]) & set(B["shots"]) and shots:
                continue
            # lignes d'une même expression : elles apparaissent (et disparaissent) ensemble
            if abs(A["start_t"] - B["start_t"]) > 0.15 or abs(A["end_t"] - B["end_t"]) > 0.3:
                continue
            ra = {r[0]: r for r in A["trajectory"]["per_frame"]}
            rb = {r[0]: r for r in B["trajectory"]["per_frame"]}
            common = sorted(set(ra) & set(rb))
            if not common:
                continue
            g = common[len(common) // 2]
            xa, ya, wa, ha = ra[g][2:6]; xb, yb, wb, hb = rb[g][2:6]
            vgap = abs(ya - yb) - (ha + hb) / 2
            hov = min(xa + wa / 2, xb + wb / 2) - max(xa - wa / 2, xb - wb / 2)
            if vgap < 0.9 * max(ha, hb) and hov > 0.2 * min(wa, wb) and max(ha, hb) / max(min(ha, hb), 1e-6) <= 3.2:
                parent[find(p)] = find(q)
    groups = defaultdict(list)
    for p in range(n_el):
        if out_el[p]["quality"] != "fragment":
            groups[find(p)].append(p)
    expressions = []
    for gi, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: min(out_el[m]["start_frame"] for m in kv[1]))):
        els = [out_el[m] for m in members]
        mid = {e["id"]: (med(e, 2), med(e, 3), med(e, 5), med(e, 4)) for e in els}
        hh = np.median([v[2] for v in mid.values()])
        order = sorted(els, key=lambda e: (round(mid[e["id"]][1] / (0.6 * hh)), mid[e["id"]][0]))
        lefts = [mid[e["id"]][0] - mid[e["id"]][3] / 2 for e in order]
        cents = [mid[e["id"]][0] for e in order]
        rights = [mid[e["id"]][0] + mid[e["id"]][3] / 2 for e in order]
        if len(order) == 1:
            align = "seul"
        elif np.std(lefts) < 6:
            align = "gauche"
        elif np.std(cents) < 6:
            align = "centré"
        elif np.std(rights) < 6:
            align = "droite"
        else:
            align = "décalé (lignes à positions libres)"
        for e in els:
            e["block"] = f"b{gi + 1:02d}"
        for e in els:
            e["expression"] = f"b{gi + 1:02d}"
        expressions.append({"id": f"b{gi + 1:02d}", "text": " ".join(e["text"] for e in order), "lines": [e["id"] for e in order],
                            "alignment": align, "tier": ("principal" if any(e["tier"] == "principal" for e in els) else ("overlay" if all(e["tier"] == "overlay" for e in els) else "secondaire")), "shots": sorted({s for e in els for s in e["shots"]}),
                            "start_frame": min(e["start_frame"] for e in els), "end_frame": max(e["end_frame"] for e in els),
                            "start_t": round(min(e["start_t"] for e in els), 3), "end_t": round(max(e["end_t"] for e in els), 3)})

    result = {"schema": SCHEMA,
              "video": {"source": os.path.basename(a.video), "fps": round(fps, 3), "width": W, "height": H, "n_frames": N},
              "method": {"ocr": "rapidocr-onnxruntime (PP-OCRv4 embarqué), image par image",
                         "text_refinement": "tesseract fra+eng sur la meilleure image de chaque élément ; adopté si similaire à ≥ 0.8",
                         "style": "mesuré sur les pixels (Otsu dans la boîte du texte) ; polices NON identifiées",
                         "units": "pixels de la vidéo d'origine ; angle en degrés (sens horaire positif) ; t en secondes ; opacité = contraste texte/fond rapporté au maximum de l'élément",
                         "caveats": ["visual_review = annotations manuelles faites en regardant les images, distinctes des mesures automatiques", "taille de police estimée à ±15 % (proportions typiques des glyphes)",
                                     "classe de police = heuristique sur le contraste des traits, pas une identification",
                                     "l'opacité est un indicateur de contraste, pas une mesure directe de l'alpha",
                                     "un texte déformé, flou ou très rapide peut être mal lu ou manqué : voir confidence et text_variants"]},
              "shots": shots, "expressions": expressions, "elements": out_el,
              "stats": {"expressions_principales": sum(1 for x in expressions if x["tier"] == "principal"), "elements_reliable": sum(1 for e in out_el if e["quality"] == "fiable"), "elements_medium": sum(1 for e in out_el if e["quality"] == "moyen"), "fragments": sum(1 for e in out_el if e["quality"] == "fragment"), "elements": n_el, "expressions": len(expressions), "frames_with_text": sum(1 for d in cache["frames"] if d)}}
    json.dump(result, open(os.path.join(a.out, "text_tracking.json"), "w"), ensure_ascii=False, indent=1)

    # --- aperçu vidéo : boîtes et étiquettes
    cap = cv2.VideoCapture(a.video)
    raw = os.path.join(a.out, "_preview_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    palette = [(0, 160, 255), (0, 200, 0), (255, 120, 0), (200, 0, 200), (0, 200, 200), (60, 60, 255)]
    rowsby = defaultdict(list)
    for e in out_el:
        for r in e["trajectory"]["per_frame"]:
            rowsby[r[0]].append((e, r))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        for e, r in rowsby.get(i, []):
            c = palette[int(e["id"][1:]) % len(palette)]
            x, y, w, h = r[2], r[3], r[4], r[5]
            cv2.rectangle(f, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), c, 2)
            lab_t = f"{e['id']} {e['text'][:22]}"
            cv2.putText(f, lab_t, (int(x - w / 2), max(12, int(y - h / 2) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
        vw.write(f)
        i += 1
    vw.release()
    final = os.path.join(a.out, "text_preview.mp4")
    if shutil.which("ffmpeg") and subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                                                  "-pix_fmt", "yuv420p", "-movflags", "+faststart", final]).returncode == 0:
        os.remove(raw)
    else:
        os.replace(raw, final)
    print(f"{n_el} éléments de texte, {len(expressions)} expressions -> {a.out}/text_tracking.json")
    for ex in expressions:
        print(f"  {ex['start_t']:5.2f}-{ex['end_t']:5.2f}s  [{ex['alignment']}]  {ex['text']}")


if __name__ == "__main__":
    main()
