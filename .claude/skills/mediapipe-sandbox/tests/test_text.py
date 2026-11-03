#!/usr/bin/env python3
"""Test du suivi de texte (voie OCR). Lancer avec /tmp/mp-ocr/bin/python tests/test_text.py
Vidéo synthétique : « HELLO WORLD » noir sur fond blanc qui glisse vers la droite. Vérifie lecture, trajectoire et style."""
import json, os, subprocess, sys, tempfile
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp(prefix="mp-text-test-")
vid = os.path.join(tmp, "t.mp4")
vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 240))
for i in range(15):
    f = np.full((240, 480, 3), 255, np.uint8)
    cv2.putText(f, "HELLO WORLD", (30 + 8 * i, 130), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)
    vw.write(f)
vw.release()
out = os.path.join(tmp, "out")
p = subprocess.run([sys.executable, os.path.join(HERE, "..", "scripts", "text_track.py"), vid, "--out", out], capture_output=True, text=True)
if p.returncode != 0:
    print(p.stdout, p.stderr[-1500:])
    sys.exit("ÉCHEC : text_track.py")
d = json.load(open(os.path.join(out, "text_tracking.json")))
els = [e for e in d["elements"] if "HELLO" in e["text"].upper() or "WORLD" in e["text"].upper()]
assert els, f"texte non lu : {[e['text'] for e in d['elements']]}"
e = max(els, key=lambda e: e["duration_s"])
xs = [r[2] for r in e["trajectory"]["per_frame"]]
assert len(xs) >= 12, f"piste trop courte : {len(xs)} images"
assert xs[-1] - xs[0] > 60, f"déplacement attendu ~112 px, obtenu {xs[-1] - xs[0]:.0f}"
assert all(b >= a - 1 for a, b in zip(xs, xs[1:])), "trajectoire non monotone"
s = e["style"]
rgb = lambda h: [int(h[i:i + 2], 16) for i in (1, 3, 5)]
assert max(rgb(s["color"])) <= 48, f"texte attendu proche du noir, obtenu {s['color']}"
assert min(rgb(s["background"])) >= 224, f"fond attendu proche du blanc, obtenu {s['background']}"
assert s["case"] == "MAJUSCULES", s["case"]
assert os.path.isfile(os.path.join(out, "text_preview.mp4"))
print(f"ok suivi de texte : {e['text']!r}, {len(xs)} images, déplacement {xs[-1] - xs[0]:.0f} px, couleur {s['color']}")
