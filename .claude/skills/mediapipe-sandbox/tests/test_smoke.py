#!/usr/bin/env python3
"""Test de fumée hors-ligne (voie legacy). Lancer avec /tmp/mp-legacy/bin/python tests/test_smoke.py
Vérifie : chargement des modèles embarqués, exécution des 6 solutions, fichiers de sortie. Aucun réseau."""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
tmp = tempfile.mkdtemp(prefix="mp-test-")


def run(*args):
    p = subprocess.run([sys.executable, *args], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout, p.stderr[-800:])
        sys.exit(f"ÉCHEC : {' '.join(args[:3])}")
    return p


run(os.path.join(SCRIPTS, "make_sample_data.py"), tmp)
img, vid = os.path.join(tmp, "sample.png"), os.path.join(tmp, "sample.mp4")
for sol in ("hands", "pose", "face_mesh", "face_detection", "holistic", "selfie_segmentation"):
    out = os.path.join(tmp, sol)
    run(os.path.join(SCRIPTS, "legacy_landmarks.py"), sol, img, "--out", out)
    for f in ("annotated.png", "landmarks.json"):
        assert os.path.isfile(os.path.join(out, f)), f"{sol}: {f} manquant"
    json.load(open(os.path.join(out, "landmarks.json")))
    print("ok", sol)
assert os.path.isfile(os.path.join(tmp, "selfie_segmentation", "mask.png"))
vout = os.path.join(tmp, "video")
run(os.path.join(SCRIPTS, "legacy_landmarks.py"), "pose", vid, "--out", vout, "--stride", "2")
frames = json.load(open(os.path.join(vout, "landmarks.json")))
assert len(frames) == 6, f"attendu 6 images, obtenu {len(frames)}"
print("ok vidéo (6 images)\nTOUS LES TESTS PASSENT")
