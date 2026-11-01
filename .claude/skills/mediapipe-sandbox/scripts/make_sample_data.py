#!/usr/bin/env python3
"""Génère des données d'exemple locales (aucun réseau) : une image et une courte vidéo synthétiques.
Elles ne contiennent pas de personne : elles servent à vérifier la chaîne de traitement, pas la qualité de détection.
Usage : python make_sample_data.py [DOSSIER]   (défaut : /tmp/mp-sample)
"""
import os, sys
import cv2, numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mp-sample"
os.makedirs(out, exist_ok=True)
img = np.full((360, 480, 3), (180, 200, 220), np.uint8)
cv2.circle(img, (240, 150), 60, (120, 160, 210), -1)
cv2.rectangle(img, (180, 210), (300, 340), (60, 90, 140), -1)
cv2.imwrite(os.path.join(out, "sample.png"), img)
vw = cv2.VideoWriter(os.path.join(out, "sample.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 360))
for i in range(12):
    vw.write(np.roll(img, i * 5, axis=1))
vw.release()
print("données d'exemple ->", out)
