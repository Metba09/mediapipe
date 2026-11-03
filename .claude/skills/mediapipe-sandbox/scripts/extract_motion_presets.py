#!/usr/bin/env python3
"""Extrait d'un text_tracking.json (text_track.py) les PRESETS DE MOUVEMENT réutilisables en animation (GSAP...).

Pour chaque mouvement mesuré on ajuste la courbe d'accélération GSAP la plus proche (erreur quadratique moyenne
sur la progression normalisée) et on conserve distance, durée et preuves (ids d'éléments, erreur d'ajustement).
Les éléments sont désignés par leur id dans CE JSON (role map ci-dessous) ; un garde-fou vérifie leur texte.

Usage : python3 extract_motion_presets.py text_tracking.json [--out motion_presets.json]
"""
import argparse, json, math, sys
import numpy as np

# rôle -> (id d'élément, texte attendu). Spécifique à la vidéo Vidtype.mp4 du 2026-11-02.
HERO = {"sans": ("t002", "most"), "didoneI": ("t003", "people"), "serif": ("t004", "think"),
        "sansB": ("t005", "branding"), "sansS": ("t006", "start with")}
OTHER = {"cover_logo": ("t009", "VOUE"), "drop": ("t010", "DESIGN"), "flicker_glitch": ("t046", "COLOURS"),
         "slide": ("t053", "shinny"), "rise": ("t070", "but that’s not"), "zoom": ("t138", "miracle studios")}

OUT_EASES = {
    "none": lambda p: p,
    "sine.out": lambda p: np.sin(p * math.pi / 2),
    "power1.out": lambda p: 1 - (1 - p) ** 2,
    "power2.out": lambda p: 1 - (1 - p) ** 3,
    "power3.out": lambda p: 1 - (1 - p) ** 4,
    "power4.out": lambda p: 1 - (1 - p) ** 5,
    "expo.out": lambda p: np.where(p >= 1, 1.0, 1.001 * (1 - 2 ** (-10 * p))),
    "circ.out": lambda p: np.sqrt(1 - (1 - p) ** 2),
}
IN_EASES = {
    "power1.in": lambda p: p ** 2, "power2.in": lambda p: p ** 3, "power3.in": lambda p: p ** 4,
    "power4.in": lambda p: p ** 5, "expo.in": lambda p: np.where(p <= 0, 0.0, 2 ** (10 * (p - 1))),
    "sine.in": lambda p: 1 - np.cos(p * math.pi / 2),
}


def fit(t, v, v_target, eases, dmin=0.04, dmax=2.0):
    """Meilleure (courbe, durée) pour passer de v[0] à v_target ; erreur sur la progression normalisée."""
    v0 = v[0]
    if abs(v_target - v0) < 1e-6:
        return None
    p = np.clip((v - v0) / (v_target - v0), -0.2, 1.2)
    best = None
    for name, fn in eases.items():
        for D in np.arange(dmin, dmax, 0.02):
            q = fn(np.clip((t - t[0]) / D, 0, 1))
            err = float(np.mean((q - p) ** 2))
            if best is None or err < best[0]:
                best = (err, name, round(float(D), 2))
    return {"ease": best[1], "duration": best[2], "fit_error": round(best[0], 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--out", default="motion_presets.json")
    a = ap.parse_args()
    d = json.load(open(a.json))
    fps, W, H = d["video"]["fps"], d["video"]["width"], d["video"]["height"]
    E = {e["id"]: e for e in d["elements"]}

    def el(key, table):
        i, txt = table[key]
        e = E[i]
        if e["text"].replace("'", "’") != txt.replace("'", "’"):
            sys.exit(f"garde-fou : {i} vaut {e['text']!r}, attendu {txt!r} (ce JSON n'est pas celui de Vidtype.mp4)")
        return e

    def ser(e, col, only_observed=True):
        rows = [r for r in e["trajectory"]["per_frame"] if not (only_observed and r[10])]
        return np.array([r[0] for r in rows]) / fps, np.array([r[col] for r in rows], float)

    out = {"schema": "motion-presets/1", "source": d["video"]["source"], "fps": round(fps, 3), "canvas_src": {"w": W, "h": H}}

    # --- plans : durées et rythme
    out["shots"] = [{"index": s["index"], "start": round(s["start_t"], 3), "duration": round(s["end_t"] - s["start_t"], 3), "cut": "hard"} for s in d["shots"]]

    # --- entrées glissées du titre (plan 0) : décalage de départ, courbe, durée, retard entre lignes
    hero, starts = {}, []
    for role, (i, _) in HERO.items():
        e = el(role, HERO)
        t, x = ser(e, 2)
        rest = float(np.median(x[(t > 1.0) & (t < 1.3)]))
        m = t <= 1.0
        f = fit(t[m], x[m], rest, OUT_EASES)
        _, h = ser(e, 5)
        _, op = ser(e, 8)
        op = np.array([o for o in op if o == o and o is not None], float)
        first = e["trajectory"]["per_frame"][0]
        hero[role] = {"element": i, "text": e["text"], "from_x_frac": round((x[0] - rest) / W, 3), "ease": f["ease"], "duration": f["duration"], "fit_error": f["fit_error"],
                      "start_opacity": round(float(op[0]), 2), "height_start_over_peak": round(float(h[0] / max(h[:6])), 2),
                      "font_px_src": e["style"]["font_size_px_est"], "color": e["style"].get("color"), "start_t": e["start_t"]}
        starts.append(e["start_t"])
    st = sorted(starts)
    out["hero_entrance"] = {"lines": hero, "stagger_s": round(float(np.median(np.diff(st))) if len(set(st)) > 1 else 0.035, 3),
                            "note": "les lignes glissent de la droite vers le repos ; l'écart entre lignes mesuré est de l'ordre d'une image"}

    # --- sortie du titre : chute accélérée puis coupe franche
    drops = []
    for role in ("sans", "didoneI", "sansB"):
        e = el(role, HERO)
        t, y = ser(e, 2 + 1)
        _, x = ser(e, 2)
        m = t >= 1.35
        tt, yy, xx = t[m], y[m], x[m]
        f = fit(tt, yy, float(yy[-1]), IN_EASES, 0.1, 1.0)
        drops.append({"dy_frac": (yy[-1] - yy[0]) / H, "dx_frac": (xx[-1] - xx[0]) / W, "dur": float(tt[-1] - tt[0]), "ease": f["ease"], "fit_error": f["fit_error"]})
    out["exit_fall"] = {"dy_frac": round(float(np.median([x["dy_frac"] for x in drops])), 3), "dx_frac": round(float(np.median([x["dx_frac"] for x in drops])), 3),
                        "duration": round(float(np.median([x["dur"] for x in drops])), 2), "ease": drops[0]["ease"],
                        "fit_errors": [x["fit_error"] for x in drops], "then": "coupe franche (le plan suivant apparaît au même instant)"}

    # --- autres mouvements nommés
    def one(key, col, v_target=None, tmax=None, tmin=None, eases=OUT_EASES, scale=1.0):
        e = el(key, OTHER)
        t, v = ser(e, col)
        if tmin is not None:
            m = t >= tmin; t, v = t[m], v[m]
        if tmax is not None:
            m = t <= tmax; t, v = t[m], v[m]
        tgt = v[-1] if v_target is None else v_target
        f = fit(t, v, tgt, eases)
        return e, v, f

    e, v, f = one("cover_logo", 2, tmax=e_t(E["t009"], fps, 5))
    snap = {"from_x_frac": round(float((v[0] - v[-1]) / W), 3), "duration": 0.04}
    e, v, f = one("cover_logo", 3, tmin=E["t009"]["start_t"] + 2 / fps)
    out["cover_logo"] = {"element": "t009", "snap_in": snap, "settle": {"dy_frac": round(float((v[-1] - v[0]) / H), 3), **f}, "font_px_src": E["t009"]["style"]["font_size_px_est"]}
    e, v, f = one("drop", 3)
    out["drop"] = {"element": "t010", "dy_frac": round(float((v[-1] - v[0]) / H), 3), **f, "font_px_src": e["style"]["font_size_px_est"]}
    e, v, f = one("slide", 2)
    out["slide"] = {"element": "t053", "from_x_frac": round(float((v[0] - v[-1]) / W), 3), **f, "font_px_src": e["style"]["font_size_px_est"]}
    e, v, f = one("rise", 3)
    out["rise"] = {"element": "t070", "from_y_frac": round(float((v[0] - v[-1]) / H), 3), **f, "font_px_src": e["style"]["font_size_px_est"]}
    e, v, f = one("zoom", 5)
    e2, v2, f2 = one("zoom", 3)
    out["zoom"] = {"element": "t138", "scale_from": round(float(v[0] / v[-1]), 3), "scale": f, "dy_frac": round(float((v2[-1] - v2[0]) / H), 3), "drift": f2, "font_px_src": e["style"]["font_size_px_est"]}

    # --- papillotement d'opacité (sigma par image) et fondu d'entrée
    def sigma(i):
        o = [r[8] for r in E[i]["trajectory"]["per_frame"] if r[8] is not None]
        return round(float(np.std(o)), 3)
    out["flicker"] = {"hero_sigma": round(float(np.mean([sigma(i) for i, _ in HERO.values()])), 3), "glitch_sigma": round(float(np.mean([sigma("t010"), sigma("t046")])), 3),
                      "note": "opacité multipliée par (1 - sigma × bruit) à chaque image"}
    out["fade_in_s"] = round(2 / fps, 3)

    # --- palette et typographie mesurées
    reds = [E[i]["style"]["color"] for i in ("t003", "t005", "t090") if E[i]["style"].get("color")]
    rr = np.median([[int(c[k:k + 2], 16) for k in (1, 3, 5)] for c in reds], axis=0)
    out["palette"] = {"white": "#ffffff", "black": "#000000", "red": "#%02x%02x%02x" % tuple(int(v) for v in rr), "red_evidence": reds,
                      "dark_bg": E["t138"]["style"].get("background", "#141414"), "beige_on_dark": "#e3c8aa", "beige_evidence": "t067 (USAN)"}
    sizes = {"sans": "t002", "didoneI": "t003", "serif": "t004", "sansB": "t005", "sansS": "t006", "didoneB": "t009", "drop": "t010", "slide": "t053", "zoom": "t138"}
    base = E["t002"]["style"]["font_size_px_est"]
    out["type_scale"] = {k: {"font_px_src": E[i]["style"]["font_size_px_est"], "ratio_to_sans": round(E[i]["style"]["font_size_px_est"] / base, 2), "weight": E[i]["style"].get("weight_class"),
                             "italic": E[i]["style"].get("italic")} for k, i in sizes.items()}
    out["type_scale"]["note"] = "ratio_to_sans : contraste de taille entre petites lignes (≈ 1) et mots forts (≈ 2,5 à 3,8)"
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("presets ->", a.out)
    print(json.dumps({k: out[k] for k in ("hero_entrance", "exit_fall", "cover_logo", "drop", "slide", "rise", "zoom", "flicker", "palette")}, ensure_ascii=False)[:1800])


def e_t(e, fps, n):
    return e["start_t"] + n / fps


if __name__ == "__main__":
    main()
