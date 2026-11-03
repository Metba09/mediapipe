#!/usr/bin/env python3
"""Anime un corpus texte (markdown léger) avec le design motion mesuré (motion_presets.json) -> un HTML5 GSAP autonome.

Sortie : une seule page (GSAP, polices WOFF2 et données en ligne), scène 1080x1920, timeline maîtresse en pause
déterministe + window.seek(t) (compatible rendu image par image). Garde-fou : tous les mots du corpus, dans l'ordre.

Usage : python3 build_gsap_corpus.py corpus.txt --presets motion_presets.json --out page.html \
          [--assets /tmp/mp-work/gsap-assets/node_modules] [--title "..."]
"""
import argparse, base64, json, math, re, sys
from pathlib import Path

MAXW = 10
CYCLE = ["hero", "rise", "hero", "slide"]
END = re.compile(r"[.!?…]$")


def parse(md):
    """-> liste de blocs {level, words:[{w,b,i}]} ; marqueurs markdown retirés, mots = séparés par espaces."""
    blocks = []
    for raw in md.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(#{1,6})\s+", line)
        level = len(m.group(1)) if m else 0
        if m:
            line = line[m.end():]
        chars, bold, ital, k = [], False, False, 0
        while k < len(line):
            if line.startswith("**", k):
                bold = not bold; k += 2
            elif line[k] == "*":
                ital = not ital; k += 1
            else:
                chars.append((line[k], bold or level > 0 and False, ital)); k += 1
        words, cur = [], []
        for c in chars + [(" ", False, False)]:
            if c[0].isspace():
                if cur:
                    txt = "".join(x[0] for x in cur)
                    f = next((x for x in cur if x[0].isalnum()), cur[0])
                    words.append({"w": txt, "b": f[1], "i": f[2], "alpha": any(x[0].isalnum() for x in cur)})
                    cur = []
            else:
                cur.append(c)
        # inertie des marqueurs sans lettres : "—" suit le mot suivant, ponctuation seule suit le précédent
        for n, w in enumerate(words):
            if not w["alpha"]:
                if w["w"] == "—" and n + 1 < len(words):
                    w["i"], w["b"] = words[n + 1]["i"], words[n + 1]["b"]
                elif n:
                    w["i"], w["b"] = words[n - 1]["i"], words[n - 1]["b"]
        if words:
            blocks.append({"level": level, "words": words})
    return blocks


def pieces(words):
    out, cur = [], []
    for w in words:
        cur.append(w)
        t = w["w"]
        if END.search(t.rstrip('"»”’')) or t.endswith(":") or t.endswith(";") or (t.endswith(",") and len(cur) >= 5):
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return out


def split_even(p):
    n = len(p)
    parts = math.ceil(n / MAXW)
    size = math.ceil(n / parts)
    return [p[k:k + size] for k in range(0, n, size)]


def chunk_block(words):
    runs, cur = [], []
    for w in words:
        if cur and cur[-1]["i"] != w["i"]:
            runs.append(cur); cur = []
        cur.append(w)
    runs.append(cur)
    chunks = []
    for run in runs:
        acc = []
        for p in pieces(run):
            for q in (split_even(p) if len(p) > MAXW else [p]):
                if acc and len(acc) + len(q) <= MAXW and not END.search(acc[-1]["w"].rstrip('"»”’')):
                    acc = acc + q
                else:
                    if acc:
                        chunks.append(acc)
                    acc = list(q)
        if acc:
            chunks.append(acc)
    return chunks


def hero_lines(ws):
    n = len(ws)
    if n <= 2:
        return [("emph", " ".join(w["w"] for w in ws))]
    bi = [k for k, w in enumerate(ws) if w["b"]]
    if bi and len(bi) <= 3 and len(bi) < n:
        a, b = bi[0], bi[-1] + 1
    else:
        a = max(range(n), key=lambda k: (len(re.sub(r"\W", "", ws[k]["w"])), -k)); b = a + 1
    L = []
    if a: L.append(("before", " ".join(w["w"] for w in ws[:a])))
    L.append(("emph", " ".join(w["w"] for w in ws[a:b])))
    if b < n: L.append(("after", " ".join(w["w"] for w in ws[b:])))
    return L


def build_beats(blocks):
    beats, cyc, section = [], 0, ""
    first_title = True
    for bl in blocks:
        lvl = bl["level"]
        chunks = chunk_block(bl["words"])
        txt0 = " ".join(w["w"] for w in bl["words"])
        if lvl == 3 and first_title:
            first_title = False
            section = ""
            beats.append({"tpl": "cover", "section": "", "words": bl["words"], "lines": [("emph", txt0)]}); continue
        if lvl in (3, 4) and len(bl["words"]) <= MAXW:
            section = txt0.rstrip(" :")
            beats.append({"tpl": "drop", "section": section, "words": bl["words"], "lines": [("emph", txt0)]}); continue
        for n, ch in enumerate(chunks):
            quote = sum(1 for w in ch if w["i"]) * 2 > len(ch)
            if lvl == 4 and n == 0 and not quote:
                tpl = "drop"
            elif quote:
                tpl = "zoom"
            else:
                tpl = CYCLE[cyc % len(CYCLE)]; cyc += 1
            if lvl == 4 and n == 0:
                section = " ".join(w["w"] for w in ch[:4]).rstrip(",.") + "…"
            beat = {"tpl": tpl, "section": section, "words": ch}
            beat["lines"] = hero_lines(ch) if tpl == "hero" else [("body", " ".join(w["w"] for w in ch))]
            beats.append(beat)
    # temps : cadence de lecture
    t = 0.0
    for b in beats:
        n = len(b["words"])
        b["t0"] = round(t, 3)
        b["dur"] = round({"cover": 3.4, "drop": 2.6}.get(b["tpl"], max(2.2, 1.2 + 0.30 * n)), 3)
        t += b["dur"]
    return beats, t


def check_coverage(md, beats):
    ref = re.sub(r"[#*]", "", md).split()
    got = [w["w"] for b in beats for w in b["words"]]
    if ref != got:
        k = next((k for k, (x, y) in enumerate(zip(ref, got)) if x != y), min(len(ref), len(got)))
        sys.exit(f"couverture : divergence au mot {k}: {ref[k:k+4]} vs {got[k:k+4]} ({len(ref)} vs {len(got)} mots)")
    return len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("--presets", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--assets", default="/tmp/mp-work/gsap-assets/node_modules"); ap.add_argument("--title", default="")
    a = ap.parse_args()
    md = Path(a.corpus).read_text(encoding="utf-8")
    presets = json.load(open(a.presets))
    blocks = parse(md)
    beats, total = build_beats(blocks)
    nwords = check_coverage(md, beats)
    assets = Path(a.assets)
    gsap = (assets / "gsap/dist/gsap.min.js").read_text(encoding="utf-8")
    def face(fam, pkg, wt, style):
        b = base64.b64encode((assets / f"@fontsource/{pkg}/files/{pkg}-latin-{wt}-{style}.woff2").read_bytes()).decode()
        return f"@font-face{{font-family:'{fam}';font-weight:{wt};font-style:{style};font-display:block;src:url(data:font/woff2;base64,{b}) format('woff2')}}"
    fonts = "".join([face("Arimo", "arimo", 400, "normal"), face("Arimo", "arimo", 500, "normal"), face("Arimo", "arimo", 700, "normal"),
                     face("Playfair", "playfair-display", 400, "normal"), face("Playfair", "playfair-display", 400, "italic"),
                     face("Tinos", "tinos", 400, "normal"), face("Tinos", "tinos", 400, "italic")])
    data = {"presets": presets, "beats": [{"tpl": b["tpl"], "section": b["section"], "t0": b["t0"], "dur": b["dur"], "lines": b["lines"], "n": len(b["words"])} for b in beats],
            "total": round(total, 3), "words": nwords}
    tpl = (Path(__file__).resolve().parent.parent / "templates/gsap_corpus.template.html").read_text(encoding="utf-8")
    title = a.title or "Odyssée de la Sagesse — motion"
    html = (tpl.replace("__TITLE__", title).replace("__FONTS__", fonts).replace("__GSAP__", gsap.replace("</script", "<\\/script"))
               .replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    Path(a.out).write_text(html, encoding="utf-8")
    from collections import Counter
    print(f"{a.out}: {len(beats)} plans, {nwords} mots, {total:.1f}s, {len(html)//1024} Ko ; modèles {dict(Counter(b['tpl'] for b in beats))}")


if __name__ == "__main__":
    main()
