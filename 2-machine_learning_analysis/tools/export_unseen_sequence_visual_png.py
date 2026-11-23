from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "1-data" / "splits" / "unseen" / "unseen_sequence_split_visual.png"
W, H = 1600, 900


def font(size, bold=False):
    base = "/usr/share/fonts/truetype/dejavu"
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"{base}/{name}", size)


def centered(draw, xy, text, fnt, fill):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def round_rect(draw, xy, fill, outline, width=2, radius=10):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def peptide(draw, x, y, text, test=False, small=False):
    w, h = (74, 46) if not small else (64, 40)
    fill = "#ffd9ca" if test else "#cfeec2"
    outline = "#e65f2a" if test else "#2f8b3a"
    round_rect(draw, (x, y, x + w, y + h), fill, outline, 2, 6)
    centered(draw, (x + w / 2, y + h / 2), text, font(18 if small else 19, True), "#122033")


def protein(draw, cx, cy, text, test=False, small=False):
    rx, ry = (38, 50) if not small else (33, 43)
    fill = "#ffe9a8" if test else "#cfe4ff"
    outline = "#d89400" if test else "#2766b1"
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fill, outline=outline, width=2)
    centered(draw, (cx, cy), text, font(15 if small else 17, True), "#122033")


def line(draw, a, b):
    draw.line((a, b), fill="#1b1f27", width=4)


def panel_header(draw, x, y, color, title):
    round_rect(draw, (x, y, x + 500, y + 650), "#ffffff", color, 3, 12)
    draw.rounded_rectangle((x, y, x + 500, y + 72), radius=12, fill=color)
    centered(draw, (x + 250, y + 38), title, font(31, True), "#ffffff")


img = Image.new("RGB", (W, H), "#fbfcff")
d = ImageDraw.Draw(img)

centered(d, (800, 46), "Visualisasi Skenario Unseen Sequence Split", font(45, True), "#10264f")

# Panel 1: unseen peptide
panel_header(d, 32, 104, "#2d8a3f", "1. unseen_peptide")
round_rect(d, (64, 206, 279, 586), "#f8fff6", "#85bf89")
round_rect(d, (308, 206, 508, 586), "#fff9f6", "#85bf89")
d.line((294, 196, 294, 596), fill="#a7adb8", width=3)
centered(d, (172, 235), "TRAIN", font(25, True), "#287735")
centered(d, (408, 235), "TEST", font(25, True), "#287735")
for yy, txt in [(300, "PEP A"), (400, "PEP B"), (500, "PEP C")]:
    peptide(d, 84, yy, txt)
for cy, txt in [(310, "PROT X"), (420, "PROT Y"), (520, "PROT Z")]:
    protein(d, 222, cy, txt)
for a, b in [((158, 323), (184, 310)), ((158, 323), (184, 420)), ((158, 423), (184, 420)), ((158, 523), (184, 520))]:
    line(d, a, b)
peptide(d, 326, 318, "PEP D", True)
peptide(d, 326, 448, "PEP E", True)
for cy, txt in [(310, "PROT X"), (420, "PROT Y"), (520, "PROT Z")]:
    protein(d, 468, cy, txt)
for a, b in [((400, 341), (430, 310)), ((400, 341), (430, 420)), ((400, 471), (430, 520))]:
    line(d, a, b)
round_rect(d, (64, 616, 532, 698), "#f2fbef", "#62a968")
centered(d, (96, 657), "OK", font(30, True), "#2d8a3f")
d.text((132, 638), "Peptide test tidak muncul di train;", font=font(22, True), fill="#171a21")
d.text((132, 668), "protein boleh overlap.", font=font(22, True), fill="#171a21")

# Panel 2: unseen protein
panel_header(d, 550, 104, "#2766b1", "2. unseen_protein")
round_rect(d, (582, 206, 797, 586), "#f7fbff", "#89b5e8")
round_rect(d, (826, 206, 1026, 586), "#fffdf5", "#89b5e8")
d.line((812, 196, 812, 596), fill="#a7adb8", width=3)
centered(d, (690, 235), "TRAIN", font(25, True), "#1f5799")
centered(d, (926, 235), "TEST", font(25, True), "#1f5799")
for yy, txt in [(300, "PEP A"), (400, "PEP B"), (500, "PEP C")]:
    peptide(d, 602, yy, txt)
for cy, txt in [(310, "PROT X"), (420, "PROT Y"), (520, "PROT Z")]:
    protein(d, 750, cy, txt)
for a, b in [((676, 323), (712, 310)), ((676, 423), (712, 420)), ((676, 523), (712, 520))]:
    line(d, a, b)
peptide(d, 844, 318, "PEP A")
peptide(d, 844, 448, "PEP B")
for cy, txt in [(310, "PROT W"), (420, "PROT Q"), (520, "PROT R")]:
    protein(d, 986, cy, txt, True)
for a, b in [((918, 341), (948, 310)), ((918, 471), (948, 420))]:
    line(d, a, b)
round_rect(d, (582, 616, 1050, 698), "#f1f7ff", "#4a89d6")
centered(d, (614, 657), "OK", font(30, True), "#2766b1")
d.text((650, 638), "Protein test tidak muncul di train;", font=font(22, True), fill="#171a21")
d.text((650, 668), "peptide boleh overlap.", font=font(22, True), fill="#171a21")

# Panel 3: unseen both
panel_header(d, 1068, 104, "#6740a0", "3. unseen_both")
round_rect(d, (1098, 206, 1268, 586), "#fcf9ff", "#b49ad5")
round_rect(d, (1292, 206, 1442, 586), "#fcf9ff", "#b49ad5")
round_rect(d, (1460, 206, 1550, 586), "#fcf9ff", "#b49ad5")
d.line((1280, 196, 1280, 596), fill="#a7adb8", width=3)
d.line((1450, 196, 1450, 596), fill="#a7adb8", width=3)
centered(d, (1183, 235), "TRAIN", font(24, True), "#563483")
centered(d, (1367, 235), "TEST", font(24, True), "#563483")
centered(d, (1505, 235), "BUFFER", font(17, True), "#563483")
for yy, txt in [(300, "PEP A"), (400, "PEP B"), (500, "PEP C")]:
    peptide(d, 1114, yy, txt, small=True)
for cy, txt in [(310, "PROT X"), (420, "PROT Y"), (520, "PROT Z")]:
    protein(d, 1232, cy, txt, small=True)
for a, b in [((1178, 320), (1199, 310)), ((1178, 320), (1199, 420)), ((1178, 420), (1199, 420)), ((1178, 520), (1199, 520))]:
    line(d, a, b)
peptide(d, 1308, 336, "PEP D", True, True)
peptide(d, 1308, 462, "PEP E", True, True)
protein(d, 1410, 350, "PROT W", True, True)
protein(d, 1410, 478, "PROT Q", True, True)
for a, b in [((1372, 356), (1378, 350)), ((1372, 482), (1378, 478)), ((1372, 482), (1378, 350))]:
    line(d, a, b)
peptide(d, 1470, 300, "PEP A", small=True)
protein(d, 1536, 317, "PROT W", True, True)
line(d, (1522, 317), (1512, 317))
peptide(d, 1470, 392, "PEP D", True, True)
protein(d, 1536, 409, "PROT X", small=True)
line(d, (1522, 409), (1512, 409))
peptide(d, 1470, 484, "PEP B", small=True)
protein(d, 1536, 501, "PROT Q", True, True)
line(d, (1522, 501), (1512, 501))
round_rect(d, (1098, 616, 1568, 708), "#f7f1ff", "#7952aa")
centered(d, (1132, 662), "OK", font(30, True), "#6740a0")
d.text((1170, 638), "Train dan test bersih dari overlap", font=font(22, True), fill="#171a21")
d.text((1170, 668), "peptide maupun protein; campuran buffer.", font=font(22, True), fill="#171a21")

round_rect(d, (160, 790, 1440, 872), "#fff8df", "#e1a100", 3, 14)
centered(d, (220, 831), "GOAL", font(27, True), "#d96b00")
d.text(
    (292, 814),
    "Tujuan: menguji generalisasi ke sekuens baru, bukan menghafal pasangan yang sudah pernah dilihat.",
    font=font(26, True),
    fill="#161b26",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT)
print(OUT)
