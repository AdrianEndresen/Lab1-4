from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, ListedColormap


# ---------- Filer ----------

mappe = Path(__file__).resolve().parent

excel_fil = mappe / "Vedlegg A – Måledata fra temperaturmålingen.xlsx"
bilde_fil = Path(
    r"C:\Users\adria\Downloads\Quick Share\20261011_162146.jpg"
)

data = pd.read_excel(excel_fil)

# Kolonne 2: servopuls. Kolonne 4: temperatur.
puls = pd.to_numeric(data.iloc[:, 1], errors="coerce")
temperatur = pd.to_numeric(data.iloc[:, 3], errors="coerce")

gyldig = puls.notna() & temperatur.notna()
puls = puls[gyldig].to_numpy()
temperatur = temperatur[gyldig].to_numpy()

if len(puls) == 0:
    raise ValueError("Fant ingen gyldige målinger i Excel-filen.")


# ---------- Første sveip og posisjoner ----------

slutt = int(np.argmax(puls)) + 1
puls_sveip = puls[:slutt]
temp = temperatur[:slutt]

# Anslått utslag: 1300 µs = 25°, 1650 µs = 0°
vinkel = np.radians((1650 - puls_sveip) / 350 * 25)

# Avstand fra servoaksen til følerspissen: 5 cm
x = 5 * np.sin(vinkel)
y = 5 * np.cos(vinkel)

# Antatt flammesentrum ved servopuls 1600 µs
sentrum_vinkel = np.radians((1650 - 1600) / 350 * 25)
sentrum_x = 5 * np.sin(sentrum_vinkel)
sentrum_y = 5 * np.cos(sentrum_vinkel)

avstand = np.hypot(x - sentrum_x, y - sentrum_y)

# Modellen bygger på målingene inn mot sentrum
inn_mot_sentrum = puls_sveip <= 1600

r = avstand[inn_mot_sentrum]
t = temp[inn_mot_sentrum]

rekkefolge = np.argsort(r)
r = r[rekkefolge]
t = t[rekkefolge]

# Slå sammen eventuelle gjentatte avstander
r_unik, grupper = np.unique(r, return_inverse=True)
t = np.bincount(grupper, weights=t) / np.bincount(grupper)
r = r_unik


# ---------- Temperaturmodell ----------

# Originalmålingene i temp endres ikke
t_modell = t.copy()

# Jevn ut lokale svingninger i den ytre delen under 200 °C.
# Modellen antar at temperaturen ikke stiger utover.
over_200 = np.flatnonzero(t >= 200)
start_lav = int(over_200[-1] + 1) if len(over_200) else 0

blokker = []

for indeks in range(start_lav, len(t)):
    blokker.append(([indeks], float(t[indeks])))

    while len(blokker) >= 2 and blokker[-2][1] < blokker[-1][1]:
        indeks_b, verdi_b = blokker.pop()
        indeks_a, verdi_a = blokker.pop()

        samlet = indeks_a + indeks_b
        gjennomsnitt = (
            verdi_a * len(indeks_a)
            + verdi_b * len(indeks_b)
        ) / len(samlet)

        blokker.append((samlet, gjennomsnitt))

for indeksgruppe, verdi in blokker:
    t_modell[indeksgruppe] = verdi

radius = float(r.max())

akse = np.linspace(-radius, radius, 500)
X, Y = np.meshgrid(akse, akse)
R = np.hypot(X, Y)

# Rotasjonssymmetrisk modell
T = np.interp(R, r, t_modell)
T = np.ma.masked_where(R > radius, T)


# ---------- Fargeskala ----------

# Samme normalisering som tidligere
temperatur_norm = PowerNorm(
    gamma=0.5,
    vmin=float(temp.min()),
    vmax=float(temp.max()),
    clip=True
)

# Juster bare de mørke nyansene under 200 °C
posisjon = np.linspace(0, 1, 1024)
f200 = float(temperatur_norm(200))

ny_posisjon = posisjon.copy()
lav = posisjon < f200

ny_posisjon[lav] = (
    f200 * (posisjon[lav] / f200)**0.7
)

inferno_justert = ListedColormap(
    plt.get_cmap("inferno")(ny_posisjon)
)
inferno_justert.set_bad((0, 0, 0, 0))

skala_merker = [
    verdi for verdi in [20, 30, 50, 100, 200, 400, 600, 800, 1000]
    if temp.min() <= verdi <= temp.max()
]


# ---------- Velg punkter i fotografiet ----------

bilde = plt.imread(bilde_fil)

fig_bilde, ax_bilde = plt.subplots(figsize=(10, 5))
ax_bilde.imshow(bilde)
ax_bilde.set_title(
    "Klikk i rekkefølge:\n"
    "1. Servoaksen   2. Følerspissen ved 1650 µs"
    "   3. Lysets sentrum"
)
ax_bilde.axis("off")
fig_bilde.tight_layout()

plt.figure(fig_bilde.number)
punkter = np.asarray(plt.ginput(3, timeout=0))

if len(punkter) != 3:
    raise RuntimeError("Du må velge tre punkter på fotografiet.")

servo, spiss, sentrum = punkter

retning = spiss - servo
lengde_px = np.linalg.norm(retning)

if lengde_px == 0:
    raise ValueError("Servoaksen og følerspissen må være ulike punkter.")

piksler_per_cm = lengde_px / 5.0

ey = retning / lengde_px
ex = np.array([-ey[1], ey[0]])

radius_px = radius * piksler_per_cm


# ---------- Temperaturkart over fotografiet ----------

kart_bilde = ax_bilde.imshow(
    T,
    extent=[
        sentrum[0] - radius_px,
        sentrum[0] + radius_px,
        sentrum[1] + radius_px,
        sentrum[1] - radius_px
    ],
    origin="upper",
    cmap=inferno_justert,
    norm=temperatur_norm,
    interpolation="bilinear",
    alpha=0.7,
    zorder=2
)

# Målepunkter relativt til det valgte flammesentrumet
punkt_x = sentrum[0] + piksler_per_cm * (
    (x - sentrum_x) * ex[0]
    + (y - sentrum_y) * ey[0]
)
punkt_y = sentrum[1] + piksler_per_cm * (
    (x - sentrum_x) * ex[1]
    + (y - sentrum_y) * ey[1]
)

ax_bilde.plot(
    punkt_x,
    punkt_y,
    color="white",
    linewidth=1,
    alpha=0.7,
    zorder=3
)

# Prikkene viser de opprinnelige temperaturmålingene
ax_bilde.scatter(
    punkt_x,
    punkt_y,
    c=temp,
    cmap=inferno_justert,
    norm=temperatur_norm,
    s=40,
    edgecolors="white",
    linewidths=0.7,
    label="Målepunkter",
    zorder=4
)

ax_bilde.legend(loc="upper left")

fig_bilde.colorbar(
    kart_bilde,
    ax=ax_bilde,
    label="Estimert temperatur [°C]",
    ticks=skala_merker,
    shrink=0.8
)

ax_bilde.set_xlim(0, bilde.shape[1])
ax_bilde.set_ylim(bilde.shape[0], 0)
ax_bilde.set_title(
    "Estimert temperaturfordeling over måleoppsettet\n"
    "Rotasjonssymmetri og utjevning under 200 °C"
)
ax_bilde.axis("off")

fig_bilde.tight_layout()
fig_bilde.savefig(
    mappe / "Heatmap_over_maaleoppsett.png",
    dpi=300,
    bbox_inches="tight"
)


# ---------- Eget temperaturkart uten fotografi ----------

fig, ax = plt.subplots(figsize=(7, 6))

kart = ax.imshow(
    T,
    extent=[-radius, radius, -radius, radius],
    origin="lower",
    cmap=inferno_justert,
    norm=temperatur_norm,
    interpolation="bilinear"
)

ax.scatter(
    x - sentrum_x,
    y - sentrum_y,
    c=temp,
    cmap=inferno_justert,
    norm=temperatur_norm,
    s=35,
    edgecolors="white",
    linewidths=0.7,
    label="Målepunkter"
)

fig.colorbar(
    kart,
    ax=ax,
    label="Temperatur [°C]",
    ticks=skala_merker
)

ax.set_xlabel("x fra antatt flammesentrum [cm]")
ax.set_ylabel("y fra antatt flammesentrum [cm]")
ax.set_title(
    "Estimert temperaturfordeling sett ovenfra\n"
    "Rotasjonssymmetri og utjevning under 200 °C"
)
ax.set_aspect("equal")
ax.legend(loc="upper right")

fig.tight_layout()
plt.show()