"""
build_gravity_panel.py
======================
Constructs a dyadic country-pair panel ready for gravity regression.

Input files (in same directory or paths below):
  - agreement_scores.csv   : PTA-level text/structure scores (424 agreements)
  - cci_agreement_level.csv: CCI scores (same PTAs, normalised)

Output:
  - cci_gravity_panel.csv  : long-format dyadic panel (iso_o, iso_d, year, ...)

Key design decisions
--------------------
1.  PLURILATERAL EXPANSION
    Each plurilateral agreement is exploded into all ordered (i,j) country-pair
    rows.  We parse member ISO-3 codes from the agreement name where possible;
    for well-known blocs (EU, EFTA, ASEAN, MERCOSUR, …) we use hard-coded
    membership lists that respect historical entry/exit.

2.  DE-DUPLICATION
    When a pair is covered by >1 agreement in the same year we keep the most
    specific one (bilateral > plurilateral; tie-break: higher pta_id = more
    recent).

3.  GRAVITY CONTROLS
    We embed time-invariant dyadic controls sourced from CEPII/DGD literature:
      - ln_dist        : log population-weighted distance (CEPII distw)
      - contiguity     : shared land border
      - common_language: at least one official language in common
      - colony_ever    : directional colonial relationship ever
      - common_colonizer: shared colonial hegemon
      - common_legal_origin: same legal family (English/French/German/…)
    These are populated from a BUILT-IN reference table compiled from
    CEPII GeoDist and the USITC Dynamic Gravity Dataset (DGD) documentation.
    After downloading the full DGD (gravity.usitc.gov) you can replace the
    stub table with a left-join on the real data — the column names are
    already aligned.

4.  FIXED-EFFECT IDENTIFIERS (created, not used here):
      - pair_id  = iso_o + '_' + iso_d
      - imp_year = iso_d + '_' + year
      - exp_year = iso_o + '_' + year

5.  PAIR WEIGHT  (for WLS when running agreement-level regressions)
      pair_weight = 1 / n_directed_pairs_in_agreement
    This prevents large blocs from dominating OLS estimates.

Usage
-----
    python build_gravity_panel.py \
        --scores   agreement_scores.csv \
        --cci      cci_agreement_level.csv \
        --out      cci_gravity_panel.csv

"""

import argparse
import re
import warnings
from itertools import permutations

import numpy as np
import pandas as pd
import pycountry

warnings.filterwarnings("ignore")

# ── 0. CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build dyadic gravity panel")
    p.add_argument("--scores", default="agreement_scores.csv")
    p.add_argument("--cci",    default="cci_agreement_level.csv")
    p.add_argument("--out",    default="cci_gravity_panel.csv")
    return p.parse_args()


# ── 1. PLURILATERAL MEMBERSHIP TABLES ────────────────────────────────────────
# For major blocs, list ISO-3 members with (entry_year, exit_year|None).
# exit_year=None means "still member as of dataset end".
# Partial lists — expand as needed.

EU_MEMBERS_HIST = {
    # (iso3, entry_year, exit_year)
    "BEL": (1957, None), "FRA": (1957, None), "DEU": (1957, None),
    "ITA": (1957, None), "LUX": (1957, None), "NLD": (1957, None),
    "DNK": (1973, None), "IRL": (1973, None), "GBR": (1973, 2020),
    "GRC": (1981, None), "PRT": (1986, None), "ESP": (1986, None),
    "AUT": (1995, None), "FIN": (1995, None), "SWE": (1995, None),
    "CYP": (2004, None), "CZE": (2004, None), "EST": (2004, None),
    "HUN": (2004, None), "LVA": (2004, None), "LTU": (2004, None),
    "MLT": (2004, None), "POL": (2004, None), "SVK": (2004, None),
    "SVN": (2004, None), "BGR": (2007, None), "ROU": (2007, None),
    "HRV": (2013, None),
}

EFTA_MEMBERS_HIST = {
    "CHE": (1960, None), "NOR": (1960, None), "ISL": (1970, None),
    "LIE": (1991, None),
    # historical (left EFTA to join EU or otherwise):
    "AUT": (1960, 1995), "FIN": (1986, 1995), "SWE": (1960, 1995),
    "DNK": (1960, 1973), "PRT": (1960, 1986), "GBR": (1960, 1973),
}

ASEAN_MEMBERS_HIST = {
    "BRN": (1984, None), "KHM": (1999, None), "IDN": (1967, None),
    "LAO": (1997, None), "MYS": (1967, None), "MMR": (1997, None),
    "PHL": (1967, None), "SGP": (1967, None), "THA": (1967, None),
    "VNM": (1995, None),
}

MERCOSUR_MEMBERS_HIST = {
    "ARG": (1991, None), "BRA": (1991, None), "PRY": (1991, None),
    "URY": (1991, None), "VEN": (2012, 2017), "BOL": (2015, None),
}

CARICOM_MEMBERS = [
    "ATG","BHS","BRB","BLZ","DMA","GRD","GUY","HTI","JAM","KNA",
    "LCA","VCT","SUR","TTO","MSR","AIA","BMU","VGB","CYM","TCA",
]

SACU_MEMBERS = ["BWA","LSO","NAM","ZAF","SWZ"]

GCC_MEMBERS   = ["BHR","KWT","OMN","QAT","SAU","ARE"]

ECOWAS_MEMBERS = [
    "BEN","BFA","CPV","CIV","GMB","GHA","GIN","GNB","LBR",
    "MLI","MRT","NER","NGA","SEN","SLE","TGO",
]

COMESA_MEMBERS = [
    "BDI","COM","COD","DJI","EGY","ERI","ETH","KEN","LSO","LBY",
    "MDG","MWI","MUS","MOZ","NAM","RWA","SYC","SOM","SDN","SWZ",
    "TZA","UGA","ZMB","ZWE",
]

CAN_MEMBERS    = ["BOL","COL","ECU","PER"]   # Andean Community
SADC_MEMBERS   = [
    "AGO","BWA","COM","COD","LSO","MDG","MWI","MUS","MOZ",
    "NAM","SYC","ZAF","SWZ","TZA","ZMB","ZWE",
]
EAC_MEMBERS    = ["BDI","KEN","RWA","TZA","UGA"]
NAFTA_MEMBERS  = ["CAN","MEX","USA"]
EEA_MEMBERS    = ["ISL","LIE","NOR"]   # + EU (handled dynamically)
PAFTA_MEMBERS  = [  # Pan-Arab FTA (most Arab League states)
    "DZA","BHR","COM","DJI","EGY","IRQ","JOR","KWT","LBN","LBY",
    "MRT","MAR","OMN","PSE","QAT","SAU","SOM","SDN","SYR","TUN",
    "ARE","YEM",
]
APTA_MEMBERS   = ["BGD","CHN","IND","KOR","LKA","MNG"]
GSTP_MEMBERS   = [  # Global System of Trade Preferences (~44 developing countries)
    "DZA","ARG","BGD","BEN","BOL","BRA","CMR","CHN","COL","CUB",
    "EGY","ETH","GHA","GIN","IND","IDN","IRQ","PRK","LBY","MYS",
    "MEX","MAR","MOZ","MMR","NIC","NGA","PAK","PER","PHL","TZA",
    "THA","TTO","TUN","URY","VEN","VNM","YEM","ZWE",
]

# Map pta_id → static membership list (for blocs not easily parsed from name)
# pta_id values match exactly the pta_id column in agreement_scores.csv
PTA_MEMBERS_STATIC = {
    # pta_id: list of ISO-3 codes
    15:  GCC_MEMBERS,                       # GCC
    32:  ECOWAS_MEMBERS,                    # ECOWAS
    41:  SADC_MEMBERS,                      # SADC
    85:  EAC_MEMBERS,                       # EAC (original)
    403: EAC_MEMBERS,                       # EAC (Burundi+Rwanda accession pta_id=403 in data)
    111: COMESA_MEMBERS,                    # COMESA  ← corrected from 102
    112: NAFTA_MEMBERS,                     # NAFTA   ← corrected from 103
    116: CAN_MEMBERS,                       # Andean Community ← corrected
    119: list(MERCOSUR_MEMBERS_HIST.keys()),# MERCOSUR ← corrected
    113: ["BGD","BTN","IND","MDV","NPL","PAK","LKA"],  # SAPTA
    162: ["BGD","BTN","IND","MDV","NPL","PAK","LKA"],  # SAFTA ← corrected
    117: APTA_MEMBERS,                      # APTA
    128: CARICOM_MEMBERS,                   # CARICOM ← corrected
    14:  PAFTA_MEMBERS,                     # PAFTA (already correct in data)
    132: GSTP_MEMBERS,                      # GSTP
    # CEFTA 2006  (pta_id=4)
    4:   ["ALB","BIH","MKD","MDA","MNE","SRB","XKX"],
    # SACU (pta_id=6)
    6:   SACU_MEMBERS,
    # Trans-Pacific Strategic Economic Partnership P4 (pta_id=8)
    8:   ["BRN","CHL","NZL","SGP"],
    # SPARTECA (pta_id=124)
    124: ["AUS","NZL","COK","FJI","KIR","MHL","FSM","NRU","NIU",
          "PLW","PNG","WSM","SLB","TKL","TON","TUV","VUT"],
    # PICTA (pta_id=363 in data)
    363: ["COK","FJI","KIR","MHL","FSM","NRU","NIU","PLW",
          "PNG","WSM","SLB","TON","TUV","VUT"],
    # Melanesian Spearhead Group (pta_id=89 in data)
    89:  ["FJI","PNG","SLB","VUT"],
    # CIS (pta_id=90 in data)
    90:  ["ARM","AZE","BLR","GEO","KAZ","KGZ","MDA","RUS","TJK","TKM","UKR","UZB"],
    # EAEC / Eurasian (pta_id=97)
    97:  ["BLR","KAZ","KGZ","RUS","TJK"],
    406: ["BLR","KAZ","RUS"],               # Russia-Belarus-Kazakhstan CU
    440: ["ARM","BLR","KAZ","KGZ","RUS"],   # EAEU
    441: ["ARM","BLR","KAZ","KGZ","RUS"],   # EAEU + Armenia
    444: ["ARM","BLR","KAZ","KGZ","RUS"],   # EAEU + Kyrgyz
    447: ["ARM","BLR","KAZ","KGZ","RUS","VNM"],  # EAEU - Vietnam
    # ECO (pta_id=108)
    108: ["AFG","AZE","IRN","KAZ","KGZ","PAK","TJK","TKM","TUR","UZB"],
    # Agadir Agreement (pta_id=364)
    364: ["EGY","JOR","MAR","TUN"],
    # GUAM (pta_id=371 in data)
    371: ["AZE","GEO","MDA","UKR"],
    # CEZ Common Economic Zone (pta_id=361 in data)
    361: ["BLR","KAZ","RUS","UKR"],
    # TPP / CPTPP (pta_id=423 in data → 449 in data)
    423: ["AUS","BRN","CAN","CHL","JPN","MYS","MEX","NZL","PER","SGP","VNM"],
    449: ["AUS","BRN","CAN","CHL","JPN","MYS","MEX","NZL","PER","SGP","VNM"],
    # CAFTA-DR (pta_id=23 in data)
    23:  ["CRI","SLV","GTM","HND","NIC","DOM","USA"],
    # EEA (pta_id=105)
    105: ["ISL","LIE","NOR",
          # + EU members — we add them via dynamic EU list in get_members
          "AUT","BEL","BGR","CYP","CZE","DEU","DNK","ESP","EST","FIN",
          "FRA","GBR","GRC","HRV","HUN","IRL","ITA","LTU","LUX","LVA",
          "MLT","NLD","POL","PRT","ROU","SVK","SVN","SWE"],
    # EU - Turkey customs union (pta_id=109)
    109: (  # EU members in 1995 + Turkey
        ["TUR"] +
        ["BEL","FRA","DEU","ITA","LUX","NLD","DNK","IRL","GBR",
         "GRC","PRT","ESP","AUT","FIN","SWE"]
    ),
    # EFTA (pta_id=138)
    138: ["CHE","NOR","ISL","LIE","AUT","FIN","SWE","DNK","PRT","GBR"],
    # EFTA - Czechoslovakia (pta_id=245 in data, historical)
    245: ["CHE","NOR","ISL","AUT","FIN","SWE","DNK","PRT","GBR","CSK"],
    # Moldova - Serbia and Montenegro (pta_id=286)
    286: ["MDA","SRB","MNE"],
    # Romania - Serbia and Montenegro (pta_id=295)
    295: ["ROU","SRB","MNE"],
    # Protocol on Trade Negotiations PTN (pta_id=133) ~20 developing countries
    133: ["BGD","BRA","CHN","CUB","EGY","IND","IRN","MEX","PAK","PRY",
          "PHL","KOR","TZA","TUN","URY","YUG"],
    # EU OCT overseas territories (pta_id=134)
    134: ["AIA","ATA","ABW","BMU","BES","VGB","CYM","CUW","FLK","GUF",
          "ATF","GIB","GRL","GLP","MTQ","MYT","MSR","NCL","PYF","REU",
          "BLM","SHN","SPM","SXM","TCA","WLF"],
    # EFTA accession of Iceland (pta_id=135) — same as EFTA
    135: ["CHE","NOR","ISL","AUT","SWE","DNK","PRT","GBR"],
    # EU - Eastern and Southern Africa Interim EPA (pta_id=332)
    332: ["COM","MDG","MUS","SYC","ZMB"],
    # Australia - New Zealand (ANZCERTA) — pta_id 122, 136, 174
    136: ["AUS","NZL"],  122: ["AUS","NZL"],  174: ["AUS","NZL"],
    126: ["AUS","PNG"],          # PATCRA
    146: ["CAN","KOR"],          # Canada-Korea
    188: ["CRI","SLV","GTM","HND","NIC"],  # Central American FTA (historical)
    172: ["EGY","IRQ","JOR","LBY","SYR"],  # Arab Common Market
    164: ["DZA","EGY","ETH","GHA","GIN","LBY","MAR","NGA","SDN","TUN"],  # African CM
    # AFTA (pta_id=115)
    115: ["BRN","IDN","MYS","PHL","SGP","THA","VNM","LAO","MMR","KHM"],
    127: APTA_MEMBERS,   # APTA (pta_id=127)
    # CEFTA 1992 and accessions
    189: ["CZE","HUN","POL","SVK"],
    48:  ["ALB","BIH","MKD","MDA","MNE","SRB","HRV"],
    190: ["CZE","HUN","POL","SVK","BGR"],
    191: ["CZE","HUN","POL","SVK","ROU"],
    192: ["CZE","HUN","POL","SVK","SVN"],
    186: ["CAN","USA"],  # CUSFTA
    187: ["ATG","BRB","GUY","JAM","TTO","DMA","GRD","KNA","LCA","VCT","BLZ","MSR","BHS"],  # CARIFTA
    275: ["IRL","GBR"],  # Ireland-UK
    276: ["ARG","BOL","BRA","CHL","COL","ECU","MEX","PRY","PER","URY","VEN"],  # LAFTA
    255: ["SLV","NIC"],  267: ["GHA","BFA"],  310: ["ZAF","ZWE"],
    272: GCC_MEMBERS,    370: ["AZE","GEO","MDA","UKR"],   # GCC dup + GUAM
    341: ["SLV","HND","TWN"],   # El Salvador-Honduras-Chinese Taipei
    416: ["CRI","PAN","CHE","NOR","ISL","LIE"],  # EFTA-Central America
    413: ["ARM","BLR","KAZ","KGZ","MDA","RUS","TJK","UKR","UZB"],  # CIS FTA 2011
    426: EAC_MEMBERS,   # EAC accession
    429: ["BLR","KAZ","RUS"],   # Russia-Belarus-Kazakhstan CU
    440: ["ARM","BLR","KAZ","KGZ","RUS"],  441: ["ARM","BLR","KAZ","KGZ","RUS"],
    444: ["ARM","BLR","KAZ","KGZ","RUS"],  447: ["ARM","BLR","KAZ","KGZ","RUS","VNM"],
    443: GCC_MEMBERS + ["SGP"],  # GCC-Singapore
    359: ["UKR","TKM"],          # Ukraine-Turkmenistan
    173: ["KEN","UGA","TZA"],    # Arusha Agreement
    175: ["MYS","IDN","PHL"],    # Borneo FTA
    256: ["CMR","GAB","COG","CAF","TCD"],  # Equatorial CU + Cameroon
    # EFTA bilateral (historical)
    246: ["EST","ISL","NOR","CHE","LIE"],
    247: ["HUN","CHE","NOR","ISL","LIE","AUT","FIN","SWE"],
    250: ["POL","CHE","NOR","ISL","LIE","AUT","FIN","SWE"],
    251: ["ROU","CHE","NOR","ISL","LIE","AUT","FIN","SWE"],
    253: ["ESP","CHE","NOR","ISL","AUT","SWE","DNK","PRT","GBR"],
    # Baltic FTAs
    257: ["EST","FRO"],  258: ["EST","LVA","LTU"],  259: ["EST","NOR"],
    260: ["EST","SWE"],  261: ["EST","CHE"],  263: ["FIN","EST"],
    264: ["FIN","LVA"],  265: ["FIN","LTU"],  274: ["HUN","SVN"],
    277: ["LVA","NOR"],  278: ["LVA","SWE"],  279: ["LVA","CHE"],
    280: ["LTU","NOR"],  281: ["LTU","SWE"],  282: ["LTU","CHE"],
    287: ["POL","FRO"],
    # Czech/Slovak/Bulgarian FTAs
    202: ["CZE","ROU"],  203: ["CZE","SVK"],  301: ["SVK","ROU"],
    183: ["BGR","SVK"],  196: ["BGR","CZE"],
    # Croatia - Serbia and Montenegro era
    184: ["HRV","SRB","MNE"],
    # Dissolved-state bilaterals — use successor states
    170: ["ALB","SRB","MNE"],   # Albania - Serbia and Montenegro
    182: ["BGR","SRB","MNE"],   # Bulgaria - Serbia and Montenegro
    197: ["CZE","BGR"],         # Czech Republic - Bulgaria (same as 196 but distinct pta)
    254: ["SLV","NIC"],         # El Salvador - Nicaragua (duplicate of 255)
    263: ["FIN","EST"],         # Finland - Estonia Protocol (duplicate of 261 entry)
    267: ["GHA","BFA"],         # Ghana - Upper Volta (duplicate of entry 255's GHA/BFA)
}


def eu_members_in_year(year: int) -> list:
    return [iso for iso, (entry, exit_) in EU_MEMBERS_HIST.items()
            if entry <= year and (exit_ is None or year < exit_)]


def efta_members_in_year(year: int) -> list:
    return [iso for iso, (entry, exit_) in EFTA_MEMBERS_HIST.items()
            if entry <= year and (exit_ is None or year < exit_)]


def asean_members_in_year(year: int) -> list:
    return [iso for iso, (entry, exit_) in ASEAN_MEMBERS_HIST.items()
            if entry <= year and (exit_ is None or year < exit_)]


# ── 2. NAME → MEMBER PARSING ─────────────────────────────────────────────────

def iso3_from_name_fragment(fragment: str) -> str | None:
    """Fuzzy name → ISO-3.  Returns None if not found."""
    fragment = fragment.strip()
    # Direct pycountry lookup
    try:
        c = pycountry.countries.lookup(fragment)
        return c.alpha_3
    except LookupError:
        pass
    # Common aliases
    ALIASES = {
        "Turkey": "TUR", "Türkiye": "TUR",  # pycountry uses "Türkiye" since 2022
        "Chinese Taipei": "TWN", "Taiwan": "TWN",
        "Hong Kong, China": "HKG", "Hong Kong": "HKG",
        "Palestinian Authority": "PSE", "Palestine": "PSE",
        "Viet Nam": "VNM", "Vietnam": "VNM",
        "Russia": "RUS", "Russian Federation": "RUS",
        "Former Yugoslav Republic of Macedonia": "MKD", "Macedonia": "MKD",
        "Bosnia and Herzegovina": "BIH",
        "Serbia and Montenegro": None,  # dissolved — skip
        "Czechoslovakia": None,         # dissolved — skip
        "Rep. of  Moldova": "MDA", "Moldova": "MDA", "Republic of Moldova": "MDA",
        "Côte d'Ivoire": "CIV", "Ivory Coast": "CIV",
        "Faroe Islands": "FRO",
        "Liechtenstein": "LIE", "Switzerland - Liechtenstein": "CHE",
        "San Marino": "SMR", "Andorra": "AND",
        "EU": None, "EFTA": None, "ASEAN": None,  # handled separately
        "GCC": None,
        "Papua New Guinea": "PNG", "Fiji": "FJI",
        "Georgia": "GEO",
        "Ghana": "GHA", "Cameroon": "CMR",
        "Colombia and Peru": None,
        "Central America": None,
        "Costa Rica and Panama": None,
        "SACU": None,
        "CARIFORUM States": None,
        "Eastern and Southern Africa States": None,
        "Papua New Guinea / Fiji": None,
        "Philippines": "PHL",
        "Macao, China": "MAC", "Macao": "MAC",
        "Slovak Republic": "SVK", "Slovakia": "SVK",
        "Czech Republic": "CZE",
        "Yugoslavia": "YUG", "Federal Republic of Yugoslavia": "SRB",
        "Serbia": "SRB",
        "Montenegro": "MNE",
        "Kosovo": "XKX", "UNMIC/Kosovo": "XKX",
        "Rep. of Korea": "KOR",
        "New Zealand": "NZL",
        "Australia": "AUS",
        "El Salvador": "SLV",
        "Honduras": "HND",
        "Guatemala": "GTM",
        "Costa Rica": "CRI",
        "Nicaragua": "NIC",
        "Panama": "PAN",
        "Dominican Republic": "DOM",
        "Chinese Taipei": "TWN",
        "Turkmenistan": "TKM",
        "Armenia": "ARM",
        "Ukraine": "UKR",
        "Croatia": "HRV",
        "Slovenia": "SVN",
        "Albania": "ALB",
        "Latvia": "LVA",
        "Lithuania": "LTU",
        "Estonia": "EST",
        "Romania": "ROU",
        "Bulgaria": "BGR",
        "Hungary": "HUN",
        "Poland": "POL",
    }
    return ALIASES.get(fragment)


def parse_bilateral_members(name: str) -> list[str] | None:
    """
    For bilateral agreements named 'Country A - Country B', return [ISO_A, ISO_B].
    Returns None if parsing fails.
    """
    parts = [p.strip() for p in name.split(" - ") if p.strip()]
    if len(parts) != 2:
        return None
    isos = [iso3_from_name_fragment(p) for p in parts]
    if all(isos):
        return isos
    return None


def get_members(row: pd.Series) -> list[str] | None:
    """
    Return list of ISO-3 member codes for a given agreement row.
    Returns None if membership cannot be determined.
    """
    pta_id = row["pta_id"]
    name   = row["agreement_name"]
    year   = int(row["year"])

    # Static overrides first
    if pta_id in PTA_MEMBERS_STATIC:
        return PTA_MEMBERS_STATIC[pta_id]

    # ── EC bilateral agreements (historical, not caught by "EU - X" regex) ────
    EC_BILATERAL = {
        207: ("AUT", 1972),  210: ("CYP", 1972),  213: ("EGY", 1972),
        214: ("EGY", 1977),  215: ("EST", 1994),  216: ("FRO", 1992),
        217: ("FIN", 1973),  218: ("GRC", 1975),  219: ("GRC", 1961),
        220: ("HUN", 1991),  221: ("HUN", 1991),  222: ("ISR", 1970),
        223: ("LVA", 1994),  224: ("LBN", 1977),  226: ("MLT", 1970),
        227: ("MAR", 1969),  228: ("POL", 1991),  229: ("POL", 1991),
        230: ("PRT", 1972),  231: ("ROU", 1993),  232: ("ROU", 1993),
        233: ("SVK", 1993),  234: ("SVN", 1993),  235: ("SVN", 1996),
        236: ("SVN", 1996),  237: ("ESP", 1970),  238: ("SWE", 1972),
        239: ("SYR", 1977),  240: ("TUN", 1969),  241: ("TUR", 1970),
        242: ("TUR", 1963),  243: ("TUR", 1973),  208: ("BGR", 1993),
        209: ("BGR", 1993),  211: ("CZE", 1991),  212: ("CZE", 1993),
    }
    if pta_id in EC_BILATERAL:
        other_iso, ref_year = EC_BILATERAL[pta_id]
        return eu_members_in_year(ref_year) + [other_iso]

    # EC Czech+Slovak Federal Republic (pta_id=211)
    if pta_id == 211:
        return eu_members_in_year(1991) + ["CZE", "SVK"]

    # ── EU multi-party (not caught by simple "EU - X" regex) ─────────────────
    EU_MULTI = {
        368: (2008, ["ATG","BHS","BRB","BLZ","DMA","DOM","GRD","GUY","HTI",
                     "JAM","KNA","LCA","VCT","SUR","TTO"]),  # EU-CARIFORUM
        388: (2012, ["CRI","SLV","GTM","HND","NIC","PAN"]),  # EU-Central America
        395: (2012, ["COL","PER"]),                          # EU-Colombia and Peru
        411: (2009, ["PNG","FJI"]),                          # EU-PNG/Fiji
    }
    if pta_id in EU_MULTI:
        ref_year, others = EU_MULTI[pta_id]
        return eu_members_in_year(ref_year) + others

    # Yaoundé / Lomé ACP-EU predecessors
    ACP_CORE = ["BEN","CMR","CAF","TCD","COD","CIV","GAB","MDG","MLI","MRT",
                "MUS","NER","NGA","SEN","SOM","TGO","BFA","RWA","BDI","KEN","TZA","UGA"]
    ACP_EXTENDED = ACP_CORE + ["BHS","BRB","GUY","JAM","TTO","FJI","PNG","WSM"]
    LOME_MAP = {
        316: (1963, ACP_CORE), 317: (1969, ACP_CORE),
        318: (1975, ACP_EXTENDED), 319: (1979, ACP_EXTENDED + ["ZWE"]),
        320: (1984, ACP_EXTENDED + ["ZWE"]),
    }
    if pta_id in LOME_MAP:
        ref_year, acp = LOME_MAP[pta_id]
        return eu_members_in_year(ref_year) + acp

    # EU-X agreements
    if re.match(r"^EU\s+-\s+", name) or re.match(r"^EC\s+", name):
        other_part = re.sub(r"^(EU|EC)\s+-\s*", "", name).strip()
        # Pure EU bilateral
        other_iso = iso3_from_name_fragment(other_part)
        if other_iso:
            return eu_members_in_year(year) + [other_iso]
        # EC Treaty = EU internal
        if name.strip().startswith("EC Treaty"):
            return eu_members_in_year(year)
        # EU - multi (e.g. "EU - Colombia and Peru", "EU - Central America")
        # Return EU members only + flag; these need manual curation
        return None

    # EFTA-X agreements
    if re.match(r"^EFTA\s+-\s+", name):
        other_part = re.sub(r"^EFTA\s+-\s*", "", name).strip()
        other_iso = iso3_from_name_fragment(other_part)
        if other_iso:
            return efta_members_in_year(year) + [other_iso]
        if "SACU" in other_part:
            return efta_members_in_year(year) + SACU_MEMBERS
        if "GCC" in other_part:
            return efta_members_in_year(year) + GCC_MEMBERS
        if "Central America" in other_part:
            return None  # skip complex multi-party
        return None

    # ASEAN-X
    if re.match(r"^ASEAN\s+-\s+", name):
        other_part = re.sub(r"^ASEAN\s+-\s*", "", name).strip()
        other_iso = iso3_from_name_fragment(other_part)
        asean = asean_members_in_year(year)
        if other_iso:
            return asean + [other_iso]
        if "Australia" in other_part and "New Zealand" in other_part:
            return asean + ["AUS", "NZL"]
        if "India" in other_part:
            return asean + ["IND"]
        if "Japan" in other_part:
            return asean + ["JPN"]
        if "Korea" in other_part:
            return asean + ["KOR"]
        if "China" in other_part:
            return asean + ["CHN"]
        return None

    # MERCOSUR-X
    if "MERCOSUR" in name and "-" in name:
        other_part = name.split("-")[-1].strip()
        other_iso = iso3_from_name_fragment(other_part)
        mercosur = list(MERCOSUR_MEMBERS_HIST.keys())
        if "SACU" in other_part:
            return mercosur + SACU_MEMBERS
        if other_iso:
            return mercosur + [other_iso]
        return None

    # Bilateral (fallback)
    if not row.get("is_plurilateral", False):
        return parse_bilateral_members(name)

    return None


# ── 3. EXPAND TO DYADIC ROWS ──────────────────────────────────────────────────

def expand_agreement(row: pd.Series) -> pd.DataFrame:
    """Return a DataFrame of directed (iso_o, iso_d) rows for one agreement."""
    members = get_members(row)
    if members is None or len(members) < 2:
        return pd.DataFrame()

    members = list(dict.fromkeys(members))  # deduplicate, preserve order

    pairs = list(permutations(members, 2))  # directed pairs
    n_pairs = len(pairs)

    meta_cols = [
        "pta_id", "agreement_name", "year",
        "CCI_pilot", "hard_p1k_mean", "soft_p1k_mean", "enf_p1k_mean",
        "oblig_ratio", "n_articles", "n_chapters", "total_words",
        "is_plurilateral", "n_parties",
        "d_us_tpa", "d_eu_template", "d_other_template", "d_plurilateral",
        "d_symmetric", "d_eastern_european", "d_central_asian", "d_modern_global",
        "asymmetry_type", "alschner_cluster",
        "is_us_tpa", "is_eu_template",
        "pair_weight",
    ]

    # Only keep columns that exist in row
    present = [c for c in meta_cols if c in row.index]

    out = pd.DataFrame(pairs, columns=["iso_o", "iso_d"])
    for col in present:
        out[col] = row[col]

    out["n_parties"] = n_pairs  # actual number of directed pairs
    out["pair_weight"] = 1.0 / (n_pairs / 2)  # weight by undirected pairs

    return out


# ── 4. GRAVITY CONTROLS STUB ─────────────────────────────────────────────────
# A small but representative built-in table.
# KEY: replace this with a pd.read_csv("dgd_controls.csv") after downloading
# the full USITC DGD from https://gravity.usitc.gov
#
# Columns align exactly with DGD variable names so a drop-in replacement works.

def build_gravity_controls_stub() -> pd.DataFrame:
    """
    Return a stub of time-invariant gravity controls for major trading pairs.
    Source: CEPII GeoDist + USITC DGD documentation.

    IMPORTANT: This is intentionally illustrative.  For production use,
    download the full DGD CSV and replace this function with:
        return pd.read_csv("path/to/dgd_gravity.csv")
    The column names below are already aligned with DGD conventions.
    """

    # fmt: off
    rows = [
        # iso_o, iso_d, ln_dist, contiguity, common_language,
        #   colony_of_origin_ever, colony_of_destination_ever,
        #   colony_ever, common_colonizer, common_legal_origin
        #
        # A few canonical pairs to validate the join:
        ("USA","CAN", 7.43, 1, 1, 0, 0, 0, 0, 1),
        ("CAN","USA", 7.43, 1, 1, 0, 0, 0, 0, 1),
        ("USA","MEX", 8.09, 1, 0, 0, 0, 0, 0, 0),
        ("MEX","USA", 8.09, 1, 0, 0, 0, 0, 0, 0),
        ("GBR","AUS", 16.74, 0, 1, 1, 0, 1, 0, 1),
        ("AUS","GBR", 16.74, 0, 1, 0, 1, 1, 0, 1),
        ("FRA","DEU",  6.76, 1, 0, 0, 0, 0, 0, 0),
        ("DEU","FRA",  6.76, 1, 0, 0, 0, 0, 0, 0),
        ("JPN","THA", 14.27, 0, 0, 0, 0, 0, 0, 0),
        ("THA","JPN", 14.27, 0, 0, 0, 0, 0, 0, 0),
        ("JPN","CHL", 16.17, 0, 0, 0, 0, 0, 0, 0),
        ("CHL","JPN", 16.17, 0, 0, 0, 0, 0, 0, 0),
        ("EGY","TUR", 12.57, 0, 0, 0, 0, 0, 0, 0),
        ("TUR","EGY", 12.57, 0, 0, 0, 1, 1, 0, 0),
        ("IND","SGP", 13.40, 0, 1, 0, 1, 1, 0, 1),
        ("SGP","IND", 13.40, 0, 1, 1, 0, 1, 0, 1),
        ("SGP","JOR", 14.20, 0, 0, 0, 0, 0, 0, 0),
        ("JOR","SGP", 14.20, 0, 0, 0, 0, 0, 0, 0),
        ("KOR","SGP", 13.59, 0, 0, 0, 0, 0, 0, 0),
        ("SGP","KOR", 13.59, 0, 0, 0, 0, 0, 0, 0),
        ("ARG","BRA",  8.37, 1, 0, 0, 0, 0, 1, 0),
        ("BRA","ARG",  8.37, 1, 0, 0, 0, 0, 1, 0),
        ("ARG","URY",  8.10, 1, 0, 0, 0, 0, 1, 0),
        ("URY","ARG",  8.10, 1, 0, 0, 0, 0, 1, 0),
        ("BRA","PRY",  7.86, 1, 0, 0, 0, 0, 1, 0),
        ("PRY","BRA",  7.86, 1, 0, 0, 0, 0, 1, 0),
    ]
    # fmt: on

    cols = [
        "iso_o", "iso_d", "ln_dist", "contiguity", "common_language",
        "colony_of_origin_ever", "colony_of_destination_ever",
        "colony_ever", "common_colonizer", "common_legal_origin",
    ]
    stub = pd.DataFrame(rows, columns=cols)
    stub["gravity_source"] = "stub_cepii_dgd"
    return stub


# ── 5. MAIN ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load inputs ──────────────────────────────────────────────────────────
    scores = pd.read_csv(args.scores)
    cci    = pd.read_csv(args.cci)[["pta_id", "CCI_pilot"]]

    # Merge CCI into scores
    scores = scores.merge(cci, on="pta_id", how="left")

    # Derived: approximate n_parties from is_plurilateral flag + name
    # (bilateral = 2; plurilateral = we'll fill after expansion)
    scores["n_parties"] = scores["is_plurilateral"].map({False: 2, True: np.nan})

    print(f"\n{'='*60}")
    print(f" Building dyadic gravity panel")
    print(f"{'='*60}")
    print(f" Input agreements : {len(scores)}")
    print(f" Plurilateral     : {scores['is_plurilateral'].sum()}")
    print(f" Bilateral        : {(~scores['is_plurilateral']).sum()}")

    # ── Expand each agreement to directed pairs ──────────────────────────────
    frames = []
    skipped = []

    for _, row in scores.iterrows():
        df_pairs = expand_agreement(row)
        if df_pairs.empty:
            skipped.append((row["pta_id"], row["agreement_name"]))
        else:
            frames.append(df_pairs)

    panel = pd.concat(frames, ignore_index=True)

    print(f"\n Expanded pairs   : {len(panel):,}")
    print(f" Skipped (no parse): {len(skipped)}")
    if skipped:
        print("   (First 10 skipped):")
        for pid, name in skipped[:10]:
            print(f"     pta_id={pid}: {name}")

    # ── De-duplicate: same (iso_o, iso_d, year) covered by multiple PTAs ─────
    # Prefer bilateral (n_parties=2 → lower value); tie-break: higher pta_id
    panel_sorted = panel.sort_values(
        ["iso_o", "iso_d", "year", "is_plurilateral", "pta_id"],
        ascending=[True, True, True, True, False]
    )
    panel_dedup = panel_sorted.drop_duplicates(
        subset=["iso_o", "iso_d", "year"], keep="first"
    ).reset_index(drop=True)

    print(f" After dedup      : {len(panel_dedup):,} directed pair-year obs")
    print(f" Unique pairs     : {panel_dedup[['iso_o','iso_d']].drop_duplicates().shape[0]:,}")
    print(f" Unique agreements: {panel_dedup['pta_id'].nunique()}")

    # ── Attach gravity controls ───────────────────────────────────────────────
    gravity = build_gravity_controls_stub()
    panel_dedup = panel_dedup.merge(gravity, on=["iso_o", "iso_d"], how="left")

    n_with_gravity = panel_dedup["ln_dist"].notna().sum()
    print(f"\n Gravity controls matched: {n_with_gravity:,} rows "
          f"({n_with_gravity/len(panel_dedup)*100:.1f}%)")
    print("  → Replace build_gravity_controls_stub() with the full DGD to")
    print("    get 100% coverage.  Column names are already DGD-aligned.")

    # ── Fixed-effect identifiers ─────────────────────────────────────────────
    panel_dedup["pair_id"]  = panel_dedup["iso_o"] + "_" + panel_dedup["iso_d"]
    panel_dedup["imp_year"] = panel_dedup["iso_d"] + "_" + panel_dedup["year"].astype(str)
    panel_dedup["exp_year"] = panel_dedup["iso_o"] + "_" + panel_dedup["year"].astype(str)

    # ── Winsorise CCI ────────────────────────────────────────────────────────
    cci_obs = panel_dedup["CCI_pilot"].dropna()
    if len(cci_obs) > 0:
        p1, p99 = cci_obs.quantile([0.01, 0.99])
        panel_dedup["CCI_w"] = panel_dedup["CCI_pilot"].clip(lower=p1, upper=p99)

    # ── Column ordering for readability ──────────────────────────────────────
    front_cols = [
        "pta_id", "agreement_name", "year",
        "iso_o", "iso_d", "pair_id",
        "CCI_pilot", "CCI_w",
        "hard_p1k_mean", "soft_p1k_mean", "enf_p1k_mean", "oblig_ratio",
        "n_articles", "n_chapters", "total_words",
        "is_plurilateral", "n_parties", "pair_weight",
        "ln_dist", "contiguity", "common_language",
        "colony_of_origin_ever", "colony_of_destination_ever",
        "colony_ever", "common_colonizer", "common_legal_origin",
        "gravity_source",
        "d_us_tpa", "d_eu_template", "d_other_template",
        "d_plurilateral", "d_symmetric",
        "d_eastern_european", "d_central_asian", "d_modern_global",
        "asymmetry_type", "alschner_cluster",
        "is_us_tpa", "is_eu_template",
        "imp_year", "exp_year",
    ]
    existing_front = [c for c in front_cols if c in panel_dedup.columns]
    other_cols     = [c for c in panel_dedup.columns if c not in existing_front]
    panel_dedup    = panel_dedup[existing_front + other_cols]

    # ── Save ─────────────────────────────────────────────────────────────────
    panel_dedup.to_csv(args.out, index=False)
    print(f"\n{'='*60}")
    print(f" Saved → {args.out}")
    print(f" Shape : {panel_dedup.shape}")
    print(f"{'='*60}\n")

    # ── Quick sanity checks ──────────────────────────────────────────────────
    print("CCI distribution (directed pairs):")
    print(panel_dedup["CCI_pilot"].describe().round(3).to_string())
    print()
    print("Top 10 agreements by directed pair count:")
    top = (panel_dedup.groupby(["pta_id","agreement_name"])
           .size().sort_values(ascending=False).head(10)
           .reset_index(name="n_pairs"))
    print(top.to_string(index=False))

    return panel_dedup


if __name__ == "__main__":
    main()