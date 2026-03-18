"""
pta_asymmetry.py
================
Power-asymmetry and template-imposition detection for PTA texts.

Motivation
----------
A high Contract Completeness Score (CCS) can mean two structurally
different things:
  (A) Genuinely complete agreement — both parties negotiated detailed,
      enforceable obligations reflecting their bilateral relationship.
  (B) Template imposition — a dominant party (US, EU, Japan) exported
      a standardized boilerplate, inflating the score regardless of
      the relationship quality.

Conflating (A) and (B) in the analytical extension regression would
bias the coefficients on dyadic relationship variables (common language,
colonial history, political alignment). This module generates flags that
let you separate the two.

Method
------
Detection uses three complementary signals, all computable from the
df_meta table produced by parse_tota_xml() — no pairwise Jaccard needed:

  1. Party-based template detection
     Known template-exporting ISO codes (USA post-2002, EU/EC/EEC, JPN, CHN)
     matched against the parties list. Era-specific rules applied where
     the template changed over time (e.g. US pre/post Trade Promotion
     Authority 2002).

  2. GDP/power asymmetry proxy
     G20 major economy membership as a proxy for bargaining power.
     Agreements with exactly one G20 party are flagged as potentially
     asymmetric even where no template is detected (e.g. Singapore-Australia).

  3. Plurilateral structure
     3+ party agreements receive a separate flag. Template imposition
     in plurilateral agreements (CAFTA-DR, TPP) is mechanically different
     from bilateral template — the dominant party sets terms for a
     coalition rather than a specific counterpart.

Grounding
---------
Template clusters are grounded in Alschner, Seiermann & Skougarevskiy (2018),
"Text-as-data analysis of preferential trade agreements: Mapping the PTA
landscape," UNCTAD Research Paper No. 5, Figures 4 and 5.

  US TPA template: Fig 4 — near-identical cluster of post-2002 US agreements
    (US-Singapore 2003, US-Chile 2003, CAFTA-DR 2004, US-Morocco 2004,
    US-Australia 2004, US-Bahrain 2005, US-Peru 2006, US-Oman 2006,
    US-Colombia 2006, US-Panama 2007, Korea-US 2007, TPP 2016)

  EU template clusters: Fig 5 — seven sub-clusters by era and partner group,
    all sharing high within-cluster similarity due to the EU's systematic
    use of Association Agreement and FTA templates.

Usage
-----
    from pta_asymmetry import classify_asymmetry, add_asymmetry_flags
    import pandas as pd

    # Apply to a full df_meta DataFrame
    df_meta = add_asymmetry_flags(df_meta)

    # Inspect one agreement
    row = df_meta[df_meta['pta_id'] == 1].iloc[0]
    print(classify_asymmetry(row))

    # Symmetric-only subsample for robustness check
    df_symmetric = df_meta[df_meta['asymmetry_type'] == 'bilateral_symmetric']

Regression use
--------------
After adding flags to df_meta, merge into df_cci:

    df_cci = df_cci.merge(
        df_meta[['pta_id', 'template_flag', 'is_template',
                 'asymmetry_type', 'n_g20_parties']],
        on='pta_id', how='left'
    )

Then in the extension regression:
    CCS = β·dyadic_vars + γ·is_template + δ·(is_template × CCS) + FE + ε

Three robustness checks:
    1. Full sample vs df_cci[df_cci['asymmetry_type']=='bilateral_symmetric']
    2. Mean CCS decomposed by template_flag
    3. Interaction: colonial_history coefficient in template vs symmetric subsamples
"""

import re
from typing import Optional

# ── G20 / major economies (ISO 3-letter codes) ───────────────────────────────
# Used as a proxy for bargaining power when no known template is detected.
G20_ISO = {
    'USA', 'CHN', 'DEU', 'GBR', 'JPN', 'FRA', 'IND', 'BRA', 'CAN', 'AUS',
    'KOR', 'ITA', 'RUS', 'MEX', 'IDN', 'TUR', 'SAU', 'ARG', 'ZAF',
    'EU',  'EC',  'EEC',  # European Union / Communities (various ISO usage)
}

# ── Template exporter rules ───────────────────────────────────────────────────
# Each key is an ISO code. Rules are checked in order; first match wins.
# 'min_year' / 'max_year' are inclusive.
#
# Source: Alschner et al. 2018, Figures 4 & 5.
TEMPLATE_RULES = {
    'USA': [
        {
            'min_year': 2002,
            'max_year': 9999,
            'label':    'us_tpa_template',
            'note':     (
                'Post-TPA 2002 US template. Trade Promotion Authority (2002) '
                'standardized subsequent US FTA design. Near-identical cluster '
                'in Alschner et al. Fig 4.'
            ),
        },
        {
            'min_year': 0,
            'max_year': 2001,
            'label':    'us_pre_tpa',
            'note':     (
                'Pre-TPA US agreement. Less standardized than post-2002 '
                'template; includes US-Israel 1985, CUSFTA 1988, NAFTA 1992, '
                'US-Jordan 2000.'
            ),
        },
    ],
    'EU': [
        {
            'min_year': 0,
            'max_year': 9999,
            'label':    'eu_template',
            'note':     (
                'EU/EC association or FTA template. Seven sub-clusters by era '
                'in Alschner et al. Fig 5, all sharing systematic EU design.'
            ),
        },
    ],
    'EC': [
        {
            'min_year': 0,
            'max_year': 9999,
            'label':    'eu_template',
            'note':     'European Communities — same EU template family.',
        },
    ],
    'EEC': [
        {
            'min_year': 0,
            'max_year': 9999,
            'label':    'eu_template',
            'note':     'EEC early association agreements (pre-1993).',
        },
    ],
    'JPN': [
        {
            'min_year': 2002,
            'max_year': 9999,
            'label':    'japan_epa_template',
            'note':     (
                'Japan Economic Partnership Agreement template post-2002. '
                'Japan began systematic EPA programme from 2002 with '
                'standardized chapter design.'
            ),
        },
    ],
    'CHN': [
        {
            'min_year': 2003,
            'max_year': 9999,
            'label':    'china_template',
            'note':     (
                'China FTA template post-2003. China began active FTA '
                'programme from 2003 with a recognizable chapter structure.'
            ),
        },
    ],
}

# ── Known US TPA agreements (named list for secondary verification) ───────────
# From Alschner et al. Fig 4, chronological order.
US_TPA_AGREEMENTS = {
    'us_singapore', 'us_chile', 'cafta', 'cafta-dr',
    'us_morocco', 'us_australia', 'us_bahrain',
    'us_peru', 'us_oman', 'us_colombia', 'us_panama',
    'us_korea', 'korea_us', 'tpp', 'trans-pacific partnership',
    'usmca', 'us_mexico_canada',
}

# ── Alschner cluster membership ───────────────────────────────────────────────
# Maps the three main clusters from Fig 3 to characteristics.
# Used as a structural covariate in the extension regression.
ALSCHNER_CLUSTERS = {
    'eastern_european': {
        'description': 'Eastern European Cluster — pre-1995 Goods FTAs, '
                       'high similarity (~50%), CIS and EU accession countries.',
        'typical_parties': {'RUS', 'POL', 'CZE', 'SVK', 'HUN', 'BGR', 'ROU',
                            'UKR', 'BLR', 'KAZ', 'UZB', 'ARM', 'GEO', 'AZE',
                            'MDA', 'TJK', 'KGZ', 'TKM'},
        'year_range':   (1991, 1999),
    },
    'modern_global': {
        'description': 'Modern Global Cluster — post-2000 comprehensive FTAs, '
                       'mid similarity (~35%), US/EU/Japan templates plus '
                       'recent South-South agreements.',
        'typical_parties': None,   # too heterogeneous to list
        'year_range':   (2000, 9999),
    },
    'central_asian': {
        'description': 'Central Asian Cluster — 52 short Goods FTAs among '
                       'former Soviet states, avg similarity 33.2%, '
                       'mostly no chapter structure.',
        'typical_parties': {'RUS', 'KAZ', 'UZB', 'TJK', 'KGZ', 'TKM',
                            'ARM', 'AZE', 'GEO', 'BLR', 'MDA', 'UKR'},
        'year_range':   (1991, 2005),
    },
}


def _parse_year(year_raw) -> int:
    """Safely parse year from string or int. Returns 0 if unparseable."""
    try:
        return int(str(year_raw)[:4])
    except (ValueError, TypeError, AttributeError):
        return 0


def classify_asymmetry(row) -> dict:
    """
    Classify one agreement for power asymmetry and template imposition.

    Parameters
    ----------
    row : dict or pandas Series
        Must contain: 'parties' (list of ISO codes), 'year' (str or int),
        'n_parties' (int), and optionally 'name' (str).

    Returns
    -------
    dict with keys:
        template_flag       str   label of detected template, or 'none'
        template_exporter   str   ISO code of template exporter, or None
        is_template         bool  True if any template detected
        is_us_tpa           bool  True if specifically US TPA post-2002
        is_eu_template      bool  True if EU/EC/EEC template
        is_plurilateral     bool  True if 3+ parties
        n_g20_parties       int   count of G20/major economies
        asymmetry_type      str   one of four categories (see below)
        asymmetry_note      str   human-readable explanation
        alschner_cluster    str   inferred cluster from Fig 3, or 'unknown'

    asymmetry_type values:
        'bilateral_symmetric'      Both or neither party is a major economy,
                                   no known template. Cleanest signal for CCS.
        'bilateral_asymmetric'     One major economy, or known template toward
                                   smaller partner.
        'plurilateral_template'    3+ parties with dominant template exporter.
        'plurilateral_symmetric'   3+ parties, no dominant template detected.
    """
    # ── Parse inputs ─────────────────────────────────────────────────────────
    parties_raw  = row.get('parties', []) or []
    parties      = set(p.upper().strip() for p in parties_raw if p)
    year         = _parse_year(row.get('year', 0))
    n_parties    = int(row.get('n_parties', len(parties)) or len(parties))
    name_lower   = str(row.get('name', '')).lower()

    is_plural = n_parties >= 3
    n_g20     = len(parties.intersection(G20_ISO))

    # ── Template detection ───────────────────────────────────────────────────
    template_flag     = 'none'
    template_exporter = None

    for iso, rules in TEMPLATE_RULES.items():
        if iso in parties:
            for rule in rules:
                if rule['min_year'] <= year <= rule['max_year']:
                    template_flag     = rule['label']
                    template_exporter = iso
                    break
        if template_flag != 'none':
            break

    # Secondary check: name-based US TPA verification
    if template_flag == 'none' and 'USA' in parties:
        for kw in US_TPA_AGREEMENTS:
            if kw in name_lower:
                template_flag     = 'us_tpa_template'
                template_exporter = 'USA'
                break

    is_template    = template_flag != 'none'
    is_us_tpa      = template_flag == 'us_tpa_template'
    is_eu_template = template_flag == 'eu_template'

    # ── Asymmetry classification ─────────────────────────────────────────────
    if is_plural and is_template:
        asym_type = 'plurilateral_template'
        asym_note = (
            f'{template_exporter} template imposed across {n_parties} parties '
            f'({template_flag})'
        )
    elif is_plural:
        asym_type = 'plurilateral_symmetric'
        asym_note = (
            f'{n_parties}-party agreement, no dominant template detected. '
            f'G20 members present: {n_g20}'
        )
    elif n_g20 >= 2 and not is_template:
        asym_type = 'bilateral_symmetric'
        asym_note = (
            'Both parties are major economies — more likely genuine '
            'bilateral negotiation.'
        )
    elif is_template:
        asym_type = 'bilateral_asymmetric'
        asym_note = (
            f'{template_exporter} template exported to smaller partner '
            f'({template_flag}). High CCS may reflect imposition, '
            f'not organic completeness.'
        )
    elif n_g20 == 1:
        asym_type = 'bilateral_asymmetric'
        asym_note = (
            'One major economy, one smaller partner — possible power '
            'asymmetry but no known template detected. Treat with caution.'
        )
    else:
        asym_type = 'bilateral_symmetric'
        asym_note = (
            'South-South or peer agreement — no asymmetry signals detected. '
            'CCS reflects genuine bilateral negotiation.'
        )

    # ── Alschner cluster inference ───────────────────────────────────────────
    alschner_cluster = 'unknown'
    if year >= 2000:
        alschner_cluster = 'modern_global'
    elif 1991 <= year <= 1999:
        ca_parties = ALSCHNER_CLUSTERS['central_asian']['typical_parties']
        ee_parties = ALSCHNER_CLUSTERS['eastern_european']['typical_parties']
        if len(parties.intersection(ca_parties)) >= 2:
            alschner_cluster = 'central_asian'
        elif len(parties.intersection(ee_parties)) >= 1:
            alschner_cluster = 'eastern_european'
        else:
            alschner_cluster = 'modern_global'
    elif year < 1991:
        alschner_cluster = 'modern_global'   # pre-1991 PTAs are misc

    return {
        'template_flag':     template_flag,
        'template_exporter': template_exporter,
        'is_template':       is_template,
        'is_us_tpa':         is_us_tpa,
        'is_eu_template':    is_eu_template,
        'is_plurilateral':   is_plural,
        'n_g20_parties':     n_g20,
        'asymmetry_type':    asym_type,
        'asymmetry_note':    asym_note,
        'alschner_cluster':  alschner_cluster,
    }


def add_asymmetry_flags(df_meta) -> object:
    """
    Apply classify_asymmetry() to every row of df_meta and merge
    the resulting flags back as new columns.

    Parameters
    ----------
    df_meta : pandas DataFrame
        Output of parse_tota_xml() loop. Must contain 'parties', 'year',
        'n_parties' columns.

    Returns
    -------
    pandas DataFrame
        df_meta with 9 new columns appended:
        template_flag, template_exporter, is_template, is_us_tpa,
        is_eu_template, is_plurilateral, n_g20_parties,
        asymmetry_type, asymmetry_note, alschner_cluster
    """
    import pandas as pd

    flags_list = df_meta.apply(classify_asymmetry, axis=1).tolist()
    df_flags   = pd.DataFrame(flags_list, index=df_meta.index)

    # Drop columns if they already exist (idempotent)
    existing = [c for c in df_flags.columns if c in df_meta.columns]
    df_out   = df_meta.drop(columns=existing).join(df_flags)

    return df_out


def asymmetry_summary(df_meta) -> None:
    """
    Print a summary of asymmetry flag distribution in the corpus.
    Call after add_asymmetry_flags().
    """
    if 'asymmetry_type' not in df_meta.columns:
        print("Run add_asymmetry_flags(df_meta) first.")
        return

    print("=" * 60)
    print("POWER ASYMMETRY FLAG SUMMARY")
    print("=" * 60)

    print("\nAsymmetry type distribution:")
    for atype, count in df_meta['asymmetry_type'].value_counts().items():
        pct = 100 * count / len(df_meta)
        print(f"  {atype:<30}  {count:>4}  ({pct:.1f}%)")

    print("\nTemplate flag distribution:")
    for flag, count in df_meta['template_flag'].value_counts().items():
        pct = 100 * count / len(df_meta)
        print(f"  {flag:<25}  {count:>4}  ({pct:.1f}%)")

    print("\nAlschner cluster distribution:")
    for cluster, count in df_meta['alschner_cluster'].value_counts().items():
        pct = 100 * count / len(df_meta)
        print(f"  {cluster:<25}  {count:>4}  ({pct:.1f}%)")

    n_template = df_meta['is_template'].sum()
    n_us_tpa   = df_meta['is_us_tpa'].sum()
    n_eu       = df_meta['is_eu_template'].sum()
    n_plural   = df_meta['is_plurilateral'].sum()
    n_sym      = (df_meta['asymmetry_type'] == 'bilateral_symmetric').sum()

    print(f"\nKey counts:")
    print(f"  Any template flag      : {n_template:>4} / {len(df_meta)}")
    print(f"  US TPA template        : {n_us_tpa:>4}")
    print(f"  EU template            : {n_eu:>4}")
    print(f"  Plurilateral           : {n_plural:>4}")
    print(f"  Bilateral symmetric    : {n_sym:>4}  ← cleanest CCS signal")
    print("=" * 60)


def regression_controls(df_cci, df_meta) -> object:
    """
    Merge asymmetry flags from df_meta into df_cci for regression use.

    Returns df_cci with asymmetry columns added. Creates dummy variables
    ready for OLS/panel regression.

    Dummy variables added:
        d_us_tpa, d_eu_template, d_other_template, d_plurilateral,
        d_symmetric, d_eastern_european, d_central_asian, d_modern_global
    """
    import pandas as pd

    flag_cols = [
        'pta_id', 'template_flag', 'template_exporter',
        'is_template', 'is_us_tpa', 'is_eu_template',
        'is_plurilateral', 'n_g20_parties',
        'asymmetry_type', 'alschner_cluster',
    ]
    available = [c for c in flag_cols if c in df_meta.columns]
    df_merged = df_cci.merge(df_meta[available], on='pta_id', how='left')

    # Binary dummies
    df_merged['d_us_tpa']           = df_merged['is_us_tpa'].astype(int)
    df_merged['d_eu_template']      = df_merged['is_eu_template'].astype(int)
    df_merged['d_other_template']   = (
        df_merged['is_template'] &
        ~df_merged['is_us_tpa'] &
        ~df_merged['is_eu_template']
    ).astype(int)
    df_merged['d_plurilateral']     = df_merged['is_plurilateral'].astype(int)
    df_merged['d_symmetric']        = (
        df_merged['asymmetry_type'] == 'bilateral_symmetric'
    ).astype(int)
    df_merged['d_eastern_european'] = (
        df_merged['alschner_cluster'] == 'eastern_european'
    ).astype(int)
    df_merged['d_central_asian']    = (
        df_merged['alschner_cluster'] == 'central_asian'
    ).astype(int)
    df_merged['d_modern_global']    = (
        df_merged['alschner_cluster'] == 'modern_global'
    ).astype(int)

    return df_merged


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import pandas as pd

    sample = pd.DataFrame([
        {'pta_id': 1,  'name': 'US-Chile FTA',       'parties': ['USA','CHL'], 'year': '2003', 'n_parties': 2},
        {'pta_id': 2,  'name': 'NAFTA',               'parties': ['USA','CAN','MEX'], 'year': '1992', 'n_parties': 3},
        {'pta_id': 3,  'name': 'EU-Morocco AA',       'parties': ['EU','MAR'], 'year': '1996', 'n_parties': 2},
        {'pta_id': 4,  'name': 'TPP',                 'parties': ['USA','JPN','AUS','NZL','CAN','MEX','SGP','VNM','MYS','BRN','CHL','PER'], 'year': '2016', 'n_parties': 12},
        {'pta_id': 5,  'name': 'Chile-Peru FTA',      'parties': ['CHL','PER'], 'year': '2009', 'n_parties': 2},
        {'pta_id': 6,  'name': 'Japan-Thailand EPA',  'parties': ['JPN','THA'], 'year': '2007', 'n_parties': 2},
        {'pta_id': 7,  'name': 'Russia-Belarus CU',   'parties': ['RUS','BLR'], 'year': '1995', 'n_parties': 2},
        {'pta_id': 8,  'name': 'US-Korea FTA',        'parties': ['USA','KOR'], 'year': '2007', 'n_parties': 2},
        {'pta_id': 9,  'name': 'Singapore-NZ FTA',    'parties': ['SGP','NZL'], 'year': '2001', 'n_parties': 2},
        {'pta_id': 10, 'name': 'India-ASEAN FTA',     'parties': ['IND','THA','MYS','VNM','IDN','PHL','SGP','BRN','KHM','LAO','MMR'], 'year': '2009', 'n_parties': 11},
    ])

    df_out = add_asymmetry_flags(sample)
    asymmetry_summary(df_out)

    print("\nDetailed flags:")
    cols = ['name', 'template_flag', 'asymmetry_type', 'alschner_cluster', 'n_g20_parties']
    print(df_out[cols].to_string(index=False))
