"""
trade_stopwords.py
==================
Domain-specific stopword lists for NLP preprocessing of
Preferential Trade Agreement (PTA) texts.

Built for the WTO/ToTA corpus (448 PTAs, Alschner et al. 2018).
Designed to work alongside the Alschner segmentation pipeline:
  chapter → article → header → full_text (XML format).

STRUCTURE
---------
Four distinct layers, each importable separately:

  LAYER_0_ENGLISH_BASE      — standard English function words (spaCy-derived,
                               pruned of legally significant terms)
  LAYER_1_LEGAL_BOILERPLATE — high-frequency legal formulas with no
                               discriminatory power across agreements
  LAYER_2_TRADE_STRUCTURAL  — trade agreement formatting & procedural words
                               (structural noise, not substantive content)
  LAYER_3_PROTECT           — NEVER stopword these; they are CCS index signals

USAGE
-----
    from trade_stopwords import get_stopwords, LAYER_3_PROTECT

    # Default: all three removal layers combined
    sw = get_stopwords()

    # Check nothing from the protected list leaked in
    assert not sw.intersection(LAYER_3_PROTECT), "Protected terms in stopword set!"

    # Extend with your own additions
    sw = get_stopwords(extra={"pursuant", "aforementioned"})

    # Use only legal boilerplate layer (e.g. for a lighter pass)
    sw = get_stopwords(layers=[1, 2])

NOTES ON DESIGN CHOICES
------------------------
- "whereas" is in LAYER_1 (boilerplate) because in preambles it is
  formulaic. However, if you are analysing preamble vs. body separately,
  you may want to keep it for preamble-tone analysis.
- "annex", "schedule", "appendix" are in LAYER_2 as structural words.
  If you are *including* annexes in your CCS (recommended for robustness
  check), remove these from the active stopword set.
- All words are lowercase. Apply .lower() before filtering.
- This list is intentionally conservative. When in doubt, we do NOT
  stopword — false removals destroy signal; false retentions only add
  minor noise.

REFERENCES
----------
Alschner, W., Seiermann, J., & Skougarevskiy, D. (2018).
  Text-as-data analysis of preferential trade agreements.
  Journal of International Economic Law, 21(1), 1–21.
LexNLP legal stopwords: github.com/LexPredict/lexpredict-lexnlp
spaCy English stopwords: spacy.io/api/language#defaults
USPTO patent stopwords: arxiv.org/abs/2006.02633
"""

# ===========================================================================
# LAYER 0 — Standard English function words
# Source: spaCy en_core_web stopwords, pruned of legally significant terms.
# Removed from standard list: "not", "no", "nor", "never", "unless",
# "except", "if", "without", "whether", "provided", "further", "other",
# "each", "every", "all", "both", "any", "only", "also", "as", "than",
# "until", "before", "after", "between", "within", "beyond", "against"
# — these carry meaning in legal conditionality and scope clauses.
# ===========================================================================

LAYER_0_ENGLISH_BASE = {
    # Articles & determiners
    "a", "an", "the",
    # Basic conjunctions (safe ones — "nor", "unless" removed)
    "and", "but", "or", "yet", "so",
    # Pronouns
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "who", "whom", "whose",
    # Prepositions (safe ones — directional/conditional ones kept)
    "of", "in", "to", "for", "on", "at", "by", "with", "about",
    "into", "through", "during", "above", "below", "from",
    "up", "down", "out", "off", "over", "under",
    # Auxiliaries (safe ones — "may", "shall", "must", "should" REMOVED)
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "would", "can", "could",
    # Common adverbs (safe to remove)
    "here", "there", "when", "how", "why",
    "very", "more", "most", "just", "now", "then", "again",
    "too", "quite", "rather", "already", "still", "once",
    # Demonstratives
    "this", "that", "these", "those",
    # Common filler words
    "accordingly", "thus", "hence", "therefore",
    "said", "same", "respectively",
}


# ===========================================================================
# LAYER 1 — Legal boilerplate with no discriminatory power
# These appear in nearly every PTA regardless of content or ambition.
# High corpus frequency, near-zero TF-IDF value across the 448 PTA corpus.
# Latin terms common in international legal drafting are included.
# ===========================================================================

LAYER_1_LEGAL_BOILERPLATE = {
    # Standard opening/closing formulas
    "whereas", "witnesseth", "now therefore",
    "in witness whereof", "done at", "done in",
    "in duplicate", "in triplicate",
    "signed", "concluded", "entered into",

    # Arcane referential language (zero content value)
    "hereof", "hereto", "hereunder", "hereafter", "hereby",
    "hereinafter", "hereinabove", "hereinbefore",
    "therein", "thereof", "thereto", "thereunder", "thereafter",
    "thereby", "therewith", "thereon",
    "wherein", "whereof", "whereto", "whereunder",
    "aforesaid", "aforementioned", "abovementioned",
    "above-mentioned", "aforegoing",

    # Latin legal phrases (boilerplate usage in PTAs)
    "inter alia", "mutatis mutandis", "sui generis", "de facto",
    "de jure", "ex officio", "prima facie", "bona fide",
    "ex ante", "ex post", "ad hoc", "pro rata",
    "vis-a-vis", "vis à vis", "et seq", "et al",
    "ibid", "idem", "supra", "infra",

    # Attestation and signature boilerplate
    "duly", "undersigned", "authorized", "authorised",
    "signatory", "signatories", "plenipotentiary", "plenipotentiaries",
    "ratification", "ratify", "ratified", "ratifying",
    "accession", "accede", "acceded",
    "entry into force", "come into force", "enter into force",

    # General legal reference words (no substantive content)
    "pursuant to", "pursuant", "in accordance with", "in accordance",
    "in conformity with", "consistent with",
    "notwithstanding the foregoing",  # "notwithstanding" alone is PROTECTED
    "as the case may be",
    "as appropriate",
    "as applicable",

    "in general",
    "inter alia",

    # Document reference words (structural, not substantive)
    "see", "refer", "referenced", "referring",
    "set forth", "set out", "laid down", "laid out",
    "contained in", "referred to in",
    "as defined in", "as provided in", "as specified in",
    "as described in", "as mentioned in",
    "as established in", "as determined in",
}


# ===========================================================================
# LAYER 2 — Trade agreement structural & procedural noise
# Formatting words, procedural terms, and WTO/PTA boilerplate that appear
# uniformly across all agreements — they describe *where* text is located
# or *how* the document is organized, not *what* it commits to.
#
# NOTE: "annex", "schedule", "appendix" are included here because the
# Alschner pipeline strips annexes. If you run a version WITH annexes,
# comment out that sub-group and re-run your CCS comparison.
# ===========================================================================

LAYER_2_TRADE_STRUCTURAL = {
    # Document structure words
    "chapter", "article", "paragraph", "subparagraph",
    "clause", "sub-clause", "subclause",
    "section", "subsection", "sub-section",
    "part", "title", "preamble",
    "page", "pages",

    # Annex/schedule group (see note above — comment out if including annexes)
    "annex", "annexes", "annex i", "annex ii", "annex iii",
    "schedule", "schedules", "appendix", "appendices",
    "attachment", "attachments", "exhibit", "exhibits",

    # Numbering & labelling
    "number", "numbers", "no.", "nos.",
    "item", "items", "list", "lists",
    "table", "tables", "figure", "figures",
    "footnote", "footnotes",

    # Standard procedural actors (appear in every PTA)
    "parties", "party", "member", "members",
    "contracting party", "contracting parties",
    "signatory parties", "signatory party",
    "member state", "member states",
    "the parties", "the party",

    # WTO/organization name boilerplate
    "world trade organization", "wto",
    "general agreement", "gatt", "gats", "trips",
    "world trade", "multilateral",
    "regional trade agreement", "free trade agreement",
    "preferential trade agreement",
    "pta", "rta", "fta",

    # Procedural/temporal markers (administrative, not substantive)
    "date", "dates", "dated",
    "year", "years", "month", "months", "day", "days",
    "calendar", "fiscal",
    "enter", "entry",
    "force", "effect", "effective",
    "amendment", "amendments", "amend", "amended",
    "modification", "modifications", "modify", "modified",
    "revision", "revisions", "revise", "revised",
    "review", "reviews",  # BUT "reviewed" can signal enforcement — watch
    "renewal", "renew", "renewed",
    "termination", "terminate", "terminated",
    "withdrawal", "withdraw", "withdrawn",
    "notification", "notify", "notified", "notifying",
    "communication", "communicate",
    "publish", "published",
    "deposit", "depositary", "deposited",

    # Generic administrative verbs (too diffuse to carry CCS signal)
    "agree", "agreed", "agreement",  # ironic, but too generic
    "establish", "established", "establishment",
    "constitute", "constituted", "constitution",
    "create", "created",
    "form", "formed", "formation",
    "adopt", "adopted", "adoption",
    "apply", "applied", "application",
    "implement", "implemented", "implementation",
    "determine", "determined", "determination",
    "consider", "considered", "consideration",
    "decide", "decided", "decision",
    "recognize", "recognised", "recognized", "recognition",
    "acknowledge", "acknowledged",
    "confirm", "confirmed", "confirmation",
    "note", "noted", "noting",
    "recall", "recalled", "recalling",

    # Country/territory reference fillers
    "territory", "territories", "territorial",
    "jurisdiction", "jurisdictions",
    "domestic", "national", "international",
    "bilateral", "plurilateral",

    # Common trade policy qualifiers (near-universal, low discriminatory value)
    "trade", "goods", "services",  # too broad — keep only in context
    # NOTE: if your pipeline extracts bigrams/phrases, pre-protect
    # "market access" (Layer 3) and "trade in services" (Layer 3) BEFORE
    # applying unigram stopwords, or these component words will be stripped.
    "market", "markets",
    "customs", "duty", "duties",
    "tariff", "tariffs",
    "quota", "quotas",

    # Procedural customs/administrative terms
    # Universal across ALL PTAs regardless of depth; near-zero TF-IDF
    # in the Alschner et al. (2018) corpus.
    # CIS Annex I–II and MSG Annex I devote pages to these procedures
    # identically in very shallow and deeper agreements alike.
    "re-export", "re-exportation", "re-exporting",
    "certificate of origin", "certificates of origin",
    "importation", "exportation",
    "competent authority", "competent authorities",
    "customs clearance", "customs formalities",
    "goods nomenclature", "harmonized system",  # HS classification references

    # Baseline trade policy terms — universal across the full ~450-agreement corpus
    # DESTA (Dür et al. 2014) codes MFN as present in >95% of all PTAs; it is
    # the floor, not a discriminator.  Quantitative restriction elimination follows
    # GATT Art. XI and is copy-pasted into even the shallowest 1-article FTAs.
    "most favoured nation", "most-favoured-nation",
    "most favored nation", "most-favored-nation",
    "mfn",
    "quantitative restriction", "quantitative restrictions",

    # Terms moved here from Layer 3 after full-corpus reassessment:
    # At 450-agreement scale these approach near-zero TF-IDF —
    # they appear in the vast majority of PTAs regardless of depth.

    # "concession"/"concessions": every PTA — even a 3-product positive-list
    # agreement — mentions "tariff concessions".  Coverage breadth matters but
    # the word itself does not discriminate; use schedule line-counts instead.
    "concession", "concessions",

    # "customs union": GATT Art. XXIV boilerplate is copy-pasted into most FTAs
    # ("consistent with forming a free trade area or customs union") regardless of
    # whether the agreement has any ambition toward a CU.
    "customs union",

    # "balance of payments": standard GATT Art. XII/XVIII carve-out language
    # present in ~90 % of all PTAs.  The exception itself is near-universal;
    # its SCOPE and conditionality are captured by surrounding Layer-3 terms
    # ("shall", "notwithstanding", "subject to") rather than by the phrase alone.
    "balance of payments", "balance-of-payments",
}


# ===========================================================================
# LAYER 3 — PROTECTED TERMS
# These are CCS index signals. NEVER add to the active stopword set.
# They measure: obligation density, conditionality, precision, enforcement.
#
# This set is used for validation only — run an assertion check to ensure
# none of these appear in your active stopword set.
# ===========================================================================

LAYER_3_PROTECT = {
    # ---- Obligation markers (hard) ----
    "shall", "must", "is required", "are required",
    "required to", "obliged to", "obligated to",
    "undertakes to", "undertake to", "undertakes",
    "commits to", "commit to", "commits",
    "binds", "bound to", "bound by",

    # ---- Soft-language / best-endeavour markers ----
    "should", "endeavour", "endeavor",
    "endeavours", "endeavors",
    "encourage", "encourages", "encouraged",
    "promote", "promotes", "promoted",
    "seek to", "seeks to",
    "strive", "strives", "strive to",
    "aim", "aims", "aim to",
    "facilitate", "facilitates",
    "cooperate", "cooperates", "cooperation",
    "recommend", "recommends",

    # ---- Conditionality triggers ----
    "if", "unless", "except", "except where",
    "provided that", "provided however",
    "subject to", "notwithstanding",
    "without prejudice to",
    "in the event", "in the event that",
    "where", "where applicable",
    "in so far as", "insofar as",
    "to the extent that", "to the extent",

    # ---- Scope qualifiers (precision indicators) ----
    "all", "each", "every", "any", "no",
    "only", "solely", "exclusively",
    "generally", "specifically",
    "not", "nor", "never",
    "both", "either", "neither",
    "other", "otherwise",
    "further", "furthermore",
    "also", "in addition", "additionally",
    "including", "includes", "include",
    "such as", "namely", "in particular",
    "without limitation", "without limiting",

    # ---- Enforcement & dispute settlement ----
    "remedy", "remedies", "remedial",
    "sanction", "sanctions",
    "penalty", "penalties", "penalize", "penalise",
    "compensation", "compensate", "compensatory",
    "retaliate", "retaliation", "retaliatory",
    "suspend", "suspension", "suspended",
    "safeguard", "safeguards",
    "countermeasure", "countermeasures",
    "enforce", "enforcement", "enforced", "enforceable",
    "comply", "compliance", "complies", "non-compliance",
    "violation", "violate", "violates",
    "breach", "breaches",
    "arbitration", "arbitral", "arbitrator",
    "panel", "panels", "panelist",
    "appellate", "appeal", "appeals",
    "ruling", "rulings",
    "finding", "findings",
    "dispute", "disputes",
    "settlement", "settle", "settles",
    "consultations", "consult", "consults",
    "good offices", "mediation", "mediator",
    "binding", "non-binding",
    "final", "finality",

    # ---- Legal precision markers ----
    "define", "defined", "definition", "definitions",
    "means", "meaning",
    "refers", "reference",
    "herein", # debated — keep protected since "as defined herein" scopes obligations
    "timeline", "timelines", "timeframe", "time frame",
    "within", "no later than", "not later than",
    "deadline", "deadlines",
    "period", "periods",

    # ---- Issue area coverage (binary provision indicators) ----
    "investment", "investor", "investors",
    "intellectual property", "copyright", "patent", "trademark",
    "competition", "anti-competitive", "monopoly",
    "procurement", "procuring entity",
    "labor", "labour", "workers", "employment",
    "environment", "environmental",
    "data", "privacy", "personal data",
    "digital", "electronic commerce", "e-commerce",
    "financial services", "banking", "insurance",
    "telecommunications", "telecom",
    "transparency", "publication",
    "regulatory", "regulation", "regulations",
    "standard", "standards", "standardization",
    "technical barrier", "tbt",
    "sanitary", "phytosanitary", "sps",
    "rules of origin", "origin",
    "subsidies", "subsidy", "countervailing",
    "anti-dumping", "dumping",
    "state-owned enterprise", "soe",
    "currency", "exchange rate", "monetary",

    # ---- Deep integration markers (Horn, Mavroidis & Sapir 2010; Dür et al. 2014) ----
    # These terms discriminate shallow from deep PTAs in the WTO/ToTA corpus.
    # MSG (21 arts, goods-only positive list) lacks virtually all of these;
    # CIS Protocol and EFTA-style agreements contain them with increasing density.

    # WTO+ core obligations (Horn, Mavroidis & Sapir 2010, Table 1)
    "national treatment",                       # NT = canonical WTO+ provision; absent in shallowest PTAs
    "market access",                            # DESTA depth dimension 1 (Dür, Baccini & Elsig 2014)

    # Regulatory convergence / behind-the-border depth
    # (Hofmann, Osnago & Ruta 2019 deep trade agreements taxonomy)
    "harmonization", "harmonize", "harmonise",
    "harmonisation", "harmonized", "harmonised",
    "mutual recognition",                       # CIS Protocol Art. 12; absent in MSG entirely

    # Market-opening depth beyond tariffs
    "liberalization", "liberalize", "liberalise",
    "liberalisation",
    "non-tariff",                               # NTB scope separates shallow/deep (DESTA coding)
    "trade facilitation",                       # Distinct modern depth dimension (WTO TFA 2013)

    # Commitment precision / lock-in signals
    "standstill",                               # Prevents rollback; marks legally binding schedules
    # NOTE: "concession"/"concessions", "customs union", "balance of payments"
    # were removed from this section after full-corpus reassessment — they are
    # near-universal across the 450-agreement WTO/ToTA corpus and are now in Layer 2.

    # Investment chapter markers — WTO-X provisions (Horn et al. 2010 WTO-X list)
    "expropriation",
    "fair and equitable treatment",
    "investor-state",                           # ISDS clause; only in agreements with full investment chapters

    # Services liberalization depth markers
    # (Hofmann, Osnago & Ruta 2019; WTO GATS framework)
    # DESTA codes services presence/depth separately from goods; these terms
    # distinguish goods-only PTAs (the majority of shallow agreements) from
    # ones with genuine services commitments — at 450-agreement scale this is
    # a strong signal even after controlling for agreement length.
    "trade in services",                        # Explicit services chapter bigram; absent in goods-only PTAs
    "mode of supply", "modes of supply",        # GATS Art. I modes 1–4; only in serious services chapters
    "positive list", "negative list",           # Scheduling approach: negative-list = deeper lock-in

    # Regulatory depth — post-2000 deep trade agreement language
    # (Hofmann, Osnago & Ruta 2019 "deep trade agreements" taxonomy;
    #  Mattli & Büthe 2003 on regulatory governance)
    # These phrases are absent from essentially all pre-2000 shallow FTAs
    # and from goods-only agreements like MSG/CIS (1994).
    "regulatory coherence",
    "regulatory cooperation",                   # Broader than "cooperation" alone; signals behind-the-border depth

    # Ratchet / lock-in mechanism
    # A ratchet clause prevents parties from rolling back autonomous
    # liberalization already granted above their scheduled commitment level.
    # It is a marker of a legally precise, deeply binding agreement.
    #
    # PRECISION vs. RECALL trade-off:
    # - Precision: when "ratchet" appears, it reliably signals a deep FTA.
    #   Concentrated in post-2000 US-template and CPTPP-style agreements
    #   (e.g., NAFTA Art. 1206, CPTPP Ch. 10).  Absent from all shallow
    #   goods-only PTAs (MSG, early CIS, etc.).
    # - Recall: POOR.  Many deep agreements implement the same mechanism
    #   without the word "ratchet" — via negative-list scheduling,
    #   "standstill" clauses (already in Layer 3), or structural binding
    #   through schedule annexes.  EU comprehensive FTAs achieve ratchet
    #   effects through different drafting conventions.
    #
    # Net assessment: keep in Layer 3 (never strip — if it appears, it IS
    # signal), but do not expect it to be a high-frequency discriminator.
    # The CCS for deep agreements will primarily be driven by the broader
    # set of Layer-3 terms above; "ratchet" contributes marginally at the
    # tail of the depth distribution.
    "ratchet",
}


# ===========================================================================
# PUBLIC API
# ===========================================================================

def get_stopwords(
    layers: list = None,
    extra: set = None,
    validate: bool = True
) -> set:
    """
    Return the active stopword set for PTA preprocessing.

    Parameters
    ----------
    layers : list of int, optional
        Which layers to include. Default is [0, 1, 2] (all removal layers).
        Options: 0 = English base, 1 = legal boilerplate, 2 = trade structural.
        Do NOT include 3 — Layer 3 is the protected set, not for removal.

    extra : set, optional
        Additional domain-specific words to add to the stopword set.
        These are NOT checked against LAYER_3_PROTECT — you are responsible
        for not adding protected terms here.

    validate : bool, optional
        If True (default), raises AssertionError if any LAYER_3_PROTECT term
        ends up in the final stopword set. Highly recommended to keep True.

    Returns
    -------
    set of str
        Lowercase stopword set ready for token filtering.

    Examples
    --------
    >>> sw = get_stopwords()
    >>> tokens = ["the", "parties", "shall", "endeavour", "to", "cooperate"]
    >>> filtered = [t for t in tokens if t.lower() not in sw]
    >>> # filtered = ["shall", "endeavour", "cooperate"]
    # Note: "to" removed (Layer 0), "the"/"parties" removed (Layer 0/2)
    # "shall", "endeavour", "cooperate" RETAINED (Layer 3 protected)
    """
    if layers is None:
        layers = [0, 1, 2]

    layer_map = {
        0: LAYER_0_ENGLISH_BASE,
        1: LAYER_1_LEGAL_BOILERPLATE,
        2: LAYER_2_TRADE_STRUCTURAL,
    }

    active = set()
    for layer_id in layers:
        if layer_id not in layer_map:
            raise ValueError(
                f"Layer {layer_id} not valid. Choose from 0, 1, 2. "
                f"Layer 3 is LAYER_3_PROTECT and is never added to stopwords."
            )
        active.update(layer_map[layer_id])

    if extra:
        active.update({w.lower() for w in extra})

    if validate:
        leaked = active.intersection(LAYER_3_PROTECT)
        if leaked:
            raise AssertionError(
                f"Protected CCS signal terms found in stopword set: {leaked}\n"
                f"Remove these from your custom additions or check layer definitions."
            )

    return active


def describe_layers() -> None:
    """Print a summary of each layer with word counts."""
    print("=" * 60)
    print("TRADE AGREEMENT STOPWORD LAYERS — SUMMARY")
    print("=" * 60)
    layers = [
        ("Layer 0 — English base (pruned)",   LAYER_0_ENGLISH_BASE,      "REMOVE"),
        ("Layer 1 — Legal boilerplate",        LAYER_1_LEGAL_BOILERPLATE, "REMOVE"),
        ("Layer 2 — Trade structural noise",   LAYER_2_TRADE_STRUCTURAL,  "REMOVE"),
        ("Layer 3 — CCS signal terms (KEEP)",  LAYER_3_PROTECT,           "PROTECT"),
    ]
    for name, layer, action in layers:
        print(f"\n  {name}")
        print(f"  Action : {action}")
        print(f"  Words  : {len(layer)}")
    total_remove = len(LAYER_0_ENGLISH_BASE | LAYER_1_LEGAL_BOILERPLATE | LAYER_2_TRADE_STRUCTURAL)
    print(f"\n  Total active stopwords (all removal layers): {total_remove}")
    print(f"  Total protected CCS signal terms          : {len(LAYER_3_PROTECT)}")
    print("=" * 60)


def check_word(word: str) -> str:
    """
    Diagnostic: tell you which layer a given word belongs to.

    Parameters
    ----------
    word : str
        A word or phrase to look up (case-insensitive).

    Returns
    -------
    str
        Description of the layer(s) the word appears in.
    """
    w = word.lower().strip()
    found = []
    if w in LAYER_0_ENGLISH_BASE:
        found.append("Layer 0 — English base (REMOVED)")
    if w in LAYER_1_LEGAL_BOILERPLATE:
        found.append("Layer 1 — Legal boilerplate (REMOVED)")
    if w in LAYER_2_TRADE_STRUCTURAL:
        found.append("Layer 2 — Trade structural (REMOVED)")
    if w in LAYER_3_PROTECT:
        found.append("Layer 3 — CCS signal (PROTECTED — never remove)")
    if not found:
        return f"'{word}' not found in any layer (treat as content word, keep by default)"
    return f"'{word}' → " + "; ".join(found)


# ===========================================================================
# QUICK DEMO — run python trade_stopwords.py
# ===========================================================================

if __name__ == "__main__":
    describe_layers()

    print("\n--- Layer check examples ---")
    test_words = [
        "shall", "must", "whereas", "hereinafter", "parties",
        "article", "endeavour", "notwithstanding", "arbitration",
        "annex", "chapter", "the", "dispute", "compliance",
        "provided that", "subject to", "mutatis mutandis",
        "territory", "investment", "wto", "remedy", "tariff",
        # New additions — should all show Layer 3 (PROTECTED)
        "national treatment", "market access", "harmonization",
        "liberalization", "non-tariff", "trade facilitation",
        "mutual recognition", "standstill", "expropriation",
        "fair and equitable treatment", "customs union",
        "concession", "balance of payments",
        # New additions — should show Layer 2 (REMOVED)
        "re-export", "re-exportation", "certificate of origin",
        "importation", "exportation", "competent authority",
    ]
    for w in test_words:
        print(f"  {check_word(w)}")

    print("\n--- Filtering demo ---")
    sample_sentence = (
        "the parties shall endeavour to ensure compliance with "
        "the provisions set forth herein and shall establish "
        "an arbitration panel pursuant to annex i of this agreement"
    )
    sw = get_stopwords()
    tokens = sample_sentence.lower().split()
    filtered = [t for t in tokens if t not in sw]
    print(f"  Original : {sample_sentence}")
    print(f"  Filtered : {' '.join(filtered)}")
    print(f"  (Kept 'shall', 'endeavour', 'compliance', 'provisions',")
    print(f"   'arbitration', 'panel' — removed structural/boilerplate noise)")
