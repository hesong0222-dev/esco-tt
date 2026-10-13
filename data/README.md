# Real data: H3N2 influenza, 3 demes (MASCOT tutorial)

Source: Taming the BEAST "Mascot-Tutorial" (Nicola F. Müller et al.), GitHub
https://github.com/Taming-the-BEAST/Mascot-Tutorial, commit 6d5b9b8e19ee78fcacc3f3632903672d759523cd (2025-06-02).
License: CC BY 4.0 (`raw/LICENSE`, "Attribution 4.0 International"). Downloaded 2026-10-05.

24 H3N2 HA sequences sampled 2000.005–2001.992 from Hong Kong (10), New York (6), New Zealand (8).
Deme indices used here: 0 = Hong_Kong, 1 = New_York, 2 = New_Zealand (from the last `|` field of the taxon label).

| file (raw/) | upstream path | sha256 |
|---|---|---|
| H3N2.mcc.tree | precooked_runs/H3N2.mcc.tree | 155cfdfd5f2455c6fb19c581d36eabf7d82f684ec97860e27c76dd12cbd676c1 |
| H3N2.ccd0.tree | precooked_runs/H3N2.ccd0.tree | 3ebc36ef7543df3ff055130d21a2cbdfff61c01eed4a21c5e471b8cedc3cb864 |
| H3N2.trees (401 posterior trees) | precooked_runs/H3N2.trees | af0eb00f6390868fbacc4100428f5a7b873059bab96548d378091efb8f850f40 |
| H3N2.log (BEAST2 + MASCOT v3 log) | precooked_runs/H3N2.log | 3a06c406ca2e2b2fe87aab95080fbfe40343a8e259c8199150555e0f95330603 |
| H3N2.nexus (alignment) | data/H3N2.nexus | 0f1124a1b859ce60283d74c31a760b25394d7e022250a2ab29c2ac3a952f6192 |
| H3N2.xml (BEAST2 analysis) | xml/H3N2.xml | 41edabcac1b042a1714450a472ef310d03706474e38bbdb0379076ce315e5032 |

## Fixed tree used
`h3n2_mcc.json` = the tutorial's precooked MCC tree (`TREE_MCC_median`, TreeAnnotator median heights), produced by
`../.venv/bin/python convert.py raw/H3N2.mcc.tree TREE_MCC_median h3n2_mcc.json`. No tree was re-estimated here.
Times are heights above the most recent tip (2001.991781), in years; the tip heights reproduce the label dates exactly
(max error 0). 24 samples, 23 coalescences, root height 3.553 y, peak concurrent lineages 12 (3^12 = 531k states, so
the dense exact likelihood is feasible). Posterior trees: peak lineages 11–17 (most 13).

## Convention check against BEAST's MASCOT
For 10 posterior samples, our `approx/mascot.py` with the logged parameters (M[a][b] = `b_migrationConstant.a_to_b`,
Ne = `NeConstant.a`) reproduces BEAST's logged `Mascot` density to within 0.003–0.46 log units (mean |Δ| ≈ 0.16);
transposing M is worse in 9/10 (up to 5.8 units) and doubling Ne in 10/10 (0.8–8.6), so the convention is right. BEAST's logged values differ from our converged
solution (DOP853, rtol 1e-10) because of BEAST 2 MASCOT's integrator; see bench/mascot_vs_beast.md and https://github.com/CompEvol/Mascot/issues/27.

# Real data 2: avian influenza H7 HA, 133 taxa, 5 host demes (slow-migration candidate)

Used by `../bench/realdata2.md`. Downloaded 2026-10-06.

**Trees** — Müller, Rasmussen & Stadler (2017, MBE) "The structured coalescent and its approximations", GitHub
https://github.com/nicfel/The-Structured-Coalescent, commit 81b647f811f84a1c038e2454d7ad2e86452f02c9 (2017-03-27),
folder `AIV/`. License (repo README): "The content of this project itself is licensed under the Creative Commons
Attribution 3.0 license" (CC BY 3.0); the java source is GPL. The three MCC trees are BEAST2 summaries of their AIV
analyses with 7 *location* demes under three structured-coalescent approximations (masco = the MASCOT precursor,
sisco, lisco); all three contain the same 133 HA sequences. We use the trees only (topology + node times), not their
location labels.

**Host demes** — BASTA (De Maio, Wu, O'Reilly & Wilson 2015, PLoS Genet), Bitbucket
https://bitbucket.org/nicofmay/basta-bayesian-structured-coalescent-approximation, commit
24df9746ab5b1be87bded5625df8d96392af692f (2022-12-26), `examples/AIV_HA_BASTA_host_asymmetric.xml`. Repository
license GPL-3.0 (`raw2/basta_COPYING`). The `typeTraitSet` assigns each of the same 133 taxa to one of 5 host groups:
ans (Anseriformes, 93 tips), cha (Charadriiformes, 30), gal (Galliformes, 6), pas (Passeriformes, 2), MexicoGal (2012
Mexican poultry outbreak, 2). The same host grouping was used by Roberts et al. (2025, PLoS Comput Biol 21:e1012995).
Deme indices here: 0 ans, 1 cha, 2 gal, 3 pas, 4 MexicoGal. Taxa are matched by the name before the first `|`
(all 133 match; asserted in `convert_aiv.py`).

| file (raw2/) | upstream path | sha256 |
|---|---|---|
| aiv_structcoal_masco_mcc.trees | AIV/structcoal/masco_mcc.trees | 86e7207b1f039d374e758446479d4afd83ceb5a82abfbe972e9739dcc888512d |
| aiv_structcoal_sisco_mcc.trees | AIV/structcoal/sisco_mcc.trees | 0e0ac20b1e63d5b31df290ba335bc9f02adf87a9ddb8a4745b2f4b7bf3163987 |
| aiv_structcoal_lisco_mcc.trees | AIV/structcoal/lisco_mcc.trees | 6d79d16962fd5430d49371456b40ad9e8a6551672dff5233f98ebfc50fa5d94d |
| aiv_structcoal_AIV.xml | AIV/structcoal/AIV.xml | c5ce4bd0db5a77ab4668370d0df1dcfe2eb4c121b5a4ca4df44e7171a139ba29 |
| aiv_AIV_1masco.log | AIV/structcoal/individual/AIV_1masco.log | ddd5c4546ecdf3282802eb06ea853df66f5897c3b56ff92821a0544c5d1504da |
| aiv_README.md | AIV/README.md | e514fe851d619435a470470ef6087eb5e5b8d836aaf73410c889ed646a1460d6 |
| basta_AIV_HA_BASTA_host_asymmetric.xml | examples/AIV_HA_BASTA_host_asymmetric.xml | 85a5fdd20cfe8d85b44d01ca13295239602ba904bf8a01086a5ded022220ca0f |
| basta_COPYING | COPYING | 8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903 |

Conversion (no tree re-estimated): `../.venv/bin/python convert_aiv.py raw2/aiv_structcoal_<m>_mcc.trees
raw2/basta_AIV_HA_BASTA_host_asymmetric.xml aivhost_<m>_mcc.json` for m in masco, sisco, lisco. Branch lengths are
those written in the MCC newick. Tip heights reproduce the label dates to < 1e-8 y. Root height 12.29–12.33 y; peak
concurrent lineages 46 (masco), 43 (sisco), 40 (lisco), so 5^46 ≈ 1e32 states: dense is impossible, TT only.
Fitch parsimony on the masco tree needs 8 host changes (root ans), as reported by Roberts et al.
The 7-location version of the same trees (`aiv7_<m>_mcc.json`, `convert.py ... alaska,northwest,northeast,southeast,
eastcoast,northmideast,center`) was converted only to look at its regime; it was not analysed further.
