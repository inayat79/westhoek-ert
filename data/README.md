# Data folders

The input data are not stored in this repository. Each comes from its own record; put the files at the paths below and the
scripts find them. `python scripts/run_all.py --check-inputs` lists every input as in place (with its recorded checksum),
missing (with how to obtain it) or restricted.

| folder | content | source | licence | in this repository |
|---|---|---|---|---|
| `data/raw/` | 2018 land and marine ERT data of the K1 line: `land_data.dat`, `sea_data.dat`, `elec.dat`, `GPS_coordinate.xlsx`, `Read_me.txt`, `Westhoek_field_description.pdf` | Hermans & Paepen (2019), doi:10.14284/390 | as stated in the record | no — download the record and unpack it here |
| `data/raw/dp_zenodo_20792093/` | CPT-EC resistivity logs (2024) and the 2025 ERT line (`DP_R.xzr`, `ERT.dat`, …) | Vrancken, Paepen & Hermans (2026), doi:10.5281/zenodo.20792093 | CC BY 4.0 | no — fetched by `python scripts/27_dp_groundtruth.py fetch` |
| `data/raw/cpt_vrancken_2024/` | mechanical CPT logs (qc, fs, Rf) and surveyed positions of the soundings of 23 September 2024 (`CPT.zip` and its extracted `CPT/`) | E. Vrancken, Ghent University | **not public** | no — to be published by the data owners |
| `data/raw/Project12_LS210110079_2018-03-07_11-22-23/` | 2018 Terrameter LS instrument project of the K1 land line (`project.db`) | T. Hermans, Ghent University | **not public** | no |
| `data/external/natural_earth/` | coastline, land and borders, 1:10m | Natural Earth | public domain | no — fetched by `python scripts/run_all.py --fetch-natural-earth` |
| `data/external/basemap/` | Copernicus DEM GLO-30, EMODnet DTM 2024, Digitaal Vlaanderen orthophoto | see `basemap/README_sources.md` | see that file | README only — fetched by `python scripts/32_location_map.py fetch` |
| `data/external/lebbe1981_K1_digitised.csv` | 1980 borehole-log interpretation on profile K1, digitised from Lebbe (1981), Fig. 3 | this work | CC BY 4.0 (cite Lebbe, 1981, too) | yes |
| `data/processed/` | inversions, prior ensembles and every other intermediate product (about 3 GB) | written by the scripts | — | no — rebuilt by `scripts/run_all.py --from-raw` |

File sizes and checksums of every input: [`raw/README_sources.md`](raw/README_sources.md).

## The restricted inputs

The mechanical CPT logs, the surveyed sounding positions and the 2018 instrument project were provided by their owners
for this analysis and may not be redistributed. No raw values and no per-depth or per-datum values derived from them are in
this repository or its history, and neither are the plotted values of Figures S1 (CPT logs) and S2 (instrument errors).
Only summary values that the paper prints (for example medians, counts and the projected positions of SI Table S1) appear,
in `results/paper/numbers.md`.
`results/figs_wrr_v5/data/Figure1.csv` gives the soundings at their positions projected on the K1 line (SI Table S1), not
their map coordinates.

Steps that need them stop with `restricted input missing: <path> (…). Needed for … See data/README.md`:

| restricted input | expected at | needed for |
|---|---|---|
| mechanical CPTs and surveyed positions | `data/raw/cpt_vrancken_2024/CPT.zip` and `CPT/` (`Location.txt`, `Proef2.txt` … `Proef8.txt`) | `27_dp_groundtruth.py cpt` (soil-behaviour classes, clean-sand filter, E21, SI Table S1, Figure S1), its product `data/processed/v1_4/cpt.json`, and through it every comparison with the soundings (`27 … samedate/dp/report`, `28`–`31`, `32_location_map.py`, `18_paper.py`, `33_review_v20.py`) |
| 2018 instrument project | `data/raw/Project12_LS210110079_2018-03-07_11-22-23/project.db` | `26_raw_abem.py` (SI Table S2, Figure S2, E13, E14) and the GPS datum check in `27_dp_groundtruth.py run` |

When the owners publish these data, place the files at these paths (the checksums in `raw/README_sources.md` identify the
versions used) and rerun the steps above.
