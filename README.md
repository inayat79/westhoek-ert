# Westhoek ERT interface uncertainty

**Code and data for: *Testing Coastal ERT Uncertainty Workflows for Saline Interface Depths Against Independent Soundings*
(submitted to Water Resources Research).**

Repository: https://github.com/inayat79/westhoek-ert

This is a **code repository**. It holds the analysis scripts, the software environment, the pre-registration (decision log),
the numbers reported in the paper and the values plotted in each figure. The field data are not stored here; they come from
their own records (see [Data](#data)).

Please cite the archived version:

> Rahman, I. U. (2026). *Westhoek ERT interface uncertainty: Code, decision log and derived products* (Version 3.1) [Software].
> Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

## What this repository contains

A pre-registered field test of two uncertainty workflows for electrical resistivity tomography (ERT): linearized posterior
bands and Bayesian Evidential Learning (BEL). They are tested on the depth of the base of an upper saline lens below the beach
at De Westhoek, Belgium, against independent soundings. The repository also has the synthetic coverage tests behind them,
a rule that predicts the direction of the bias of fixed-contour interfaces, and the measured damping of the resistivity below
the lens in smooth ERT images.

## Pre-registration

The expectations, counting rules and analysis choices were written in [`docs/decision_log.md`](docs/decision_log.md) and
committed before the data they concern were read. To see when each entry was added and how it changed:

```
git log --follow -p docs/decision_log.md
```

Git lets the person who commits set the commit time, so the commit times alone do not prove when an entry was written. The
record is the archived history: the commit hashes, the Zenodo deposit, and the commit map described below.

**Why the commit hashes differ from the original repository.** Before publication, the history was filtered so that it
holds only code, the decision log, the reported numbers and the plotted values; the field data and the files derived from
restricted data were removed from every commit (`git filter-repo` with an allow-list). Commit messages, authors and dates
are unchanged. [`docs/commit_map_v2.2_to_v3.1.csv`](docs/commit_map_v2.2_to_v3.1.csv) maps every original commit to its new
hash (two commits that changed only removed files are marked as pruned). The unfiltered repository is kept as a private
bundle (SHA-256 `b2f76038af37b48b138356466caa65756f8ec96a62c7baeda5f1dea069ec0b40`) and can be shown to the editor on request.

## Data

Put each input where the "local path" column says; the scripts find it there. `python scripts/run_all.py --check-inputs`
lists every input as in place (with its recorded checksum), missing (with how to obtain it) or restricted.

| input | source | DOI / URL | licence | local path | how to obtain |
|---|---|---|---|---|---|
| 2018 land and marine ERT data (K1 line), electrode coordinates | Hermans & Paepen (2019), Marine Data Archive | [doi:10.14284/390](https://doi.org/10.14284/390) | as stated in the record | `data/raw/` (`land_data.dat`, `elec.dat`, `sea_data.dat`, `GPS_coordinate.xlsx`, …) | download the record and unpack its files into `data/raw/` |
| CPT-EC resistivity logs (2024) and the 2025 ERT line | Vrancken, Paepen & Hermans (2026), Zenodo | [doi:10.5281/zenodo.20792093](https://doi.org/10.5281/zenodo.20792093) | CC BY 4.0 | `data/raw/dp_zenodo_20792093/` | `python scripts/27_dp_groundtruth.py fetch` (checks the Zenodo MD5s) |
| Coastline, land and borders (1:10m) | Natural Earth | https://www.naturalearthdata.com | public domain | `data/external/natural_earth/` | `python scripts/run_all.py --fetch-natural-earth` |
| Bathymetry DTM 2024 | EMODnet Bathymetry Consortium | [doi:10.12770/cf51df64-56f9-4a99-b1aa-36b8d7b743a1](https://doi.org/10.12770/cf51df64-56f9-4a99-b1aa-36b8d7b743a1) | CC BY 4.0 | `data/external/basemap/` | `python scripts/32_location_map.py fetch` |
| Copernicus DEM GLO-30 | ESA / Copernicus (AWS Open Data) | https://copernicus-dem-30m.s3.amazonaws.com | Copernicus DEM licence | `data/external/basemap/` | `python scripts/32_location_map.py fetch` |
| Orthophoto, winter mosaic 2025 | Digitaal Vlaanderen | https://geo.api.vlaanderen.be/OMW/wms | Modellicentie Gratis Hergebruik | `data/external/basemap/` | `python scripts/32_location_map.py fetch` |
| 1980 saline base on profile K1 (digitised) | Lebbe (1981), Fig. 3 | — | digitised values, CC BY 4.0 (this repository) | `data/external/lebbe1981_K1_digitised.csv` | included |

Sources, file sizes and checksums: [`data/raw/README_sources.md`](data/raw/README_sources.md),
[`data/external/basemap/README_sources.md`](data/external/basemap/README_sources.md); folder overview:
[`data/README.md`](data/README.md).

### Data not included

Three inputs are not public and are not in this repository, in its history or in the Zenodo archive:

- **Mechanical CPT logs** (cone resistance, sleeve friction, friction ratio) **and the surveyed positions** of the seven
  soundings of 23 September 2024 (E. Vrancken, Ghent University). The data owners plan to publish them.
- **The 2018 Terrameter LS instrument project** of the K1 land line (stacking errors, reciprocal measurements, timestamps;
  T. Hermans, Ghent University).

They are needed for: the soil-behaviour classification and the clean-sand filter of the sounding resistivity below the
lens; expectation E21 (lithology at the sounding bases); the surveyed geometry of the soundings (SI Table S1) and through it
every comparison with the soundings (Tables 1–2, SI Tables S7–S11, Figures 4–6 and S11–S14, the sounding positions in Figure 1); the 2018 error analysis
(SI Table S2, Figure S2, expectations E13–E14); and SI Figure S1. Scripts that need them stop with
`restricted input missing: … see data/README.md`, and every step before them runs from public data. Once the owners
publish the data, place them at the paths in [`data/README.md`](data/README.md) and rerun those steps.

Not needed: the reference inversion, the linearized uncertainty, the synthetic tests (site-like, factorial, 1980-based),
BEL calibration on held-out models, the bias rule and its out-of-sample tests, and the BEL field estimates from the 2018
data.

The values plotted in each figure are in [`results/figs_wrr_v5/data/`](results/figs_wrr_v5/data/), except Figures S1 and S2,
which derive from the restricted records. In `Figure1.csv` the soundings appear at their positions projected on the K1 line
(SI Table S1); their map coordinates are not public.

## Environment

```
conda env create -f environment.yml          # Windows 11, the exact environment used (Python 3.11, pyGIMLi 1.5.5)
conda activate westhoek-ert
```

`environment.yml` pins every package of the Windows build. On Linux or macOS use `environment-crossplatform.yml` (same
versions of the main packages, other packages resolved by conda; not tested on those systems). Some script docstrings still
call the environment `pg`, its earlier name.

## Reproducing the results

All commands run from `scripts/`.

1. **Inputs.** Obtain the public data (table above) and check them: `python run_all.py --check-inputs`.
2. **Everything from the raw data.** `python run_all.py --from-raw --out <new folder>` lists the steps with their expected run
   times; add `--execute` to run them. On the development machine (Windows 11, 16 GB RAM, 4 worker processes) the full chain
   takes about 33 h (about 24 h for the steps the paper needs); a single inversion uses about 1 GB of memory. Without the
   restricted inputs the run stops at the first step that needs them (the 2018 instrument records, step `26-run`), after all
   public-data steps before it.
3. **Post hoc analyses requested by the review (v2.0).** `python 33_review_v20.py <task>` (tasks `a1` … `a10b`, see the
   docstring), reports with `python 35_reports_v20.py`, figures with `python 36_figs_wrr_v5.py`.
4. **Numbers check.** `python run_all.py --from-cache --out <folder>` regenerates every reported number from the stored
   intermediate products (about 15 min) and compares it with [`results/paper/numbers.md`](results/paper/numbers.md);
   `python 35_reports_v20.py numbers` regenerates [`results/v2_0/numbers_v20.md`](results/v2_0/numbers_v20.md). The
   intermediate products (inversions, prior ensembles, about 3 GB) are not in this repository; they are rebuilt by step 2.

## From paper to code

Figures use the WRR numbering; every figure is drawn by `scripts/paper_figures.py` and exported by `scripts/36_figs_wrr_v5.py`
(images in `results/figs_wrr_v5/`, not in this repository; plotted values in `results/figs_wrr_v5/data/<Figure>.csv`).

| item | content | script and function | plotted values |
|---|---|---|---|
| Figure 1 | study area and data | `paper_figures.fig01` (map layers: `32_location_map.py`) | `Figure1.csv` (soundings projected, see above) |
| Figure 2 | reference model | `paper_figures.fig02` (`02_land_inversion.py`) | `Figure2.csv` |
| Figure 3 | synthetic test of the linearized bands | `paper_figures.fig04` (`10_synthetic_benchmark.py`, `11_stage4_closeout.py`) | `Figure3.csv` |
| Figure 4 | lens base: BEL, linearized, soundings, 1980 | `paper_figures.fig08` (`18_paper.py`, `27_dp_groundtruth.py`) | `Figure4.csv` |
| Figure 5 | 2025 ERT line at the soundings | `paper_figures.fig09` + `36_figs_wrr_v5.add_fig5` (`27_dp_groundtruth.py samedate`) | `Figure5.csv` |
| Figure 6 | water below the lens, BEL prior v4 | `36_figs_wrr_v5.fig_6`, `add_fig6` (`30_bel_v4.py`) | `Figure6.csv` |
| Figure S1 | CPT logs | `paper_figures.figS04` (`27_dp_groundtruth.py cpt`) | not included (restricted) |
| Figure S2 | errors in the 2018 instrument records | `paper_figures.figS02` (`26_raw_abem.py`) | not included (restricted) |
| Figure S3 | BEL with prior v3 | `paper_figures.fig06` (`13_bel_full.py`, `16_stage6_salinity.py`) | `FigureS3.csv` |
| Figure S4 | linearized lens base, 2018 | `paper_figures.fig03` (`11_stage4_closeout.py`) | `FigureS4.csv` |
| Figure S5 | outliers | `paper_figures.figS01` (`05_outlier_characterisation.py`) | `FigureS5.csv` |
| Figure S6 | factorial coverage | `paper_figures.fig05` (`25_generalisation.py`) | `FigureS6.csv` |
| Figure S7 | bias rule | `paper_figures.figS03` + `36_figs_wrr_v5.add_figS7` (`25_generalisation.fit_rule`) | `FigureS7.csv` |
| Figure S8 | synthetic scores | `paper_figures.figS10` (`31_scores.py`) | `FigureS8.csv` |
| Figure S9 | lens base from the 2018 data | `paper_figures.fig07` (`18_paper.py`) | `FigureS9.csv` |
| Figure S10 | salinity, BEL prior v3 | `paper_figures.figS08` (`17_salinity_v2.py`) | `FigureS10.csv` |
| Figure S11 | CPT-EC logs against the 2018 results | `paper_figures.figS05` (`27_dp_groundtruth.py dp`) | `FigureS11.csv` |
| Figure S12 | field scores | `36_figs_wrr_v5.fig_s12` (`31_scores.py`, `33_review_v20.py a1`, `a3`) | `FigureS12.csv` |
| Figure S13 | synthetic truths from the soundings | `paper_figures.figS06` (`28_dp_synthetic.py`) | `FigureS13.csv` |
| Figure S14 | BEL prior v4 on synthetic data | `paper_figures.figS07` (`30_bel_v4.py`) | `FigureS14.csv` |
| Figure S15 | lens base against the 1980 logs | `paper_figures.figS09` (`21_k1_comparison.py`) | `FigureS15.csv` |
| Table 1 | expectations E13–E21 and outcomes | `26_raw_abem.py report`, `27_dp_groundtruth.py report` | — |
| Table 2 | scores at DP2–DP7 | `31_scores.py report`, `33_review_v20.py a3` | — |
| Table S1 | soundings projected on the K1 line | `27_dp_groundtruth.py cpt` (restricted input) | — |
| Table S2 | errors in the 2018 instrument records | `26_raw_abem.py run` (restricted input) | — |
| Table S3 | site-like synthetic truths | `10_synthetic_benchmark.py` (truth specification) | — |
| Table S4 | BEL prior v3 / v4 | `13_bel_full.py`, `16_stage6_salinity.py`, `30_bel_v4.py` | — |
| Table S5 | every logged expectation | `34_alllog_v20.py` | — |
| Table S6 | lens base, 2018 linearized posterior | `11_stage4_closeout.py`, `18_paper.py` | — |
| Table S7 | lens base at the soundings | `27_dp_groundtruth.py dp`, `33_review_v20.py a3` | — |
| Table S8 | outlier treatments and reciprocal error model | `33_review_v20.py a4`, `a4diag` | — |
| Table S9 | resistivity below the lens, % seawater | `27_dp_groundtruth.py dp`, `33_review_v20.py a5` | — |
| Table S10 | scores at each sounding | `33_review_v20.py a3` | — |
| Table S11 | scores with a representativeness error | `33_review_v20.py a3` | — |
| Table S12 | BEL estimates of ρ below the lens | `17_salinity_v2.py`, `30_bel_v4.py`, `18_paper.py` | — |
| Equation (1) | normal matrix and resolution | `02_land_inversion.compute_resolution_matrix`, `westhoek_common.cholesky_A` | — |
| Equation (2) | linearized posterior samples | `westhoek_common.cholesky_A`, `09_posterior_sampling.py`, `11_stage4_closeout.py` | — |
| Equation (3) | fixed-contour bias rule | `25_generalisation.fit_rule` | — |
| Equation (4) | interval score | `31_scores.interval_score` | — |

Every number quoted in the paper is listed with its source in [`results/paper/numbers.md`](results/paper/numbers.md) (v1.x)
and [`results/v2_0/numbers_v20.md`](results/v2_0/numbers_v20.md) (v2.0).

## Repository layout

| folder | content |
|---|---|
| `scripts/` | all analysis and figure code (see [`scripts/README.md`](scripts/README.md)) |
| `docs/` | the decision log (pre-registration) and the commit map |
| `results/` | reported numbers and the values plotted in each figure |
| `data/` | source notes and checksums of the inputs; the small digitised 1980 profile |

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Results, documents and figure data (`results/`, `docs/`, `data/**/README*`,
`data/external/lebbe1981_K1_digitised.csv`): CC BY 4.0 ([`LICENSE-data`](LICENSE-data)). The input data keep the licences of
their own records.

## Contact

Inayat Ur Rahman, gigyani.rahman@gmail.com

## AI assistance

The scripts were developed with the help of Claude Code (Anthropic); the author reviewed the code and takes responsibility
for it.
