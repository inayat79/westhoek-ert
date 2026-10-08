# Raw data sources

Files under `data/raw/` are read-only inputs and are not committed to git (repository convention). This file records
their origin, the date received, the licence or permission, size and SHA256, so that `run_all.py` can locate or fetch
them and check them.

## 1. Westhoek land and marine ERT, 2018 (original published dataset)

- Hermans, T. and Paepen, M. (2019). Electrical resistivity tomography at De Westhoek. doi:10.14284/390.
- Received / used since the start of the project (file dates 2019). Licence: see the record behind the DOI (not restated).

| file | bytes | SHA256 |
|---|---|---|
| `land_data.dat` | 59 256 | 68c1469aea2c0a99027364f57b5efff0c842dd9e3fd85e6fcbf46923db69bb1c |
| `sea_data.dat` | 34 547 | c7eee24005906f88795a9677b8d80c1e20b93cfd0dc75448e9d4f983847ed3ea |
| `elec.dat` | 26 057 | 82f72df9ba1d703ef9f58b013dd30ad4c6ed137fb335ec14306ced95bcb21c64 |
| `GPS_coordinate.xlsx` | 8 198 | 2e5bf9997aa32afea9c648dad9e39cc19ab7f02f7b4c5ff23afb1762269ce89e |
| `Read_me.txt` | 955 | ca9f17f8dfcbed7985d183f004015a3b613a89794e62667f4476cd49b7e3f310 |
| `Westhoek_field_description.pdf` | 424 507 | 53364d6ad7f08e4290a0014538556cd90afa739cbfa35ad005fc1cf756492d95 |

## 2. Raw Terrameter LS project of the 2018 K1 land line

- Folder `Project12_LS210110079_2018-03-07_11-22-23/`: Terrameter LS SQLite project "Project12", instrument LS210110079,
  acquired 7 March 2018 (instrument clock 11:22–16:14; time zone unknown).
- Source: T. Hermans, Ghent University; received October 2026 (folder created locally 2026-10-05). Permission: provided by
  the data owner for this analysis; not redistributed (not in git, not in the release bundle).
- Read with SQLite in read-only mode (`26_raw_abem.py`); never modified.

| file | bytes | SHA256 |
|---|---|---|
| `project.db` | 4 683 776 | 6ff26603008bbc024f44fb5b56a7358165bbb3c49251e14aa17c309266d77a57 |
| `project_name.txt` | 9 | 2674fcea1a8b35c0e7b23f95eb9cf383df634b57698d99fd94a4f51835127ba7 |

## 3. Direct-push resistivity and a newer ERT line — Zenodo record 20792093

- Vrancken, E., Paepen, M. and Hermans, T. (2026). *Uncertainty analysis of deterministic ensemble inversions from
  electrical resistivity tomography and direct push resistivity data – Supporting dataset.* Zenodo.
  doi:10.5281/zenodo.20792093 (concept DOI 10.5281/zenodo.20792092). Published 2026-06-22; no version number given.
- Licence: **CC-BY 4.0** (redistribution allowed with attribution; kept out of git by the convention above).
- Downloaded 2026-10-03 (UTC) into `dp_zenodo_20792093/` with the record metadata; re-fetch with
  `27_dp_groundtruth.py fetch` (checks the Zenodo MD5s).

| file | bytes | MD5 (Zenodo) | SHA256 |
|---|---|---|---|
| `DP_R.xzr` | 508 448 | dff0ed3851efcbddb270bf6a8d235e24 | ef4ec39e33747ed512c380c6bc00a47ec45bc41bd25bcc39a1fa0a2990c401fc |
| `ERT.dat` | 63 845 | 9a844221e4d45e1772d6be06749460ad | 393a934803cf6e35d7d7220f14f71214e31208d92d0f1eb2e0ebe5505bb7e4a7 |
| `SGS_DP_R.txt` | 261 420 | 2fa612ce98110e87b09fa7115b340e47 | 39830373542a6184a29f9d8dd3030452925b3343423666962d228680ec607cfc |
| `Mesh.bms` | 902 254 | 04a596b38b2a8a4c10846644d676b0ff | 55997b1f97a4cbbff968cace88cc0162959febcf5cec5d33cae24e26ef614a9f |
| `Python_script.py` | 8 721 | ec438220954b88319151a6106a540c3c | 05c388c08fb9a633b39e661d3ff02b3064504af6bf0c8db667fbc11b40d5e722 |
| `zenodo_record_20792093.json` | 5 337 | — (record metadata, fetched 2026-10-03) | ec59d6129ecdc683efbe15a1bd81b90d899a77f6c13fe68419afbf5579b72c23 |

Not in the record: map coordinates, survey dates, DP ground elevations, units / temperature reference of ρ. Line
placement on our profile: `results/v1_4/1_line_placement.md`.

## 4. Natural Earth

- See `data/external/natural_earth/README.md` (public domain; fetched by `run_all.py --fetch-natural-earth`).

## 5. Mechanical CPT soundings, 23 September 2024 (E. Vrancken, Ghent University)

- Source: E. Vrancken, Ghent University; received 8 October 2026. Survey date 23 September 2024 (E. Vrancken, pers. comm.).
- Status: **unpublished, used with permission, not to be redistributed.** The folder `cpt_vrancken_2024/` is in
  `.gitignore`; only derived tables and figures are committed. The raw files are available from the data owner.
- Per E. Vrancken and T. Hermans (October 2026) the Zenodo DP resistivity logs are CPT-EC measurements from the same pushes.
- `CPT/` is the extracted content of `CPT.zip`; `27_dp_groundtruth.py cpt` checks that every file is byte-identical to its archive member.

| file | bytes | SHA256 |
|---|---|---|
| `CPT.zip` | 14330 | 08c9f3a09b69fba44cf258bb45a06055493abc292189b2161c91d04bcdbe9f8c |
| `CPT/Location.txt` | 221 | fd0057009d612f04da97a10aac1838c6cbb5cd32eb87443896e3e65b2a04b523 |
| `CPT/Proef2.txt` | 5400 | 63261d86bffbeb1ae878d4022ccfc0db428da4f66ace964ee899370e0667d033 |
| `CPT/Proef3.txt` | 5400 | d9a548f9319f8a02c5a1dc303953ac84179e702b95046d7ded804a9ca6c99731 |
| `CPT/Proef4.txt` | 4554 | da032cae082b9db3d917f8e0d2a20efae7d8d9ec392655c95c0ebda98b6b1133 |
| `CPT/Proef5.txt` | 5377 | 9be37997d59c36a4b0f4cd1bba1535b3e1f92c07ef6d49f453bae6768ca0cd34 |
| `CPT/Proef6.txt` | 4926 | 55bedcab17283c571ac309d04af52b0eeb9a12509422420a40ee046688c618b8 |
| `CPT/Proef7.txt` | 3350 | b120e2da5cca0c13910673a378c182913325a399cd8166296a0ad18a234a8f21 |
| `CPT/Proef8.txt` | 4491 | b24dc6be222c88b669ab8629c5a4a9ff864f2a89ce6af08bfa094890305c7155 |
