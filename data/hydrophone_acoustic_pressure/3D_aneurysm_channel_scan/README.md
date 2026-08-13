# 3D aneurysm-channel hydrophone scan — 2 MHz

Second experiment in this series (first: `quasi2D_pdms_channel_scan/`, the
baseline PDMS-channel scan). This one is a hydrophone scan of the 3D aneurysm
channel, single Z-plane.

## Contents

- `raw_csv/` — 24 raw oscilloscope waveform exports, named
  `CSV{n}_X{x}_Y{y}_Z{z}.csv` (n = 00-23, in acquisition order, X/Y/Z in mm).
  Originally exported by the scope as TEK00037.CSV-TEK00064.CSV; the files have
  since been renumbered 00-23 after removing the discarded repeats (see below).
- `data_table.csv` — one row per scan point, columns `CSV, X_mm, Y_mm, Z_mm,
  Vpp_mV, Pressure_pp_kPa, RawFile`.
- `plot_field.py` — generates the three figures in `figures/` from
  `data_table.csv`.

## Experimental conditions

- Frequency: 2 MHz (transducer drive)
- Drive: 30 Vpp
- Medium: water, no microbubbles (bubble-present reference readings held
  separately, see above)
- Hydrophone: Precision Acoustics SN 4746 (same system as the first
  experiment), 50 Ω termination
- Scan: single XY plane at Z = 13 mm, covering roughly X = 5-14 mm,
  Y = 7-11 mm (denser sampling near X=9-11, Y=8-11)

## Calibration

Because the drive frequency here is exactly 2 MHz, the calibrated sensitivity
from certificate 20251211-03 is used directly with no interpolation:
S = 654.0046 mV/MPa, uncertainty 8.04%. `P_pp [kPa] = Vpp [mV] / 654.0046 x 1000`.

Same caveat as the first experiment: this assumes no external booster
amplifier in the signal chain. See the main project's
`hydrophone_experiment_summary.md` Section 6 for background — that question
is still unresolved and applies here too.

## Data collection method and verification

Readings were called out live during acquisition (position + Vpp) and logged
in real time, then cross-checked against the raw oscilloscope CSVs once
uploaded. All points matched the live-dictated values within normal noise
(largest deviation 5.2%) — `data_table.csv` uses the raw-waveform values
throughout.

## Repeated positions (first readings discarded)

Four positions were measured twice. In each case the first reading was taken
at a slightly wrong probe position and was then repeated more precisely, so the
first (imprecise) reading has been **discarded** — only the corrected reading is
retained. After removal, the remaining 24 waveforms were renumbered 00-23 in
acquisition order. The dataset now has exactly one reading per position.

For traceability, the discarded first reading and the retained one at each
repeated position (original TEK numbering) were:

| Position | Discarded (1st, wrong pos.) | Retained (2nd) → new index |
|---|---|---|
| X=10, Y=11 | TEK37 = 44.0 mV | TEK57 = 33.0 mV → CSV16 |
| X=9, Y=11  | TEK38 = 57.0 mV | TEK58 = 42.5 mV → CSV17 |
| X=12, Y=9  | TEK44 = 36.5 mV | TEK45 = 39.0 mV → CSV05 |
| X=13, Y=9  | TEK46 = 56.0 mV | TEK47 = 65.5 mV → CSV06 |

## Regenerating the figures

```bash
pip install numpy scipy matplotlib
python3 plot_field.py
```
