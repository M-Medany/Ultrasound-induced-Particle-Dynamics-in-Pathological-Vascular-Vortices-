# Hydrophone spatial field mapping — 1.7 MHz, quasi-2D PDMS channel scan

Data and analysis for a Precision Acoustics needle-hydrophone spatial scan of an
ultrasound transducer field, driven at 1.7 MHz / 20 Vpp. First experiment in
this series (second: `3D_aneurysm_channel_scan/`, the 3D aneurysm-channel scan
at 2 MHz).

## Contents

- `raw_csv/` — 19 raw oscilloscope waveform exports, named
  `CSV{n}_X{x}_Y{y}_Z{z}.csv` (e.g. `CSV15_X10_Y12_Z06.csv`), where `n` is the
  original scan number (15-33, for cross-reference to lab notes) and X/Y/Z are
  the scan position in mm. Originally exported by the scope as
  TEK00015.CSV-TEK00033.CSV. Tektronix TBS2072, `TIME`/`CH1` columns in
  seconds/volts, 2000 points each, 8 ns sample interval. These are the primary
  data. This is the no-microbubble baseline scan (first experiment); a second
  scan with microbubbles present is planned as a separate dataset.
- `data_table.csv` — one row per scan point (`CSV`, `X_mm`, `Y_mm`, `Z_mm`,
  `Vpp_mV`, `Pressure_pp_kPa`), derived from `raw_csv/` as described below.
- `plot_field.py` — reads `data_table.csv` and produces:
  - `figures/field_map_2D.png` — interpolated pressure map of the XY scan at Z = 6 mm
  - `figures/field_profiles.png` — X, Y and Z line profiles through the pressure peak
  - `figures/field_scatter_3D.png` — 3D scatter of all 19 points
- `plot_field_3D_channel_style.py` — same data, styled 3D scatter with a
  PDMS channel-block overlay (produces `figures/field_scatter_3D_channel_style.png`).

## How `Vpp_mV` and `Pressure_pp_kPa` were derived

1. For each raw CSV, peak-to-peak voltage was computed directly from the waveform
   (max − min over the full trace).
2. Pressure was calculated as `P_pp [kPa] = Vpp [mV] / 655.63 * 1000`, using the
   hydrophone system's calibrated sensitivity at 1.7 MHz (linear interpolation
   between the 1 MHz and 2 MHz points on Precision Acoustics calibration
   certificate 20251211-03, hydrophone SN 4746 + preamplifier HP34211 + DC
   coupler DCPS0895). Calibration uncertainty near 1–2 MHz is ~8%.

## Known caveats — read before citing these numbers

- **Booster amplifier not confirmed.** Precision Acoustics also supplies an
  optional Hydrophone Booster Amplifier (~27 dB / ×22.4 gain). The calibration
  certificate covers the hydrophone + preamplifier + DC coupler only — it does
  not account for this booster. Whether the booster was in the signal chain
  during this scan has not been confirmed. If it was, every `Pressure_pp_kPa`
  value here must be divided by ~22.4. **Treat the pressure column as
  provisional** until this is resolved; the raw waveforms and `Vpp_mV` are not
  affected by this question and can be trusted as recorded.
- **Two points were corrected from an earlier transcribed table.** CSV 19
  (X=14, Y=12, Z=6) and CSV 28 (X=9, Y=11, Z=6) were originally logged as 44 mV
  and 42 mV respectively. Direct measurement from the raw waveforms in
  `raw_csv/` gives ~32 mV and ~28 mV. The values in `data_table.csv` reflect
  the raw-waveform measurement, not the earlier transcription. All other 17
  points matched the earlier transcription within 1–3 mV.
- Scan geometry is sparse (a cross pattern plus four corner points at Z = 6 mm,
  plus a 3-point depth scan at X=10, Y=12), not a dense grid — the 2D map in
  `field_map_2D.png` is interpolated only within the convex hull of the sampled
  points and should not be over-interpreted between widely spaced points.

## Regenerating the figures

```bash
pip install numpy scipy matplotlib
python3 plot_field.py
python3 plot_field_3D_channel_style.py
```
