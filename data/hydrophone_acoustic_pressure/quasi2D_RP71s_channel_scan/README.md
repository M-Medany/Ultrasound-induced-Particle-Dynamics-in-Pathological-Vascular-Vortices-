# Hydrophone field map — quasi-2D PDMS channel (RP Acoustics RP 71s, 1.7 MHz)

Second hydrophone measurement of the quasi-2D PDMS channel, using an
**RP Acoustics RP 71s** PVDF hydrophone — a *different instrument* from the
Precision Acoustics needle SN 4746 used in [`quasi2D_pdms_channel_scan/`](../quasi2D_pdms_channel_scan/).
1.7 MHz / 20 Vpp, 5-cycle burst; transducer 20 × 15 mm.

## Contents

- `raw_csv/` — 15 raw Tektronix TBS2072 waveforms, `CSV{n}_X{x}_Y{y}_Z{z}.csv`
  (n = 00–14; `TIME`/`CH2` in s/V, 2000 points, 4 ns interval). Originally
  TEK00000–TEK00014.
- `data_table.csv` — `CSV, X_mm, Y_mm, Z_mm, Vpp_mV, Pressure_pp_kPa, RawFile`.
- `hydrophone_data.csv` — richer provenance table (adds centered X/Y and the
  peak-pressure / bar columns).
- `compute_vpp_from_raw.py` — recompute Vpp straight from `raw_csv/` (verified to
  match `data_table.csv` exactly).
- `plot_field.py` — regenerates `figures/field_scatter_3D.png`.

## Signal chain and conversion

RP Acoustics RP 71s PVDF hydrophone → RP Acoustics HVA-10M-60-F amplifier at
40 dB (×100) → Tektronix TBS2072. `Vpp_mV` is full-window peak-to-peak.

```
P_pp [bar] = Vpp[mV] / (100 × S),   S = 3.7 mV/bar
P_pp [kPa] = P_pp[bar] × 100  =  Vpp[mV] / 3.7
```

`data_table.csv` reports **peak-to-peak** kPa (to match the other folders).
`hydrophone_data.csv` additionally lists the **peak** value (half of p-p).

## Caveats — read before citing the kPa numbers

- **Sensitivity is out of band.** S = 3.7 mV/bar is RP Acoustics' certified value
  for **1–100 kHz only**. At 1.7 MHz the true sensitivity is higher (its
  calibration curve peaks ~9.3 mV/bar near 600 kHz, then decays), so the kPa here
  is a working estimate and **likely an overestimate**. Get S(1.7 MHz) from RP
  Acoustics to correct it. `Vpp_mV` is unaffected and can be trusted as recorded.
- **Different hydrophone** from `quasi2D_pdms_channel_scan/` (Precision Acoustics
  SN 4746, 654 mV/MPa) — do not mix the two sensitivities.
- **Sparse cross scan**: an X-line (y = 11, z = 6), a Y-line (x = 10, z = 6) and a
  Z-line (x = 10, y = 11) through a shared point — not a dense volumetric grid.
- Channel box dimensions in `plot_field.py` are still provisional/estimated.

Coordinate origin = upper-right corner of the transducer; centered coordinates
`X = x − 10`, `Y = y − 7.5` (mm). See the paper and its supplementary note for
full experimental context.
