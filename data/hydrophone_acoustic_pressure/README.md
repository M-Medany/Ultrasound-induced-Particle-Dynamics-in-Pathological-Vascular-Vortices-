# Hydrophone acoustic pressure mapping

Needle-hydrophone calibration and spatial acoustic pressure mapping data, collected as
two related scans:

- [`quasi2D_pdms_channel_scan/`](quasi2D_pdms_channel_scan/) — baseline PDMS-channel scan,
  1.7 MHz / 20 Vpp, 19 points (first experiment).
- [`3D_aneurysm_channel_scan/`](3D_aneurysm_channel_scan/) — 3D aneurysm-channel scan,
  2 MHz / 30 Vpp, 28 points (second experiment).

Both scans used the same Precision Acoustics needle hydrophone system (SN 4746,
preamplifier HP34211, DC coupler DCPS0895) and the same calibration certificate
(20251211-03, issued 11 Dec 2025). See each sub-folder's own README for scan-specific
details, derivation of pressure values, and known caveats — including an unresolved
question of whether an external booster amplifier was in the signal chain, which affects
every pressure value in both datasets (raw Vpp values are unaffected).

A combined write-up covering both experiments is at
[`quasi2D_pdms_channel_scan/Hydrophone_Supplementary_Note.docx`](quasi2D_pdms_channel_scan/Hydrophone_Supplementary_Note.docx)
(Sections S1-S5: Experiment 1; Sections S6-S7: Experiment 2).
