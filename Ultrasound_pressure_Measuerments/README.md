# Ultrasound pressure measurements

Hydrophone spatial acoustic-pressure maps of the transducer field around the
PDMS channels — three datasets across two hydrophones.

## Field maps

![Aneurysm-channel peak-to-peak pressure field map](3D_aneurysm_channel_scan/figures/field_map_2D.png)

*Example result — peak-to-peak pressure mapped across the aneurysm channel. The
hydrophone scans in the water above the sealed channel; the PZT drives the field
from below (all z measured from the transducer face). Each dataset folder has its
own maps. (Experimental setup photos to be added.)*

## Datasets

- [`quasi2D_pdms_channel_scan/`](quasi2D_pdms_channel_scan/) — quasi-2D PDMS channel · Precision Acoustics SN 4746 · 1.7 MHz / 20 Vpp · 19 points.
- [`3D_aneurysm_channel_scan/`](3D_aneurysm_channel_scan/) — 3D aneurysm channel · Precision Acoustics SN 4746 · 2 MHz / 30 Vpp · 24 points.
- [`quasi2D_RP71s_channel_scan/`](quasi2D_RP71s_channel_scan/) — quasi-2D PDMS channel, re-measured with an RP Acoustics RP 71s PVDF hydrophone · 1.7 MHz / 20 Vpp · 15 points.

The first two use the Precision Acoustics needle system (SN 4746, preamplifier
HP34211, DC coupler DCPS0895, certificate 20251211-03); the third uses the RP
Acoustics RP 71s (separate calibration — see its README). See each sub-folder's
README, and the paper and its supplementary note, for full experimental context.
