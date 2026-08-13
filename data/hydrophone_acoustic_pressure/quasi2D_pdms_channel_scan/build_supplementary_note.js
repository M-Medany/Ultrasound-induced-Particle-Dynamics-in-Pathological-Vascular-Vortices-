const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, ImageRun, PageBreak,
  LevelFormat, convertInchesToTwip,
} = require("docx");

const FIG = path.join(__dirname, "figures");
const ANEURYSM_DIR = path.join(__dirname, "..", "3D_aneurysm_channel_scan");
const ANEURYSM_FIG = path.join(ANEURYSM_DIR, "figures");

// ---------- helpers ----------
function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 150 } });
}
function pRuns(runs, opts = {}) {
  return new Paragraph({ children: runs, spacing: { after: 150 }, ...opts });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 20 })],
    spacing: { after: 300 },
    alignment: AlignmentType.CENTER,
  });
}
function cell(text, opts = {}) {
  const { bold = false, width = 1000, shade = null, align = AlignmentType.CENTER } = opts;
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text), bold })],
    })],
  });
}
function dataTable(headers, rows, widths) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((htext, i) => cell(htext, { bold: true, width: widths[i], shade: "D9E2F3" })),
  });
  const bodyRows = rows.map((r, idx) => new TableRow({
    children: r.map((v, i) => cell(v, { width: widths[i], shade: idx % 2 === 1 ? "F2F2F2" : null })),
  }));
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    rows: [headerRow, ...bodyRows],
  });
}
function image(file, { width, height, dir = FIG }) {
  const data = fs.readFileSync(path.join(dir, file));
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 100 },
    children: [new ImageRun({ type: "png", data, transformation: { width, height } })],
  });
}

// ---------- Experiment 2: aneurysm-channel scan data (verified against raw CSV waveforms) ----------
const aneurysmRows = [
  [37, 10, 11, 13, 44.0, 67.3], [38, 9, 11, 13, 57.0, 87.2], [39, 8, 11, 13, 41.0, 62.7],
  [40, 7, 11, 13, 20.0, 30.6], [41, 6, 11, 13, 33.0, 50.5], [42, 10, 9, 13, 62.5, 95.6],
  [43, 11, 9, 13, 56.5, 86.4], [44, 12, 9, 13, 36.5, 55.8], [45, 12, 9, 13, 39.0, 59.6],
  [46, 13, 9, 13, 56.0, 85.6], [47, 13, 9, 13, 65.5, 100.2], [48, 14, 9, 13, 59.5, 91.0],
  [49, 9, 9, 13, 70.5, 107.8], [50, 8, 9, 13, 43.5, 66.5], [51, 7, 9, 13, 31.5, 48.2],
  [52, 6, 9, 13, 43.5, 66.5], [53, 5, 9, 13, 30.5, 46.6], [54, 10, 8, 13, 40.5, 61.9],
  [55, 10, 7, 13, 27.5, 42.0], [56, 10, 10, 13, 67.5, 103.2], [57, 10, 11, 13, 33.0, 50.5],
  [58, 9, 11, 13, 42.5, 65.0], [59, 9, 10, 13, 70.5, 107.8], [60, 9, 8, 13, 54.0, 82.6],
  [61, 11, 8, 13, 31.5, 48.2], [62, 11, 7, 13, 34.0, 52.0], [63, 11, 10, 13, 39.0, 59.6],
  [64, 11, 11, 13, 28.0, 42.8],
].map(r => [
  `${r[0]}`, `${r[1]}`, `${r[2]}`, `${r[3]}`, r[4].toFixed(2), r[5].toFixed(1),
]);

// ---------- Experiment 1: baseline PDMS-channel scan data (verified against raw CSV waveforms) ----------
const scanRows = [
  [15, 10, 12, 6, 33.75, 51.5],
  [16, 11, 12, 6, 50.00, 76.3],
  [17, 12, 12, 6, 51.50, 78.6],
  [18, 13, 12, 6, 51.75, 78.9],
  [19, 14, 12, 6, 31.75, 48.4],
  [20, 15, 12, 6, 32.75, 50.0],
  [21, 9, 12, 6, 44.25, 67.5],
  [22, 8, 12, 6, 32.00, 48.8],
  [23, 7, 12, 6, 27.50, 41.9],
  [24, 10, 11, 6, 44.00, 67.1],
  [25, 10, 10, 6, 35.50, 54.1],
  [26, 10, 9, 6, 42.00, 64.1],
  [27, 10, 13, 6, 26.50, 40.4],
  [28, 9, 11, 6, 28.00, 42.7],
  [29, 11, 11, 6, 27.50, 41.9],
  [30, 11, 13, 6, 30.50, 46.5],
  [31, 9, 13, 6, 22.50, 34.3],
  [32, 10, 12, 7, 41.25, 62.9],
  [33, 10, 12, 8, 38.25, 58.3],
].map(r => [
  `${r[0]}`, `${r[1]}`, `${r[2]}`, `${r[3]}`, r[4].toFixed(2), r[5].toFixed(1),
]);

const doc = new Document({
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } },
    },
    children: [
      new Paragraph({
        text: "Supplementary Note: Hydrophone Calibration and Spatial Field Mapping",
        heading: HeadingLevel.TITLE,
        spacing: { after: 100 },
      }),
      p("Needle-hydrophone characterization of the ultrasound transducer field: baseline PDMS-channel scan (1.7 MHz) and 3D aneurysm-channel scan (2 MHz)", { italics: true, size: 22 }),
      p("Prepared 12 Aug 2026; updated 13 Aug 2026 with the aneurysm-channel scan.", { size: 20, color: "666666" }),

      h1("Experiment 1: baseline PDMS-channel scan (1.7 MHz)"),

      h1("S1. Hydrophone system and calibration"),
      p("Spatial pressure mapping was performed using a needle hydrophone system supplied by Precision Acoustics Ltd (Dorchester, UK), comprising:"),
      bullet("Needle hydrophone, 0.5 mm sensor diameter (model NH0500), serial number 4746"),
      bullet("Submersible preamplifier, serial number HP34211"),
      bullet("DC coupler with power supply, serial number DCPS0895"),
      p("The system was calibrated as a unit (hydrophone + preamplifier + DC coupler) under UKAS-accredited, ISO/IEC 17025 conditions. Calibration certificate 20251211-03 was issued 11 Dec 2025 (measurement date 10 Dec 2025) for University of Bern, using a shocked-wave substitution method (Smith & Bacon, 1990) per IEC 62127-2, with reported uncertainties evaluated per UKAS M3003 and JCGM 100:2008 at a coverage factor k=2 (~95% confidence). Calibration conditions: RG58 cable (1.5 m, 50 Ω), 25-cycle toneburst, de-ionised/de-gassed/filtered water (conductivity <5 μS/cm), water temperature 20 ± 1 °C, >30 min soak period."),

      h2("S1.1 Calibrated sensitivity"),
      p("The full calibration table (1–30 MHz) is provided in the attached original certificate (Precision_Acoustics_Calibration_Certificate_20251211-03.pdf, supplementary file). At the experimental frequency of 1.7 MHz, linear interpolation between the certificate's 1 MHz and 2 MHz sensitivity values (659.4124 mV/MPa and 654.0046 mV/MPa respectively) gives a sensitivity of S = 655.63 mV/MPa, with an interpolated uncertainty of approximately 8.0% (both bracketing points are individually quoted at 8.04% on the certificate)."),
      pRuns([
        new TextRun("Pressure was calculated from measured peak-to-peak voltage as: "),
        new TextRun({ text: "P", italics: true }),
        new TextRun({ text: "pp", subScript: true }),
        new TextRun(" [kPa] = "),
        new TextRun({ text: "V", italics: true }),
        new TextRun({ text: "pp", subScript: true }),
        new TextRun(" [mV] / 655.63 × 1000."),
      ]),

      h2("S1.2 Electrical configuration"),
      p("Precision Acoustics specifies a 50 Ω acquisition load for this hydrophone system (needle hydrophone → HP34211 preamplifier → DCPS0895 DC coupler → 50 Ω acquisition system). Where the oscilloscope input is high-impedance, an external inline 50 Ω terminator is required at the scope input; the RG58/U cable used has a 50 Ω characteristic impedance but does not itself provide the required 50 Ω load. A 50 Ω termination was used at the oscilloscope input for all measurements reported here. The submersible preamplifier and DC coupler provide impedance buffering and a small signal gain that is already incorporated into the system calibration above and must not be separately divided out."),

      h1("S2. Experimental conditions"),
      bullet("Operating frequency: 1.7 MHz"),
      bullet("Transducer drive: 20 Vpp"),
      bullet("Medium: water"),
      bullet("Hydrophone: Precision Acoustics needle hydrophone SN 4746 (as calibrated above)"),
      bullet("Oscilloscope: Tektronix TBS2072, 50 Ω termination, 2000-point records, 8 ns sample interval"),
      bullet("Transducer (reported separately, not confirmed for this specific unit): aperture ~19.5–20 mm, nominal focal distance ~25 mm, nominal operating frequency ~1.8 MHz; driven here at 1.7 MHz"),

      new Paragraph({ children: [new PageBreak()] }),

      h1("S3. Important caveat: booster amplifier"),
      p("Precision Acoustics separately supplies an optional Hydrophone Booster Amplifier for use with low-sensitivity needle hydrophones (typical gain 27 dB / voltage gain G ≈ 22.4, 50 kHz–130 MHz bandwidth, 50 Ω input/output). The calibration certificate for SN 4746 lists the hydrophone, preamplifier and DC coupler, but the booster amplifier serial-number field is blank on the calibration data sheet (distinct from the attenuator and signal-amplifier fields, which are explicitly marked “N/A”). The published sensitivity (655.63 mV/MPa at 1.7 MHz) therefore applies to the hydrophone + preamplifier + DC coupler system only, and NOT to an additional external booster stage."),
      pRuns([
        new TextRun({ text: "Whether the booster amplifier was present in the signal chain for the measurements reported in Section S4 has not been confirmed. ", bold: true }),
        new TextRun("If it was, every pressure value in this note must be divided by G ≈ 22.4 (e.g. 34 mV → 1.52 mV before applying the 655.63 mV/MPa sensitivity). This changes the absolute pressure scale by roughly an order of magnitude but does not affect the relative spatial pattern. The raw peak-to-peak voltages (Vpp) are unaffected by this question and should be treated as the reliable primary measurement; the pressure values in this note are provisional pending confirmation of the signal chain."),
      ]),

      h1("S4. Spatial field-mapping data"),
      p("The transducer field was mapped at 19 positions: a cross-pattern scan in X and Y at Z = 6 mm centered near (X=10, Y=12), four diagonal points, and a 3-point depth scan in Z at (X=10, Y=12). Peak-to-peak voltage (Vpp) was computed directly from each raw oscilloscope waveform in raw_csv/ (full-trace maximum minus minimum); pressure was then obtained from Vpp using the 655.63 mV/MPa sensitivity from Section S1.1."),
      dataTable(
        ["CSV", "X (mm)", "Y (mm)", "Z (mm)", "Vpp (mV)", "P_pp (kPa)"],
        scanRows,
        [1000, 1400, 1400, 1400, 1900, 1900]
      ),
      new Paragraph({ text: "", spacing: { after: 150 } }),
      p("Note: during preparation of this note, two points (CSV 19 and CSV 28) were found to disagree with an earlier, hand-logged version of this table by 28–33% (hand-logged as 44 mV and 42 mV respectively). Recomputing Vpp directly from the raw waveform files resolved the discrepancy; the values above (31.75 mV and 28.00 mV) are taken from that direct recomputation. All other 17 points agreed with the hand-logged values within 1–3 mV.", { italics: true, size: 20 }),

      h1("S5. Field maps"),
      image("field_map_2D.png", { width: 470, height: 382 }),
      caption("Figure S1. Interpolated pressure map of the XY scan plane at Z = 6 mm (provisional, no booster correction). Circles mark measured points; the filled region is linear-interpolated within the convex hull of the sampled points only."),

      image("field_profiles.png", { width: 560, height: 168 }),
      caption("Figure S2. Line profiles through the pressure maximum: X-scan (Y=12, Z=6), Y-scan (X=10, Z=6), and Z-scan / depth (X=10, Y=12). Provisional pressures, no booster correction."),

      new Paragraph({ children: [new PageBreak()] }),

      image("field_scatter_3D_channel_style.png", { width: 520, height: 440 }),
      caption("Figure S3. 3D scatter of all 19 measurement points relative to the PDMS channel block (0–5.5 mm in Z; channel height marker at Z=5 mm; cross-section Y=8–15 mm; long axis 20 mm in X, centered on X=10, the original scan center — the transducer offset along X was not independently confirmed). Marker size and color both encode provisional peak-to-peak pressure."),

      h2("S5.1 Notable feature"),
      p("Both the X-scan and Y-scan profiles (Figure S2) show a local minimum at the scan's nominal center point (X=10, Y=12, Z=6; 51.5 kPa), flanked by higher readings at adjacent grid points in both directions (e.g. X=9: 67.5 kPa, X=11: 76.3 kPa; Y=11: 67.1 kPa, Y=9: 64.1 kPa). This point's raw waveform matched the originally logged value closely (~1% difference), so it is not attributable to a transcription error; it may reflect a local interference null or an off-axis effect near the field center, and is noted here for awareness rather than explained."),

      new Paragraph({ children: [new PageBreak()] }),

      h1("Experiment 2: 3D aneurysm-channel scan (2 MHz)"),

      h1("S6. Aneurysm-channel scan"),
      p("A second scan was performed on a 3D-printed/PDMS aneurysm-channel model, using the same Precision Acoustics hydrophone system (SN 4746) described in Section S1. The transducer drive was changed for this experiment."),

      h2("S6.1 Experimental conditions"),
      bullet("Operating frequency: 2 MHz"),
      bullet("Transducer drive: 30 Vpp"),
      bullet("Medium: water, no microbubbles (a small set of bubble-reference readings was taken at the scan's start position but is held pending confirmation of measurement order — see Section S6.5)"),
      bullet("Hydrophone: Precision Acoustics needle hydrophone SN 4746 (same system and calibration as Experiment 1)"),
      bullet("Oscilloscope: Tektronix TBS2072, 50 Ω termination, 2000-point records, 8 ns sample interval"),
      bullet("Channel: larger cross-section than the PDMS channel in Experiment 1 (aneurysm phantom geometry)"),

      h2("S6.2 Calibration at 2 MHz"),
      pRuns([
        new TextRun("2 MHz is an exact calibration point on certificate 20251211-03, so no interpolation is needed: S = 654.0046 mV/MPa, uncertainty 8.04%. "),
        new TextRun({ text: "P", italics: true }),
        new TextRun({ text: "pp", subScript: true }),
        new TextRun(" [kPa] = "),
        new TextRun({ text: "V", italics: true }),
        new TextRun({ text: "pp", subScript: true }),
        new TextRun(" [mV] / 654.0046 × 1000. The same booster-amplifier caveat from Section S3 applies here: pressure values are provisional pending confirmation of whether the external booster was in the signal chain."),
      ]),

      h2("S6.3 Data collection and verification"),
      p("Readings were called out live during acquisition (position + Vpp) and logged in real time as the scan progressed, then cross-checked once the raw oscilloscope CSVs were uploaded. All 28 points matched the live-logged values within normal measurement noise (largest deviation 5.2%); the table below uses the raw-waveform values throughout."),
      dataTable(
        ["CSV", "X (mm)", "Y (mm)", "Z (mm)", "Vpp (mV)", "P_pp (kPa)"],
        aneurysmRows,
        [1000, 1400, 1400, 1400, 1900, 1900]
      ),
      new Paragraph({ text: "", spacing: { after: 150 } }),
      p("Scan pattern: a single XY plane at Z = 13 mm, covering roughly X = 5-14 mm and Y = 7-11 mm, with denser sampling near X = 9-11, Y = 8-11. Four positions were measured twice; both readings are listed above (see Section S6.5).", { italics: true, size: 20 }),

      h2("S6.4 Field maps"),
      image("field_map_2D.png", { width: 470, height: 366, dir: ANEURYSM_FIG }),
      caption("Figure S4. Interpolated pressure map of the XY scan plane at Z = 13 mm through the aneurysm channel (provisional, no booster correction). Circles mark measured points (* = second/later reading kept at a repeated position, per lab decision — see Section S6.5); the filled region is linear-interpolated within the convex hull of the sampled points only."),

      image("field_profiles.png", { width: 560, height: 215, dir: ANEURYSM_FIG }),
      caption("Figure S5. X-scan cuts at Y=9 and Y=11, and Y-scan cuts at X=9, 10 and 11, through the aneurysm channel. Provisional pressures, no booster correction."),

      new Paragraph({ children: [new PageBreak()] }),

      h2("S6.5 Reproducibility check"),
      p("Four positions were measured twice during this scan:"),
      dataTable(
        ["Position", "Readings (mV)", "Spread"],
        [
          ["X=10, Y=11 (CSV37/57)", "44.0 / 33.0", "25%"],
          ["X=9, Y=11 (CSV38/58)", "57.0 / 42.5", "25%"],
          ["X=12, Y=9 (CSV44/45)", "36.5 / 39.0", "7%"],
          ["X=13, Y=9 (CSV46/47)", "56.0 / 65.5", "15%"],
        ],
        [3200, 2800, 1700]
      ),
      new Paragraph({ text: "", spacing: { after: 150 } }),
      image("reproducibility_check.png", { width: 420, height: 308, dir: ANEURYSM_FIG }),
      caption("Figure S6. Both readings at each of the 4 repeated positions. The two largest-spread repeats are both at Y=11, the edge of the denser sampling region."),
      p("The 2D map and profile plots (Figures S4-S5) use the second (later) reading of each repeated pair, not an average, per lab decision; the table in Section S6.3 lists every individual raw reading, including the first (discarded-for-plotting) reading at each repeated position.", { italics: true, size: 20 }),

      h2("S6.6 Bubble-reference readings (pending)"),
      p("Three additional readings (TEK00034-036) were taken at the scan's start position (X=10, Y=11, Z=13) before the main scan: 13.5 mV, 40.0 mV and 56.5 mV. The much lower first value is consistent with acoustic attenuation/scattering from microbubbles in the beam path, but the before/with/after-bubbles order across these three readings has not been confirmed and is not included in the data table above. These files are retained separately (raw_csv_pending_bubble_ref/) pending that confirmation."),

      h1("S7. Data and code availability"),
      p("Experiment 1 (baseline PDMS channel): the original Precision Acoustics calibration certificate (20251211-03), raw oscilloscope waveforms, the derived data table, and the Python scripts used to generate Figures S1-S3 are provided as supplementary data files."),
      p("Experiment 2 (aneurysm channel): raw oscilloscope waveforms, the derived data table, and the Python script used to generate Figures S4-S6 are provided as supplementary data files, including the three pending bubble-reference readings held separately from the main dataset."),
      p("All files will be made available in the associated GitHub repository for this project."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(__dirname, "Hydrophone_Supplementary_Note.docx"), buf);
  console.log("wrote Hydrophone_Supplementary_Note.docx");
});
