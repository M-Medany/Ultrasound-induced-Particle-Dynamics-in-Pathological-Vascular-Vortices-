"""
Parses the raw Tektronix oscilloscope CSVs (raw_csv/CSV*.csv) and computes
the full-window peak-to-peak voltage (Vpp) for each capture, as a check
against the manually-recorded values in hydrophone_data.csv.

Each TEK CSV has a header block (scope settings) followed by a "TIME,CH2"
line and then the waveform samples (2000 rows: time in s, voltage in V).
Vpp here = max(CH2) - min(CH2) over the full recorded window, in mV.

Usage: python3 compute_vpp_from_raw.py
"""
import csv
import glob
import os

def read_vpp(path):
    values = []
    in_data = False
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if not in_data:
                if row[0].strip().upper() == "TIME":
                    in_data = True
                continue
            if len(row) >= 2:
                try:
                    values.append(float(row[1]))
                except ValueError:
                    continue
    return (max(values) - min(values)) * 1000  # V -> mV

def main():
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "raw_csv", "CSV*.csv")))
    print(f"{'file':<12}{'computed Vpp (mV)':>20}")
    for path in files:
        name = os.path.basename(path).replace(".CSV", "")
        vpp = read_vpp(path)
        print(f"{name:<12}{vpp:>20.1f}")

if __name__ == "__main__":
    main()
