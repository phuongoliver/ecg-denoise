import os
from pathlib import Path
import wfdb
import matplotlib.pyplot as plt

# === PATH ===

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent        # ecg-fog-denoise/
DATA_DIR  = PROJ_ROOT / "data"
FIG_DIR   = PROJ_ROOT / "results" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MITDB_PATH = DATA_DIR / "mitdb/physionet.org/files/mitdb/1.0.0"
NSTDB_PATH = DATA_DIR / "nstdb/physionet.org/files/nstdb/1.0.0"

# === Helper to plot ECG ===
def plot_ecg(record_path, record_name, fs=360, duration_sec=10, title=""):
    """
    record_path: path to the folder
    record_name: file prefix without extension, e.g. '100'
    fs: sampling frequency (Hz)
    duration_sec: seconds to plot
    """
    rec = wfdb.rdrecord(os.path.join(record_path, record_name))
    sig = rec.p_signal[:, 0]  # lấy lead đầu tiên
    t = [i / fs for i in range(len(sig))]

    # lấy duration_sec giây đầu
    N = int(duration_sec * fs)
    plt.figure(figsize=(12, 3))
    plt.plot(t[:N], sig[:N], linewidth=1)
    plt.title(f"{title} ({record_name})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.grid(True)
    plt.tight_layout()
    out_path = FIG_DIR / f"{title.replace(' ', '_')}_{record_name}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    # === EDA for MIT-BIH Arrhythmia ===
    print("Loading MIT-BIH record 100...")
    plot_ecg(MITDB_PATH, "100", fs=360, duration_sec=10, title="MIT-BIH Arrhythmia")

    # === EDA for NSTDB noise ===
    # NSTDB có nhiều file noise: bw (baseline wander), em (electrode motion), ma (motion artifact)
    # vd: 118e00, 118e06, 118e12, 118e18
    print("Loading NSTDB noise (baseline wander example)...")
    plot_ecg(NSTDB_PATH, "118e00", fs=360, duration_sec=10, title="NSTDB Noise (Baseline Wander)")

    print("EDA finished.")