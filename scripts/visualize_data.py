# %% [markdown]
# # Exploratory Data Analysis (EDA)
# This notebook explores the MIT-BIH Arrhythmia Database and NSTDB (Noise Stress Test Database).
# We visualizes raw ECG signals and various noise artifacts to understand the data characteristics
# before feeding them into the Denoising Model.

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure paths
# Note: When running as a script, __file__ is used. In Jupyter, we might need to set root manually.
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd())

PROJ_ROOT = SCRIPT_DIR.parent
DATA_DIR  = PROJ_ROOT / "data"

# Define specific database paths (Adjust if your structure is different)
MITDB_PATH = DATA_DIR / "mitdb/physionet.org/files/mitdb/1.0.0"
NSTDB_PATH = DATA_DIR / "nstdb/physionet.org/files/nstdb/1.0.0"

print(f"Project Root: {PROJ_ROOT}")
print(f"Data Dir:     {DATA_DIR}")

# %% [markdown]
# ## 2. Helper Functions

# %%
def plot_ecg_segment(record_path, record_name, fs=360, duration_sec=10, title="ECG Signal"):
    """
    Reads an ECG record and plots a segment of it.
    
    Args:
        record_path (Path): Directory containing the record.
        record_name (str): Name of the record (e.g., '100').
        fs (int): Sampling frequency.
        duration_sec (float): Duration to visualize.
        title (str): Plot title.
    """
    full_path = record_path / record_name
    if not (record_path / (record_name + ".dat")).exists():
        print(f"[ERROR] Record not found: {full_path}")
        return

    try:
        # Read record using wfdb
        rec = wfdb.rdrecord(str(full_path))
        sig = rec.p_signal[:, 0]  # Take the first lead (usually MLII)
        
        # Prepare time axis
        points = int(duration_sec * fs)
        t = np.arange(points) / fs
        
        # Plot
        plt.figure(figsize=(15, 4))
        plt.plot(t, sig[:points], color='#0052cc', linewidth=1.2)
        plt.title(f"{title} (Record: {record_name})", fontsize=14, fontweight='bold')
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Amplitude (mV)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not read record {record_name}: {e}")

# %% [markdown]
# ## 3. Visualizing Clean ECG (MIT-BIH)
# The MIT-BIH Arrhythmia Database contains 48 half-hour excerpts of two-channel ambulatory ECG recordings.

# %%
print("Visualizing generic ECG signal from MIT-BIH Record 100...")
plot_ecg_segment(MITDB_PATH, "100", title="Clean ECG Baseline")

print("Visualizing Record 117 (often has more operational irregularities)...")
plot_ecg_segment(MITDB_PATH, "117", title="ECG Record 117")

# %% [markdown]
# ## 4. Visualizing Noise Artifacts (NSTDB)
# The Noise Stress Test Database (NSTDB) includes valid noise samples:
# - **bw**: Baseline Wander (low frequency, caused by breathing)
# - **em**: Electrode Motion (irregular, caused by contact issues)
# - **ma**: Muscle Artifact (high frequency, caused by muscle tension)

# %%
# Example noise files in NSTDB often follow 'bw', 'em', 'ma' naming or specific records like '118e_6'
# Note: Ensure you have run 'scripts/download_data.py' to populate these folders.

noise_types = {
    "Baseline Wander (bw)": "bw",
    "Electrode Motion (em)": "em",
    "Muscle Artifact (ma)": "ma"
}

# The actual filenames in NSTDB v1.0.0 might differ (e.g., 'bw' record might be inside the folder)
# We will try to plot 'bw' record if it exists, otherwise generic noise
# Note: NSTDB often requires mapping specific records. Let's try standard names if they exist.

print("Visualizing Noise Types...")
plot_ecg_segment(NSTDB_PATH, "bw", title="Noise: Baseline Wander")
plot_ecg_segment(NSTDB_PATH, "ma", title="Noise: Muscle Artifact")
plot_ecg_segment(NSTDB_PATH, "em", title="Noise: Electrode Motion")