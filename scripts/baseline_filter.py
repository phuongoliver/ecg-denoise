import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def baseline_filter(x, fs=360.0, notch_freq=50.0):  # x: (L,) or (B,L)
    def _apply(sig):
        # 1) high-pass 0.5 Hz (remove baseline wander)
        b, a = butter(2, 0.5/(fs/2), btype='high')
        y = filtfilt(b, a, sig)
        # 2) notch 50 Hz - match Vietnam power grid
        if notch_freq:
            b0, a0 = iirnotch(w0=notch_freq/(fs/2), Q=30)
            y = filtfilt(b0, a0, y)
        # 3) low-pass 40 Hz (giữ hầu hết morphology ECG chẩn đoán)
        b2, a2 = butter(2, 40.0/(fs/2), btype='low')
        y = filtfilt(b2, a2, y)
        return y.astype(np.float32)
    return np.apply_along_axis(_apply, -1, x) if x.ndim==2 else _apply(x)