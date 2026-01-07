import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.utils.metrics import snr_db, count_params
from src.utils.baseline import baseline_filter

class TestUtils(unittest.TestCase):
    def test_snr_db_calculation(self):
        """Test SNR dB calculation correctness."""
        # Case 1: Identical signals -> Infinite SNR (practically very large due to eps)
        clean = torch.ones(2, 1, 100)
        est = torch.ones(2, 1, 100)
        snr = snr_db(clean, est)
        self.assertTrue(torch.all(snr > 100), "SNR for identical signals should be very high")

        # Case 2: Known ratio
        # Signal power = 1, Noise power = 0.1 -> SNR = 10log10(1/0.1) = 10dB
        clean = torch.ones(1, 1, 1000)
        noise = torch.sqrt(torch.tensor(0.1)) * torch.ones(1, 1, 1000) 
        # Note: simplistic approximation for checking formula logic
        # Ideally: SNR = 10 * log10 ( sum(x^2) / sum((x-y)^2) )
        
        val_clean = torch.tensor([10.0]) # Power = 100
        val_noise = torch.tensor([1.0])  # Power = 1
        # Manual check
        snr_val = 10 * torch.log10(val_clean / val_noise) # Should be 10
        
        # Test function
        res = snr_db(torch.sqrt(val_clean), torch.zeros_like(val_clean)) # est=0 -> noise=clean
        # wait, snr_db(clean, est). Noise = clean - est. 
        # If est=0, noise=clean. SNR = 0 dB.
        self.assertAlmostEqual(res.item(), 0.0, places=1)

    def test_count_params(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 2) # weights 2x10=20, bias 2. Total 22.
        self.assertEqual(count_params(model), 22)

    def test_baseline_filter_shape(self):
        """Test if baseline filter preserves shape."""
        # 1D input
        sig_1d = np.random.randn(1000)
        out_1d = baseline_filter(sig_1d, fs=360)
        self.assertEqual(sig_1d.shape, out_1d.shape)
        
        # 2D input (Batch, Length)
        sig_2d = np.random.randn(5, 1000)
        out_2d = baseline_filter(sig_2d, fs=360)
        self.assertEqual(sig_2d.shape, out_2d.shape)

if __name__ == '__main__':
    unittest.main()
