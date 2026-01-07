import unittest
import torch
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.models.architecture import DenoiseCNN_SWT_Transformer
from src.models.losses import PeakWeightedMSE

class TestDenoiseModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Standard ECG window size (e.g., 2 seconds @ 360Hz = 720 samples)
        self.input_shape = (2, 1, 720)  # (Batch, Channels, Length)
        self.dummy_input = torch.randn(*self.input_shape).to(self.device)
        self.model = DenoiseCNN_SWT_Transformer(in_ch=1, base_ch=16, nhead=2, ffn_mult=2, num_transformer_layers=1).to(self.device)

    def test_output_shape(self):
        """Test if the model output shape matches the input shape (denoising task)."""
        with torch.no_grad():
            output = self.model(self.dummy_input)
        
        self.assertEqual(output.shape, self.input_shape, 
                         f"Output shape {output.shape} mismatch with input {self.input_shape}")

    def test_loss_function(self):
        """Test if PeakWeightedMSE calculates valid loss."""
        loss_fn = PeakWeightedMSE(alpha=10.0)
        target = self.dummy_input.clone()
        # Add small noise to create differences
        prediction = target + 0.1 * torch.randn_like(target)
        
        loss = loss_fn(prediction, target)
        self.assertFalse(torch.isnan(loss), "Loss should not be NaN")
        self.assertGreaterEqual(loss.item(), 0.0, "Loss should be non-negative")

    def test_overfit_small_batch(self):
        """(Sanity Check) Test if model can reduce loss on a tiny batch."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        
        initial_loss = loss_fn(self.model(self.dummy_input), self.dummy_input).item()
        
        # Train for a few steps
        for _ in range(5):
            optimizer.zero_grad()
            out = self.model(self.dummy_input)
            loss = loss_fn(out, self.dummy_input)
            loss.backward()
            optimizer.step()
            
        final_loss = loss_fn(self.model(self.dummy_input), self.dummy_input).item()
        self.assertLess(final_loss, initial_loss, "Model failed to learn on a single batch (Sanity Check)")

class TestModelVariants(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_shape = (2, 1, 720) 
        self.dummy_input = torch.randn(*self.input_shape).to(self.device)

    def test_no_swt_variant(self):
        """Test DenoiseCNN_NoSWT (Ablation)."""
        from src.models.architecture import DenoiseCNN_NoSWT
        model = DenoiseCNN_NoSWT(in_ch=1, base_ch=16, nhead=2, ffn_mult=2, num_transformer_layers=1).to(self.device)
        with torch.no_grad():
            out = model(self.dummy_input)
        self.assertEqual(out.shape, self.input_shape, "No-SWT Variant output shape mismatch")

    def test_no_transformer_variant(self):
        """Test DenoiseCNN_NoTransformer (Ablation)."""
        from src.models.architecture import DenoiseCNN_NoTransformer
        model = DenoiseCNN_NoTransformer(in_ch=1, base_ch=16).to(self.device)
        with torch.no_grad():
            out = model(self.dummy_input)
        self.assertEqual(out.shape, self.input_shape, "No-Transformer Variant output shape mismatch")

    def test_baseline_cnn(self):
        """Test DenoiseCNN (Baseline Simple AE)."""
        from src.models.architecture import DenoiseCNN
        model = DenoiseCNN(in_ch=1, base_ch=16).to(self.device)
        with torch.no_grad():
            out = model(self.dummy_input)
        self.assertEqual(out.shape, self.input_shape, "Baseline CNN output shape mismatch")

if __name__ == '__main__':
    unittest.main()
