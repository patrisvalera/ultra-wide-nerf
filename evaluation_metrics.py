"""
Evaluation metrics for 3D ultrasound volume reconstruction.

Implements comprehensive metrics for comparing predicted and target volumes:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- NRMSE (Normalized Root Mean Squared Error)
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- LPIPS (Learned Perceptual Image Patch Similarity)
- SNR (Signal-to-Noise Ratio)
- Correlation Coefficient
- MI/NMI (Mutual Information / Normalized Mutual Information)

Supports both 3D volume metrics and 2D slice metrics.
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
import lpips


class VolumeMetrics:
    """
    Comprehensive metrics calculator for 3D ultrasound volumes and 2D slices.
    """
    
    def __init__(self, device=None):
        """
        Initialize metrics calculator.
        
        Args:
            device: Device for LPIPS computation ('cuda' or 'cpu'). Defaults to CUDA if available.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        # Initialize LPIPS model (alexnet perceptual loss)
        # Use 'alex' network as it's faster and works well for medical images
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()
    
    def normalize_data(self, data, method='minmax'):
        """
        Normalize data to [0, 1] range.
        
        Args:
            data: Input array (numpy or torch)
            method: 'minmax' or 'percentile' (robust to outliers)
        
        Returns:
            Normalized array in [0, 1]
        """
        is_torch = isinstance(data, torch.Tensor)
        
        if method == 'minmax':
            data_min = data.min()
            data_max = data.max()
            if data_max - data_min < 1e-8:
                return data * 0.0  # Avoid division by zero
            normalized = (data - data_min) / (data_max - data_min)
        elif method == 'percentile':
            # Use 1st and 99th percentiles to be robust to outliers
            if is_torch:
                p1 = torch.quantile(data.flatten(), 0.01)
                p99 = torch.quantile(data.flatten(), 0.99)
            else:
                p1 = np.percentile(data, 1)
                p99 = np.percentile(data, 99)
            
            if p99 - p1 < 1e-8:
                return data * 0.0
            normalized = (data - p1) / (p99 - p1)
            normalized = normalized.clamp(0, 1) if is_torch else np.clip(normalized, 0, 1)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def compute_mse(self, pred, target):
        """
        Compute Mean Squared Error.
        
        Args:
            pred: Predicted volume/slice (numpy array)
            target: Target volume/slice (numpy array)
        
        Returns:
            float: MSE value
        """
        return np.mean((pred - target) ** 2)
    
    def compute_mae(self, pred, target):
        """
        Compute Mean Absolute Error.
        
        Args:
            pred: Predicted volume/slice (numpy array)
            target: Target volume/slice (numpy array)
        
        Returns:
            float: MAE value
        """
        return np.mean(np.abs(pred - target))
    
    def compute_rmse(self, pred, target):
        """
        Compute Root Mean Squared Error.
        
        Args:
            pred: Predicted volume/slice (numpy array)
            target: Target volume/slice (numpy array)
        
        Returns:
            float: RMSE value
        """
        return np.sqrt(self.compute_mse(pred, target))
    
    def compute_nrmse(self, pred, target, normalization='range'):
        """
        Compute Normalized Root Mean Squared Error.
        
        Args:
            pred: Predicted volume/slice (numpy array)
            target: Target volume/slice (numpy array)
            normalization: 'range' (max-min), 'mean', or 'std'
        
        Returns:
            float: NRMSE value
        """
        rmse = self.compute_rmse(pred, target)
        
        if normalization == 'range':
            # Normalize by range of target
            norm_factor = target.max() - target.min()
        elif normalization == 'mean':
            # Normalize by mean of target
            norm_factor = target.mean()
        elif normalization == 'std':
            # Normalize by standard deviation of target
            norm_factor = target.std()
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        if norm_factor < 1e-8:
            return float('nan')  # Undefined for a constant target
        
        return rmse / norm_factor
    
    def compute_psnr(self, pred, target, data_range=None):
        """
        Compute Peak Signal-to-Noise Ratio.
        
        Args:
            pred: Predicted volume/slice (numpy array)
            target: Target volume/slice (numpy array)
            data_range: Range of data (default: max-min of target)
        
        Returns:
            float: PSNR value in dB
        """
        if data_range is None:
            data_range = target.max() - target.min()
        
        if data_range < 1e-8:
            return float('nan')  # Undefined for a constant target
        
        mse = self.compute_mse(pred, target)
        if mse < 1e-10:
            return float('inf')  # Perfect match
        
        return 20 * np.log10(data_range / np.sqrt(mse))
    
    def compute_snr(self, pred, target):
        """
        Compute Signal-to-Noise Ratio.
        
        Args:
            pred: Predicted volume/slice (numpy array)
            target: Target volume/slice (numpy array)
        
        Returns:
            float: SNR value in dB
        """
        signal_power = np.mean(target ** 2)
        noise_power = np.mean((pred - target) ** 2)
        
        if noise_power < 1e-10:
            return float('inf')  # Perfect match
        if signal_power < 1e-10:
            return float('-inf')  # No signal
        
        return 10 * np.log10(signal_power / noise_power)
    
    def compute_correlation(self, pred, target):
        """
        Compute Pearson correlation coefficient.
        
        Args:
            pred: Predicted volume/slice (numpy array)
            target: Target volume/slice (numpy array)
        
        Returns:
            float: Correlation coefficient in [-1, 1]
        """
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        if pred_flat.std() < 1e-12 or target_flat.std() < 1e-12:
            return float('nan')  # Undefined when either input is constant

        return np.corrcoef(pred_flat, target_flat)[0, 1]
    
    def compute_mutual_information(self, pred, target, bins=256):
        """
        Compute Mutual Information and Normalized Mutual Information between
        predicted and target volumes from a single binning.

        MI measures the amount of information shared between two images.
        Higher values indicate better alignment and similarity.

        Args:
            pred: Predicted volume/slice (numpy array)
            target: Target volume/slice (numpy array)
            bins: Number of bins for histogram computation (default: 256)

        Returns:
            tuple: (mi, nmi) -- raw MI in nats (unbounded) and NMI in [0, 1]
        """
        # Normalize to [0, 1] and discretize into bins
        pred_norm = self.normalize_data(pred, method='minmax')
        target_norm = self.normalize_data(target, method='minmax')

        pred_bins = np.clip((pred_norm.flatten() * (bins - 1)).astype(int), 0, bins - 1)
        target_bins = np.clip((target_norm.flatten() * (bins - 1)).astype(int), 0, bins - 1)

        mi = mutual_info_score(target_bins, pred_bins)
        nmi = normalized_mutual_info_score(target_bins, pred_bins, average_method='arithmetic')
        return mi, nmi
    
    def compute_joint_histogram(self, pred, target, bins=256):
        """
        Compute joint histogram for MI computation.
        
        This is useful for visualizing the relationship between predicted and target values.
        
        Args:
            pred: Predicted volume/slice (numpy array)
            target: Target volume/slice (numpy array)
            bins: Number of bins for histogram
        
        Returns:
            tuple: (joint_hist, pred_edges, target_edges)
        """
        # Normalize to [0, 1]
        pred_norm = self.normalize_data(pred, method='minmax')
        target_norm = self.normalize_data(target, method='minmax')
        
        # Compute 2D histogram
        joint_hist, pred_edges, target_edges = np.histogram2d(
            pred_norm.flatten(),
            target_norm.flatten(),
            bins=bins,
            range=[[0, 1], [0, 1]]
        )
        
        return joint_hist, pred_edges, target_edges
    
    def compute_ssim_3d(self, pred, target, data_range=None):
        """
        Compute 3D Structural Similarity Index.
        
        Uses slice-by-slice SSIM and averages across depth dimension.
        This is more stable than full 3D SSIM for anisotropic ultrasound data.
        
        Args:
            pred: Predicted volume (numpy array, shape: [D, H, W])
            target: Target volume (numpy array, shape: [D, H, W])
            data_range: Range of data (default: max-min of target)
        
        Returns:
            float: Average SSIM value in [0, 1]
        """
        if data_range is None:
            data_range = target.max() - target.min()
        
        if pred.ndim != 3 or target.ndim != 3:
            raise ValueError(f"Expected 3D volumes, got shapes {pred.shape} and {target.shape}")
        
        ssim_values = []
        for d in range(pred.shape[0]):
            # Compute SSIM for each slice
            ssim_val = ssim_skimage(
                target[d],
                pred[d],
                data_range=data_range,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False
            )
            ssim_values.append(ssim_val)

        return np.mean(ssim_values)
    
    def compute_ssim_2d(self, pred, target, data_range=None):
        """
        Compute 2D Structural Similarity Index for a single slice.
        
        Args:
            pred: Predicted slice (numpy array, shape: [H, W])
            target: Target slice (numpy array, shape: [H, W])
            data_range: Range of data (default: max-min of target)
        
        Returns:
            float: SSIM value in [0, 1]
        """
        if data_range is None:
            data_range = target.max() - target.min()
        
        if pred.ndim != 2 or target.ndim != 2:
            raise ValueError(f"Expected 2D slices, got shapes {pred.shape} and {target.shape}")
        
        return ssim_skimage(
            target,
            pred,
            data_range=data_range,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False
        )
    
    def compute_lpips_3d(self, pred, target):
        """
        Compute 3D LPIPS (Learned Perceptual Image Patch Similarity).
        
        Averages LPIPS across all slices in the depth dimension.
        Lower values indicate better perceptual similarity.
        
        Args:
            pred: Predicted volume (numpy array, shape: [D, H, W])
            target: Target volume (numpy array, shape: [D, H, W])
        
        Returns:
            float: Average LPIPS value (lower is better)
        """
        if pred.ndim != 3 or target.ndim != 3:
            raise ValueError(f"Expected 3D volumes, got shapes {pred.shape} and {target.shape}")
        
        # Normalize to [0, 1]
        pred_norm = self.normalize_data(pred, method='minmax')
        target_norm = self.normalize_data(target, method='minmax')
        
        lpips_values = []
        
        with torch.no_grad():
            for d in range(pred.shape[0]):
                # Convert to torch tensors and add batch + channel dimensions
                # LPIPS expects input in range [-1, 1], so we transform [0, 1] -> [-1, 1]
                pred_slice = torch.from_numpy(pred_norm[d]).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                target_slice = torch.from_numpy(target_norm[d]).float().unsqueeze(0).unsqueeze(0)
                
                # Repeat grayscale to 3 channels (LPIPS expects RGB)
                pred_slice = pred_slice.repeat(1, 3, 1, 1).to(self.device)  # [1, 3, H, W]
                target_slice = target_slice.repeat(1, 3, 1, 1).to(self.device)
                
                # Transform [0, 1] -> [-1, 1]
                pred_slice = pred_slice * 2.0 - 1.0
                target_slice = target_slice * 2.0 - 1.0

                lpips_val = self.lpips_model(pred_slice, target_slice)
                lpips_values.append(lpips_val.item())
        
        return np.mean(lpips_values)
    
    def compute_lpips_2d(self, pred, target):
        """
        Compute 2D LPIPS for a single slice.
        
        Args:
            pred: Predicted slice (numpy array, shape: [H, W])
            target: Target slice (numpy array, shape: [H, W])
        
        Returns:
            float: LPIPS value (lower is better)
        """
        if pred.ndim != 2 or target.ndim != 2:
            raise ValueError(f"Expected 2D slices, got shapes {pred.shape} and {target.shape}")
        
        # Normalize to [0, 1]
        pred_norm = self.normalize_data(pred, method='minmax')
        target_norm = self.normalize_data(target, method='minmax')
        
        with torch.no_grad():
            # Convert to torch tensors and add batch + channel dimensions
            pred_tensor = torch.from_numpy(pred_norm).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            target_tensor = torch.from_numpy(target_norm).float().unsqueeze(0).unsqueeze(0)
            
            # Repeat grayscale to 3 channels
            pred_tensor = pred_tensor.repeat(1, 3, 1, 1).to(self.device)  # [1, 3, H, W]
            target_tensor = target_tensor.repeat(1, 3, 1, 1).to(self.device)
            
            # Transform [0, 1] -> [-1, 1]
            pred_tensor = pred_tensor * 2.0 - 1.0
            target_tensor = target_tensor * 2.0 - 1.0

            return self.lpips_model(pred_tensor, target_tensor).item()
    
    def compute_all_metrics_3d(self, pred, target):
        """
        Compute all metrics for 3D volume.
        
        Args:
            pred: Predicted volume (numpy array, shape: [D, H, W])
            target: Target volume (numpy array, shape: [D, H, W])
        
        Returns:
            dict: Dictionary with all metric values
        """
        data_range = target.max() - target.min()
        mi, nmi = self.compute_mutual_information(pred, target, bins=256)
        
        metrics = {
            'mse': self.compute_mse(pred, target),
            'mae': self.compute_mae(pred, target),
            'rmse': self.compute_rmse(pred, target),
            'nrmse': self.compute_nrmse(pred, target, normalization='range'),
            'psnr': self.compute_psnr(pred, target, data_range=data_range),
            'ssim': self.compute_ssim_3d(pred, target, data_range=data_range),
            'lpips': self.compute_lpips_3d(pred, target),
            'snr': self.compute_snr(pred, target),
            'correlation': self.compute_correlation(pred, target),
            'mi': mi,
            'nmi': nmi
        }
        
        return metrics
    
    def compute_all_metrics_2d(self, pred, target):
        """
        Compute all metrics for 2D slice.
        
        Args:
            pred: Predicted slice (numpy array, shape: [H, W])
            target: Target slice (numpy array, shape: [H, W])
        
        Returns:
            dict: Dictionary with all metric values
        """
        data_range = target.max() - target.min()
        mi, nmi = self.compute_mutual_information(pred, target, bins=256)
        
        metrics = {
            'mse': self.compute_mse(pred, target),
            'mae': self.compute_mae(pred, target),
            'rmse': self.compute_rmse(pred, target),
            'nrmse': self.compute_nrmse(pred, target, normalization='range'),
            'psnr': self.compute_psnr(pred, target, data_range=data_range),
            'ssim': self.compute_ssim_2d(pred, target, data_range=data_range),
            'lpips': self.compute_lpips_2d(pred, target),
            'snr': self.compute_snr(pred, target),
            'correlation': self.compute_correlation(pred, target),
            'mi': mi,
            'nmi': nmi
        }
        
        return metrics
    
    @staticmethod
    def format_metrics(metrics, prefix=""):
        """
        Format metrics dictionary into a readable string.
        
        Args:
            metrics: Dictionary of metric values
            prefix: Optional prefix for each line
        
        Returns:
            str: Formatted string
        """
        lines = []
        lines.append(f"{prefix}MSE:         {metrics['mse']:.6f}")
        lines.append(f"{prefix}MAE:         {metrics['mae']:.6f}")
        lines.append(f"{prefix}RMSE:        {metrics['rmse']:.6f}")
        lines.append(f"{prefix}NRMSE:       {metrics['nrmse']:.6f}")
        lines.append(f"{prefix}PSNR:        {metrics['psnr']:.2f} dB")
        lines.append(f"{prefix}SSIM:        {metrics['ssim']:.4f}")
        lines.append(f"{prefix}LPIPS:       {metrics['lpips']:.4f}")
        lines.append(f"{prefix}SNR:         {metrics['snr']:.2f} dB")
        lines.append(f"{prefix}Correlation: {metrics['correlation']:.4f}")
        lines.append(f"{prefix}MI:          {metrics['mi']:.4f}")
        lines.append(f"{prefix}NMI:         {metrics['nmi']:.4f}")
        
        return "\n".join(lines)


_calculators = {}


def get_metrics_calculator(device=None):
    """Return one VolumeMetrics per device, so the LPIPS network is loaded only once."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    key = str(device)
    if key not in _calculators:
        _calculators[key] = VolumeMetrics(device=device)
    return _calculators[key]


def normalize_minmax(volume):
    """Scale a volume to [0, 1] by its own min and max (a constant volume maps to 0)."""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)
    return (volume - vmin) / (vmax - vmin)


def evaluate_volume_with_slices(pred_volume, target_volume, device=None):
    """
    Comprehensive evaluation of 3D volume with 2D slice analysis.

    Prediction and target are each min-max normalized to [0, 1] by their own range
    before any metric is computed, so training and evaluation use the same convention.

    Computes:
    1. 3D volume metrics
    2. Coronal middle slice metrics (volume[:, H//2, :])
    3. Axial middle slice metrics (volume[:, :, W//2])

    Args:
        pred_volume: Predicted volume (numpy array, shape: [D, H, W])
        target_volume: Target volume (numpy array, shape: [D, H, W])
        device: Device for LPIPS computation (defaults to CUDA if available)

    Returns:
        dict: Dictionary containing '3d', 'coronal_slice', and 'axial_slice' metrics
    """
    if pred_volume.shape != target_volume.shape:
        raise ValueError(f"Shape mismatch: pred {pred_volume.shape} vs target {target_volume.shape}")

    pred_volume = normalize_minmax(pred_volume)
    target_volume = normalize_minmax(target_volume)

    calculator = get_metrics_calculator(device)

    results = {}
    
    # 3D volume metrics
    print("Computing 3D volume metrics...")
    results['3d'] = calculator.compute_all_metrics_3d(pred_volume, target_volume)
    
    # Coronal middle slice: volume[:, H//2, :]
    h_mid = pred_volume.shape[1] // 2
    coronal_pred = pred_volume[:, h_mid, :]
    coronal_target = target_volume[:, h_mid, :]
    
    print(f"Computing coronal slice metrics (y={h_mid})...")
    results['coronal_slice'] = calculator.compute_all_metrics_2d(coronal_pred, coronal_target)
    
    # Axial middle slice: volume[:, :, W//2]
    w_mid = pred_volume.shape[2] // 2
    axial_pred = pred_volume[:, :, w_mid]
    axial_target = target_volume[:, :, w_mid]
    
    print(f"Computing axial slice metrics (x={w_mid})...")
    results['axial_slice'] = calculator.compute_all_metrics_2d(axial_pred, axial_target)
    
    return results


def print_evaluation_results(results):
    """
    Pretty print evaluation results.
    
    Args:
        results: Dictionary from evaluate_volume_with_slices()
    """
    calculator = VolumeMetrics
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\n3D VOLUME METRICS:")
    print("-" * 70)
    print(calculator.format_metrics(results['3d'], prefix="  "))
    
    print("\n\nCORONAL MIDDLE SLICE METRICS (volume[:, H//2, :]):")
    print("-" * 70)
    print(calculator.format_metrics(results['coronal_slice'], prefix="  "))
    
    print("\n\nAXIAL MIDDLE SLICE METRICS (volume[:, :, W//2]):")
    print("-" * 70)
    print(calculator.format_metrics(results['axial_slice'], prefix="  "))
    
    print("\n" + "="*70)


def save_metrics_to_file(results, filepath):
    """
    Save metrics to text file.
    
    Args:
        results: Dictionary from evaluate_volume_with_slices()
        filepath: Path to save file
    """
    calculator = VolumeMetrics
    
    with open(filepath, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("3D VOLUME METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write(calculator.format_metrics(results['3d'], prefix="  ") + "\n\n")
        
        f.write("CORONAL MIDDLE SLICE METRICS (volume[:, H//2, :]):\n")
        f.write("-" * 70 + "\n")
        f.write(calculator.format_metrics(results['coronal_slice'], prefix="  ") + "\n\n")
        
        f.write("AXIAL MIDDLE SLICE METRICS (volume[:, :, W//2]):\n")
        f.write("-" * 70 + "\n")
        f.write(calculator.format_metrics(results['axial_slice'], prefix="  ") + "\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"Metrics saved to: {filepath}")
