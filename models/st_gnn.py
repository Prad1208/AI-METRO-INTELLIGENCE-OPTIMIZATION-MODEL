"""
Spatio-Temporal Graph Neural Network (ST-GNN) for Metro Flow & Congestion Forecasting.
Combines Chebyshev Spectral Graph Convolutions with Temporal Gated Convolutions (GLU/GRU).
Strictly adheres to chronological splitting and zero-leakage normalization.
"""
from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from config import ST_GNN_CONFIG, CHECKPOINT_DIR
from data.network_topology import MetroTopology


class ChebGraphConv(nn.Module):
    """
    Chebyshev Spectral Graph Convolutional Layer.
    Computes: X' = sum_{k=0}^{K-1} T_k(L_tilde) * X * W_k
    """

    def __init__(self, in_features: int, out_features: int, K: int = 2):
        super().__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        # Weight tensor for K Chebyshev polynomials: (K, in_features, out_features)
        self.weights = nn.Parameter(torch.FloatTensor(K, in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, cheb_polys: List[torch.Tensor]) -> torch.Tensor:
        """
        x: (batch_size, num_nodes, in_features)
        cheb_polys: list of K tensors, each of shape (num_nodes, num_nodes)
        returns: (batch_size, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = x.shape
        out = torch.zeros(batch_size, num_nodes, self.out_features, device=x.device)

        for k in range(self.K):
            # T_k * x -> (batch, nodes, in_features)
            # cheb_polys[k] is (nodes, nodes)
            T_k = cheb_polys[k] # (N, N)
            # Batch matrix multiply: T_k @ x
            x_k = torch.einsum("nm,bmf->bnf", T_k, x)
            # x_k * W_k -> (batch, nodes, out_features)
            out = out + torch.matmul(x_k, self.weights[k])

        out = out + self.bias
        return F.relu(out)


class SpatioTemporalBlock(nn.Module):
    """
    ST-GCN Block: Temporal 1D Convolution -> Spatial Cheb Conv -> Temporal 1D Convolution
    with residual skip connections and LayerNorm.
    """

    def __init__(self, in_channels: int, hidden_dim: int, K: int = 2, kernel_size: int = 3):
        super().__init__()
        self.temporal_conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.spatial_conv = ChebGraphConv(hidden_dim, hidden_dim, K=K)
        self.temporal_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        
        # Residual projection if channel dimensions differ
        self.residual_conv = nn.Conv1d(in_channels, hidden_dim, 1) if in_channels != hidden_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x: torch.Tensor, cheb_polys: List[torch.Tensor]) -> torch.Tensor:
        """
        x: (batch_size, timesteps, num_nodes, features)
        """
        batch_size, T, N, F_dim = x.shape
        res = x # (B, T, N, F_dim)

        # 1. Temporal Conv 1: Reshape to (B * N, F_dim, T)
        x_t = x.permute(0, 2, 3, 1).reshape(batch_size * N, F_dim, T)
        t1 = F.glu(self.temporal_conv1(x_t), dim=1) # Half channels if GLU, or use regular Conv
        # Let's adjust for standard channels:
        # Instead of glu, use relu
        t1 = F.relu(self.temporal_conv1(x_t)) # (B * N, hidden_dim, T)

        # Reshape back to (B, T, N, hidden_dim)
        x_s = t1.reshape(batch_size, N, -1, T).permute(0, 3, 1, 2)

        # 2. Spatial Cheb Conv applied at each timestep
        spatial_outs = []
        for t in range(T):
            x_step = x_s[:, t, :, :] # (B, N, hidden_dim)
            s_out = self.spatial_conv(x_step, cheb_polys)
            spatial_outs.append(s_out)
        s_combined = torch.stack(spatial_outs, dim=1) # (B, T, N, hidden_dim)

        # 3. Temporal Conv 2
        s_t = s_combined.permute(0, 2, 3, 1).reshape(batch_size * N, -1, T)
        t2 = F.relu(self.temporal_conv2(s_t))
        t2_out = t2.reshape(batch_size, N, -1, T).permute(0, 3, 1, 2)

        # 4. Residual Connection + LayerNorm
        res_t = x.permute(0, 2, 3, 1).reshape(batch_size * N, F_dim, T)
        res_proj = self.residual_conv(res_t).reshape(batch_size, N, -1, T).permute(0, 3, 1, 2)
        out = self.layer_norm(t2_out + res_proj)
        return self.dropout(out)


class MetroSTGNN(nn.Module):
    """
    Complete Spatio-Temporal Graph Neural Network Architecture for Metro Demand Prediction.
    Maps historical sequence (B, T_in, N, F_in) -> (B, T_out, N, F_out).
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int = 3,
        out_features: int = 3,
        input_timesteps: int = 12,
        output_timesteps: int = 3,
        hidden_dim: int = 64,
        K: int = 2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps

        # Stacked Spatio-Temporal GNN Blocks
        self.st_block1 = SpatioTemporalBlock(in_features, hidden_dim, K=K)
        self.st_block2 = SpatioTemporalBlock(hidden_dim, hidden_dim, K=K)

        # Global Temporal Pooling & Prediction Head
        self.temporal_fc = nn.Linear(input_timesteps, output_timesteps)
        self.output_layer = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor, cheb_polys: List[torch.Tensor]) -> torch.Tensor:
        """
        x: (batch, T_in, N, F_in)
        returns: (batch, T_out, N, F_out)
        """
        h1 = self.st_block1(x, cheb_polys)
        h2 = self.st_block2(h1, cheb_polys) # (B, T_in, N, hidden_dim)

        # Project time dimension: (B, N, hidden_dim, T_in) -> (B, N, hidden_dim, T_out)
        h_proj = h2.permute(0, 2, 3, 1) # (B, N, hidden_dim, T_in)
        h_time = self.temporal_fc(h_proj).permute(0, 3, 1, 2) # (B, T_out, N, hidden_dim)

        # Final projection to output features (Inflow, Outflow, Crowd Ratio)
        out = self.output_layer(h_time) # (B, T_out, N, F_out)
        return out


# ---------------------------------------------------------------------------
# Training & Dataset Utilities (ML Best Practices)
# ---------------------------------------------------------------------------

class MetroTimeSeriesDataset:
    """
    Prepares sliding windows with strict chronological ordering (no future leakage).
    """

    def __init__(self, tensor: np.ndarray, t_in: int = 12, t_out: int = 3):
        self.raw_tensor = tensor # (Total_T, N, F)
        self.t_in = t_in
        self.t_out = t_out
        
        # Chronological splits (70% Train, 15% Val, 15% Test)
        total_len = len(tensor)
        train_end = int(total_len * ST_GNN_CONFIG["train_ratio"])
        val_end = int(total_len * (ST_GNN_CONFIG["train_ratio"] + ST_GNN_CONFIG["val_ratio"]))

        self.train_data = tensor[:train_end]
        self.val_data = tensor[train_end:val_end]
        self.test_data = tensor[val_end:]

        # Fit Scaler strictly on training partition
        self.mean = np.mean(self.train_data, axis=(0, 1), keepdims=True)
        self.std = np.std(self.train_data, axis=(0, 1), keepdims=True) + 1e-6

        # Standardize partitions
        self.train_norm = (self.train_data - self.mean) / self.std
        self.val_norm = (self.val_data - self.mean) / self.std
        self.test_norm = (self.test_data - self.mean) / self.std

    def _create_sliding_windows(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        X_list, Y_list = [], []
        num_steps = len(data) - self.t_in - self.t_out + 1
        for i in range(num_steps):
            x = data[i : i + self.t_in]
            y = data[i + self.t_in : i + self.t_in + self.t_out]
            X_list.append(x)
            Y_list.append(y)
        return torch.tensor(np.array(X_list), dtype=torch.float32), torch.tensor(np.array(Y_list), dtype=torch.float32)

    def get_dataloaders(self, batch_size: int = 32):
        X_train, Y_train = self._create_sliding_windows(self.train_norm)
        X_val, Y_val = self._create_sliding_windows(self.val_norm)
        X_test, Y_test = self._create_sliding_windows(self.test_norm)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val, Y_val),
            batch_size=batch_size,
            shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_test, Y_test),
            batch_size=batch_size,
            shuffle=False
        )
        return train_loader, val_loader, test_loader

    def denormalize(self, tensor_norm: torch.Tensor) -> np.ndarray:
        """
        Converts normalized predictions back to original passenger flow counts.
        """
        arr = tensor_norm.detach().cpu().numpy()
        return arr * self.std + self.mean


def train_st_gnn(
    tensor: np.ndarray,
    topology: MetroTopology,
    epochs: int = 20,
    device: str = "cpu"
) -> Tuple[MetroSTGNN, Dict[str, Any]]:
    """
    End-to-End training routine for Metro ST-GNN with validation checkpoints.
    """
    dataset = MetroTimeSeriesDataset(
        tensor,
        t_in=ST_GNN_CONFIG["num_timesteps_input"],
        t_out=ST_GNN_CONFIG["num_timesteps_output"]
    )
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=ST_GNN_CONFIG["batch_size"])

    # Prepare Chebyshev Spectral Polynomials on device
    cheb_np = topology.get_chebyshev_polynomials(k=ST_GNN_CONFIG["cheb_order"])
    cheb_tensors = [torch.tensor(m, dtype=torch.float32, device=device) for m in cheb_np]

    model = MetroSTGNN(
        num_nodes=topology.num_stations,
        in_features=3,
        out_features=3,
        input_timesteps=ST_GNN_CONFIG["num_timesteps_input"],
        output_timesteps=ST_GNN_CONFIG["num_timesteps_output"],
        hidden_dim=ST_GNN_CONFIG["hidden_dim"],
        K=len(cheb_tensors)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=ST_GNN_CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "test_mae": 0.0, "test_rmse": 0.0}

    print(f"🚀 Starting ST-GNN Training on {device} ({epochs} epochs)...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b, cheb_tensors)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation step
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                v_preds = model(x_v, cheb_tensors)
                v_loss = criterion(v_preds, y_v)
                val_losses.append(v_loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model_state": model.state_dict(),
                "scaler_mean": torch.tensor(dataset.mean, dtype=torch.float32),
                "scaler_std": torch.tensor(dataset.std, dtype=torch.float32),
                "config": ST_GNN_CONFIG
            }, CHECKPOINT_DIR / "st_gnn_best.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Best Val: {best_val_loss:.4f}")

    # Final Test Set Evaluation
    try:
        checkpoint = torch.load(CHECKPOINT_DIR / "st_gnn_best.pt", map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_DIR / "st_gnn_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_test_preds, all_test_targets = [], []
    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_t = x_t.to(device)
            p = model(x_t, cheb_tensors)
            all_test_preds.append(p.cpu())
            all_test_targets.append(y_t.cpu())

    test_preds_cat = torch.cat(all_test_preds, dim=0)
    test_targets_cat = torch.cat(all_test_targets, dim=0)

    # Denormalize to evaluate real-world passenger count error
    pred_real = dataset.denormalize(test_preds_cat)
    target_real = dataset.denormalize(test_targets_cat)

    test_mae = float(np.mean(np.abs(pred_real - target_real)))
    test_rmse = float(np.sqrt(np.mean((pred_real - target_real) ** 2)))
    history["test_mae"] = test_mae
    history["test_rmse"] = test_rmse
    print(f"🎯 Test Evaluation Complete -> MAE: {test_mae:.2f} passengers/step | RMSE: {test_rmse:.2f}")

    return model, history
