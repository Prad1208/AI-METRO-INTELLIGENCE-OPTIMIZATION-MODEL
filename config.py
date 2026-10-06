"""
Centralized Configuration for AI Metro Intelligence & Route Optimization System.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
EXPORTS_DIR = BASE_DIR / "exports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Metro Simulation Settings
TIME_STEP_MINUTES = 5          # Granularity of time-series observations
HOURS_PER_DAY = 24
DAYS_OF_SIMULATION = 14        # 2 weeks of synthetic operational data
SAMPLING_INTERVAL_SEC = 300    # 5 minutes in seconds

# Spatio-Temporal GNN Forecaster Settings
ST_GNN_CONFIG = {
    "num_timesteps_input": 12,   # Past 60 minutes (12 * 5 min)
    "num_timesteps_output": 3,   # Future 15 minutes (3 * 5 min)
    "hidden_dim": 64,
    "cheb_order": 2,             # Spatial Chebyshev polynomial order
    "learning_rate": 0.003,
    "batch_size": 32,
    "epochs": 25,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}

# Reinforcement Learning Dispatcher Settings
RL_DISPATCHER_CONFIG = {
    "min_headway_sec": 90,       # 1.5 min minimum safe headway
    "max_headway_sec": 420,      # 7 min maximum acceptable headway
    "target_headway_sec": 180,   # 3 min nominal headway
    "min_dwell_sec": 20,         # Minimum station stop time
    "max_dwell_sec": 90,         # Maximum dwell time
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "episodes": 150,
    "batch_size": 64,
}

# Train Physical Dynamics & Energy Optimization (Davis Equation)
TRAIN_DYNAMICS = {
    "empty_mass_tonnes": 220.0,  # 6-car metro train empty weight
    "passenger_capacity": 1500,  # Max crush load passengers
    "avg_passenger_weight_kg": 70.0,
    "max_traction_force_kn": 320.0,
    "max_braking_force_kn": 300.0,
    "max_speed_kmh": 80.0,
    "max_acceleration_mps2": 1.1,
    "max_service_braking_mps2": 1.1,
    "max_jerk_mps3": 0.75,
    # Davis Resistance Equation: R(v) = A + B*v + C*v^2 [kN, v in m/s]
    "davis_A": 2.50,             # Rolling & mechanical resistance
    "davis_B": 0.035,            # Flange & track interaction
    "davis_C": 0.0042,           # Aerodynamic drag coefficient
    "regen_efficiency": 0.82,    # Regenerative braking recovery efficiency
}

# Predictive Maintenance & Anomaly Detection
ANOMALY_CONFIG = {
    "telemetry_features": ["vibration_x", "vibration_y", "vibration_z", "motor_current", "bearing_temp"],
    "hidden_dims": [32, 16, 8, 16, 32],
    "learning_rate": 0.001,
    "epochs": 20,
    "contamination": 0.05,       # ~5% expected mechanical anomalies
}
