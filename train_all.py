"""
Master End-to-End Pipeline Runner.
1. Generates Spatio-Temporal Datasets & Telemetry.
2. Trains Spatio-Temporal Graph Neural Network (ST-GNN).
3. Trains Reinforcement Learning Dispatcher (DQN).
4. Trains Predictive Maintenance Autoencoder.
5. Benchmarks Trajectory Energy Optimization and Multi-Objective Routing.
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, CHECKPOINT_DIR
from data.network_topology import MetroTopology
from data.data_generator import MetroDataGenerator
from models.st_gnn import train_st_gnn
from models.rl_dispatcher import train_rl_dispatcher
from models.anomaly_detector import PredictiveMaintenanceEngine
from models.energy_optimizer import TrainTrajectoryOptimizer
from models.router import MetroRouteFinder


def run_full_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("🚇 AI METRO INTELLIGENCE & ROUTE OPTIMIZATION: MASTER PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Network Topology & Data Generation
    # -------------------------------------------------------------------------
    print("\n[STAGE 1/5] Building Network Topology & Synthesizing Data...")
    topology = MetroTopology()
    data_gen = MetroDataGenerator(topology)
    data_meta = data_gen.save_all_datasets()
    print(f"✅ Generated flow tensor of shape: {data_meta['tensor_shape']}")
    print(f"✅ Generated telemetry dataset of {data_meta['telemetry_rows']} samples with {data_meta['anomaly_count']} anomalies.")

    # -------------------------------------------------------------------------
    # Step 2: Spatio-Temporal Graph Neural Network (ST-GNN)
    # -------------------------------------------------------------------------
    print("\n[STAGE 2/5] Training Spatio-Temporal Graph Neural Network (ST-GNN)...")
    flow_tensor = np.load(DATA_DIR / "passenger_flow_tensor.npy")
    st_model, st_history = train_st_gnn(flow_tensor, topology, epochs=12, device="cpu")
    print(f"✅ ST-GNN Training Complete -> Final Test MAE: {st_history['test_mae']:.2f} passengers/step | RMSE: {st_history['test_rmse']:.2f}")

    # -------------------------------------------------------------------------
    # Step 3: Deep Reinforcement Learning Headway Dispatcher (DQN)
    # -------------------------------------------------------------------------
    print("\n[STAGE 3/5] Training Reinforcement Learning Headway Dispatcher...")
    rl_agent, rl_rewards = train_rl_dispatcher(episodes=60)
    print(f"✅ RL Dispatcher Training Complete -> Final Mean Reward: {np.mean(rl_rewards[-10:]):.2f}")

    # -------------------------------------------------------------------------
    # Step 4: Predictive Maintenance Deep Autoencoder
    # -------------------------------------------------------------------------
    print("\n[STAGE 4/5] Training Predictive Maintenance Autoencoder...")
    telemetry_df = pd.read_csv(DATA_DIR / "telemetry_data.csv")
    pdm_engine = PredictiveMaintenanceEngine()
    pdm_metrics = pdm_engine.fit(telemetry_df, epochs=15)
    print(f"✅ Maintenance Model Complete -> ROC-AUC: {pdm_metrics['roc_auc']} | F1-Score: {pdm_metrics['f1_score']}")

    # -------------------------------------------------------------------------
    # Step 5: Physics Trajectory & Multi-Objective Route Optimization
    # -------------------------------------------------------------------------
    print("\n[STAGE 5/5] Evaluating Physics Trajectory & Pareto Route Optimization...")
    traj_opt = TrainTrajectoryOptimizer()
    energy_res = traj_opt.compare_driving_strategies(distance_km=2.8, target_time_sec=170.0)
    print(f"⚡ Energy Saving Result: {energy_res['energy_saved_kwh']} kWh saved ({energy_res['savings_percent']}%) per inter-station run.")

    router = MetroRouteFinder(topology)
    sample_routes = router.find_routes("Rajiv Chowk (CP)", "IGI Airport Terminal 3")
    print(f"🧭 Delhi Metro Route Optimization (Rajiv Chowk (CP) -> IGI Airport Terminal 3):")
    for mode, r in sample_routes.items():
        print(f"   • {mode.upper()}: {r['total_time_min']} mins | Crowd Index: {r['avg_crowd_score']} | Transfers: {r['transfers']}")

    total_duration = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 ALL AI/ML MODULES SUCCESSFULLY TRAINED & VALIDATED in {total_duration:.1f}s!")
    print(f"Model Checkpoints persisted in: {CHECKPOINT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
