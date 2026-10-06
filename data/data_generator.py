"""
Realistic Synthetic Data Generator for Metro Passenger Flows, OD Matrices, and Telemetry.
Simulates diurnal rush hours, weekend patterns, weather impacts, stadium events, and sensor anomalies.
"""
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_DIR, DAYS_OF_SIMULATION, TIME_STEP_MINUTES
from data.network_topology import MetroTopology


class MetroDataGenerator:
    """
    Generates high-fidelity spatio-temporal transit data and mechanical telemetry.
    """

    def __init__(self, topology: MetroTopology, random_seed: int = 42):
        self.topology = topology
        self.rng = np.random.default_rng(random_seed)
        self.num_stations = topology.num_stations
        self.total_timesteps = (DAYS_OF_SIMULATION * 24 * 60) // TIME_STEP_MINUTES

    def _get_diurnal_multiplier(self, hour: float, is_weekend: bool, station_type: str) -> Tuple[float, float]:
        """
        Computes inflow and outflow multipliers based on time of day, day of week, and station type.
        """
        if is_weekend:
            # Weekend curve: late morning peak, steady afternoon
            in_mult = 0.2 + 0.6 * np.exp(-((hour - 14) ** 2) / 18.0)
            out_mult = 0.2 + 0.6 * np.exp(-((hour - 16) ** 2) / 18.0)
            return in_mult, out_mult

        # Weekday Curves
        if station_type == "residential":
            # High morning outflow (people leaving home), high evening inflow (people returning)
            in_mult = 0.1 + 0.9 * np.exp(-((hour - 18.5) ** 2) / 3.0) + 0.2 * np.exp(-((hour - 8.5) ** 2) / 4.0)
            out_mult = 0.1 + 0.95 * np.exp(-((hour - 8.0) ** 2) / 2.5) + 0.2 * np.exp(-((hour - 18.0) ** 2) / 4.0)
        elif station_type == "business":
            # High morning inflow (work arrival), high evening outflow (commute home)
            in_mult = 0.1 + 1.0 * np.exp(-((hour - 8.5) ** 2) / 2.5) + 0.2 * np.exp(-((hour - 18.0) ** 2) / 4.0)
            out_mult = 0.1 + 0.95 * np.exp(-((hour - 18.0) ** 2) / 2.5) + 0.2 * np.exp(-((hour - 8.5) ** 2) / 4.0)
        elif station_type == "stadium":
            # Base moderate flow, spikes during event hours (e.g. 7-10 PM)
            in_mult = 0.15 + 0.25 * np.sin(np.pi * hour / 24.0)
            out_mult = 0.15 + 0.25 * np.sin(np.pi * hour / 24.0)
        else: # Interchange / Mixed
            in_mult = 0.15 + 0.7 * np.exp(-((hour - 8.5) ** 2) / 4.0) + 0.75 * np.exp(-((hour - 18.0) ** 2) / 4.0)
            out_mult = 0.15 + 0.75 * np.exp(-((hour - 8.5) ** 2) / 4.0) + 0.7 * np.exp(-((hour - 18.0) ** 2) / 4.0)

        # Baseline night shutdown (1 AM - 5 AM)
        if 1.0 <= hour <= 5.0:
            in_mult *= 0.02
            out_mult *= 0.02

        return max(0.01, in_mult), max(0.01, out_mult)

    def _classify_station(self, name: str) -> str:
        """
        Classifies station into functional archetype for demand profiling.
        """
        if any(term in name for term in ["Rajiv Chowk", "New Delhi", "Gurugram", "Noida", "Central Secretariat", "Karol Bagh", "Mandi House"]):
            return "business"
        elif any(term in name for term in ["Dwarka", "Mayur Vihar", "Janakpuri", "Badli", "Vishwavidyalaya", "Kashmere"]):
            return "residential"
        elif "Stadium" in name or "Chandni" in name:
            return "stadium"
        return "mixed"

    def generate_passenger_flow_series(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generates full continuous spatio-temporal flow tensor:
        Shape: (num_timesteps, num_stations, 3) -> [Inflow, Outflow, Platform Crowd Ratio]
        """
        tensor = np.zeros((self.total_timesteps, self.num_stations, 3), dtype=np.float32)
        timestamps = []

        base_date = pd.Timestamp("2026-09-01 00:00:00")
        step_delta = pd.Timedelta(minutes=TIME_STEP_MINUTES)

        station_types = [self._classify_station(name) for name in self.topology.station_names]
        capacities = [self.topology.stations[name]["capacity"] for name in self.topology.station_names]

        # Tracking active platform queue
        platform_occupancy = np.zeros(self.num_stations, dtype=np.float32)

        for t in range(self.total_timesteps):
            current_time = base_date + t * step_delta
            timestamps.append(current_time)

            hour = current_time.hour + current_time.minute / 60.0
            is_weekend = current_time.dayofweek >= 5

            # Weather perturbation (e.g. rain on certain days)
            rain_factor = 1.18 if (current_time.day % 4 == 0 and 12 <= hour <= 19) else 1.0

            for i in range(self.num_stations):
                st_type = station_types[i]
                cap = capacities[i]
                in_mult, out_mult = self._get_diurnal_multiplier(hour, is_weekend, st_type)

                # Special Stadium Match Event on Saturday night
                if st_type == "stadium" and is_weekend and current_time.dayofweek == 5:
                    if 18.0 <= hour <= 19.5:
                        in_mult *= 3.5  # Surge arriving for game
                    elif 21.5 <= hour <= 23.0:
                        out_mult *= 4.0 # Surge departing stadium

                # Base nominal flow rate per 5-min step (scaled by station capacity)
                base_flow = cap * 0.08 * rain_factor
                noise_in = self.rng.normal(1.0, 0.06)
                noise_out = self.rng.normal(1.0, 0.06)

                inflow = float(max(0, base_flow * in_mult * noise_in))
                outflow = float(max(0, base_flow * out_mult * noise_out))

                # Update platform accumulation dynamics
                platform_occupancy[i] = max(0.0, platform_occupancy[i] + inflow - outflow * 0.95)
                # Cap platform occupancy
                platform_occupancy[i] = min(cap * 1.1, platform_occupancy[i])
                crowd_ratio = float(platform_occupancy[i] / cap)

                tensor[t, i, 0] = inflow
                tensor[t, i, 1] = outflow
                tensor[t, i, 2] = crowd_ratio

        # Save metadata dataframe
        meta_df = pd.DataFrame({"timestamp": timestamps})
        return tensor, meta_df

    def generate_sensor_telemetry(self, num_samples: int = 15000) -> pd.DataFrame:
        """
        Generates train bogie and motor telemetry with injected mechanical anomalies.
        Features: vibration_x, vibration_y, vibration_z, motor_current, bearing_temp, speed, is_anomaly
        """
        # Baseline normal healthy train operation
        speed = self.rng.uniform(20.0, 78.0, size=num_samples)
        
        # Vibration correlates with speed
        vib_x = 0.05 * (speed / 50.0) + self.rng.normal(0.12, 0.02, size=num_samples)
        vib_y = 0.04 * (speed / 50.0) + self.rng.normal(0.10, 0.02, size=num_samples)
        vib_z = 0.08 * (speed / 50.0) + self.rng.normal(0.18, 0.03, size=num_samples)
        
        motor_current = 280.0 * (speed / 60.0) + self.rng.normal(30.0, 15.0, size=num_samples)
        bearing_temp = 45.0 + 0.25 * speed + self.rng.normal(0.0, 1.5, size=num_samples)
        
        is_anomaly = np.zeros(num_samples, dtype=int)
        
        # Inject ~5% realistic mechanical faults
        num_anomalies = int(num_samples * 0.05)
        anomaly_indices = self.rng.choice(num_samples, size=num_anomalies, replace=False)
        
        for idx in anomaly_indices:
            fault_type = self.rng.choice(["wheel_flat", "bearing_overheat", "motor_surge"])
            if fault_type == "wheel_flat":
                # Severe vertical and radial vibration shockwaves
                vib_z[idx] += self.rng.uniform(0.35, 0.85)
                vib_x[idx] += self.rng.uniform(0.20, 0.50)
            elif fault_type == "bearing_overheat":
                bearing_temp[idx] += self.rng.uniform(25.0, 55.0)
                vib_y[idx] += self.rng.uniform(0.25, 0.60)
            elif fault_type == "motor_surge":
                motor_current[idx] += self.rng.uniform(150.0, 320.0)
            is_anomaly[idx] = 1

        df = pd.DataFrame({
            "speed_kmh": np.round(speed, 2),
            "vibration_x": np.round(vib_x, 4),
            "vibration_y": np.round(vib_y, 4),
            "vibration_z": np.round(vib_z, 4),
            "motor_current_a": np.round(motor_current, 2),
            "bearing_temp_c": np.round(bearing_temp, 2),
            "is_anomaly": is_anomaly
        })
        return df

    def save_all_datasets(self) -> Dict[str, Any]:
        """
        Executes full generation and persists datasets to disk.
        """
        print("Generating spatio-temporal metro passenger flow tensor...")
        tensor, meta_df = self.generate_passenger_flow_series()
        tensor_path = DATA_DIR / "passenger_flow_tensor.npy"
        np.save(tensor_path, tensor)
        meta_df.to_csv(DATA_DIR / "time_index.csv", index=False)
        print(f" Saved flow tensor: {tensor.shape} -> {tensor_path}")

        print("Generating rolling stock mechanical sensor telemetry...")
        telemetry_df = self.generate_sensor_telemetry()
        telemetry_path = DATA_DIR / "telemetry_data.csv"
        telemetry_df.to_csv(telemetry_path, index=False)
        print(f" Saved telemetry dataset: {len(telemetry_df)} rows -> {telemetry_path}")

        # Save network topology json
        topo_path = DATA_DIR / "network_topology.json"
        with open(topo_path, "w") as f:
            f.write(self.topology.export_topology_json())
        print(f" Exported network topology: {topo_path}")

        return {
            "tensor_shape": tensor.shape,
            "telemetry_rows": len(telemetry_df),
            "anomaly_count": int(telemetry_df["is_anomaly"].sum()),
            "tensor_path": str(tensor_path),
            "telemetry_path": str(telemetry_path)
        }


if __name__ == "__main__":
    topo = MetroTopology()
    gen = MetroDataGenerator(topo)
    res = gen.save_all_datasets()
    print("Data Generation Complete:", res)
