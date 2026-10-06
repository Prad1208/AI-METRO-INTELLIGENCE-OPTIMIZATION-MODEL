"""
Real-Time Metro Network Digital Twin & Dynamic Simulation Engine.
Synchronizes train movements, passenger accumulation, ST-GNN crowding forecasts,
and RL dynamic dispatching decisions across all metro lines.
"""
from typing import Dict, List, Any, Optional
import numpy as np
import time
from data.network_topology import MetroTopology
from models.router import MetroRouteFinder
from models.energy_optimizer import TrainTrajectoryOptimizer


class MetroDigitalTwin:
    """
    Stateful digital twin of the metropolitan rail network.
    """

    def __init__(self, topology: MetroTopology):
        self.topology = topology
        self.router = MetroRouteFinder(topology)
        self.trajectory_opt = TrainTrajectoryOptimizer()

        self.sim_time_sec = 8.0 * 3600 # Starts at 08:00 AM (Morning Peak)
        self.station_crowds = {name: float(np.random.uniform(0.15, 0.40)) for name in topology.station_names}
        self.station_queues = {name: int(self.station_crowds[name] * topology.stations[name]["capacity"]) for name in topology.station_names}
        
        # Initialize Trains across the 4 lines
        self.trains = self._initialize_trains()
        
        # Operational Event Flags
        self.active_disruptions = []
        self.total_passengers_served = 0
        self.total_energy_kwh = 0.0

    def _initialize_trains(self) -> List[Dict[str, Any]]:
        """
        Populates moving train fleets across Red, Blue, Green, and Yellow lines.
        """
        trains = []
        t_id = 1
        for line_name, stations in self.topology.lines.items():
            num_trains_on_line = 4 if line_name in ["Yellow", "Blue"] else 3
            interval = max(1, (len(stations) - 1) // max(1, num_trains_on_line - 1))
            
            for idx in range(num_trains_on_line):
                curr_station_idx = min(len(stations) - 1, idx * interval)
                st_name = stations[curr_station_idx]
                
                # Determine direction: 1 for forward, -1 for reverse
                if curr_station_idx >= len(stations) - 1:
                    train_dir = -1
                    next_st_name = stations[curr_station_idx - 1]
                else:
                    train_dir = 1
                    next_st_name = stations[curr_station_idx + 1]
                
                trains.append({
                    "id": f"DMRC-{line_name[:1]}-{t_id:02d}",
                    "line": line_name,
                    "direction": train_dir,
                    "current_station": st_name,
                    "next_station": next_st_name,
                    "status": "STOPPED", # STOPPED, RUNNING, HOLDING
                    "progress_pct": 0.0,
                    "passengers_onboard": int(np.random.uniform(400, 950)),
                    "dwell_remaining_sec": 25.0,
                    "run_time_remaining_sec": 120.0,
                    "speed_kmh": 0.0,
                    "speed_profile": "ECO_DRIVE",
                    "mechanical_health": "HEALTHY", # HEALTHY, WARNING, ANOMALY
                })
                t_id += 1
        return trains

    def trigger_event(self, event_type: str):
        """
        Injects real-time events into the metro simulation.
        Supported: 'STADIUM_MATCH', 'MORNING_RUSH', 'SIGNAL_FAULT', 'HEAVY_RAIN'
        """
        if event_type == "STADIUM_MATCH":
            if "Shivaji Stadium" in self.station_queues:
                self.station_queues["Shivaji Stadium"] += 3500
                self.station_crowds["Shivaji Stadium"] = min(1.0, self.station_queues["Shivaji Stadium"] / 4000.0)
            self.active_disruptions.append("🏏 Match Event Surge: 3,500+ passengers arriving at Shivaji Stadium / CP")
        elif event_type == "MORNING_RUSH":
            for st in ["Rajiv Chowk (CP)", "Kashmere Gate", "Hauz Khas"]:
                if st in self.station_queues:
                    self.station_queues[st] += 2500
                    self.station_crowds[st] = min(0.98, self.station_queues[st] / self.topology.stations[st]["capacity"])
            self.active_disruptions.append("🌅 Peak Morning Rush Hour: Heavy interchange boarding influx at Rajiv Chowk & Kashmere Gate")
        elif event_type == "SIGNAL_FAULT":
            for t in self.trains:
                if t["line"] == "Yellow":
                    t["dwell_remaining_sec"] += 60.0
                    t["status"] = "HOLDING"
            self.active_disruptions.append("⚠️ Yellow Line Interlock Delay at Central Secretariat: 60s holding penalty imposed")
        elif event_type == "HEAVY_RAIN":
            for st in self.station_queues:
                self.station_queues[st] = int(self.station_queues[st] * 1.30)
            self.active_disruptions.append("🌧️ Heavy Monsoon Rain in Delhi-NCR: Citywide +30% transit ridership surge")

    def step(self, dt_seconds: float = 15.0) -> Dict[str, Any]:
        """
        Advances the network state by dt_seconds.
        Handles passenger boarding, train traversal, and energy consumption.
        """
        self.sim_time_sec += dt_seconds
        hour = (self.sim_time_sec / 3600.0) % 24.0

        # 1. Update Station Passenger Queues & Dynamic Inflow
        for name in self.topology.station_names:
            cap = self.topology.stations[name]["capacity"]
            # Natural inflow rate based on time of day
            rate = 15.0 if (7.5 <= hour <= 9.5 or 17.0 <= hour <= 19.0) else 6.0
            new_passengers = int(np.random.poisson(rate * (dt_seconds / 15.0)))
            self.station_queues[name] = min(cap, self.station_queues[name] + new_passengers)
            self.station_crowds[name] = round(self.station_queues[name] / cap, 3)

        # 2. Update Moving Trains
        for t in self.trains:
            if t["status"] == "STOPPED" or t["status"] == "HOLDING":
                t["dwell_remaining_sec"] -= dt_seconds
                t["speed_kmh"] = 0.0

                # Passenger exchange
                st_name = t["current_station"]
                alight = min(t["passengers_onboard"], int(np.random.uniform(20, 60)))
                t["passengers_onboard"] -= alight
                self.total_passengers_served += alight

                space = 1500 - t["passengers_onboard"]
                board = min(space, self.station_queues[st_name], int(np.random.uniform(40, 90)))
                self.station_queues[st_name] -= board
                t["passengers_onboard"] += board

                if t["dwell_remaining_sec"] <= 0:
                    t["status"] = "RUNNING"
                    line_seq = self.topology.lines[t["line"]]
                    curr_idx = line_seq.index(t["current_station"])
                    
                    # Turn around at terminal stations
                    if curr_idx >= len(line_seq) - 1:
                        t["direction"] = -1
                    elif curr_idx <= 0:
                        t["direction"] = 1
                    
                    next_idx = curr_idx + t.get("direction", 1)
                    t["next_station"] = line_seq[next_idx]
                    
                    # Inter-station travel duration
                    edge_time = self.topology.graph[t["current_station"]][t["next_station"]]["base_time_min"]
                    t["run_time_remaining_sec"] = edge_time * 60.0
                    t["total_segment_sec"] = t["run_time_remaining_sec"]
                    t["progress_pct"] = 0.0

            elif t["status"] == "RUNNING":
                t["run_time_remaining_sec"] -= dt_seconds
                elapsed = t["total_segment_sec"] - max(0.0, t["run_time_remaining_sec"])
                t["progress_pct"] = min(100.0, round((elapsed / t["total_segment_sec"]) * 100.0, 1))

                # Speed profile approximation
                if t["progress_pct"] < 30.0:
                    t["speed_kmh"] = round(min(72.0, (t["progress_pct"] / 30.0) * 72.0), 1)
                elif t["progress_pct"] > 80.0:
                    t["speed_kmh"] = round(max(0.0, (1.0 - (t["progress_pct"] - 80.0) / 20.0) * 72.0), 1)
                else:
                    t["speed_kmh"] = 65.0 # Coasting / Cruising

                # Energy consumption step
                self.total_energy_kwh += (t["speed_kmh"] * 0.02 * (dt_seconds / 3600.0))

                if t["run_time_remaining_sec"] <= 0:
                    # Arrived at next station
                    t["current_station"] = t["next_station"]
                    t["status"] = "STOPPED"
                    t["progress_pct"] = 0.0
                    t["speed_kmh"] = 0.0
                    # RL-suggested dwell (normal 25s with crowd adjustment)
                    next_crowd = self.station_crowds[t["current_station"]]
                    t["dwell_remaining_sec"] = float(round(20.0 + next_crowd * 30.0, 1))

        sim_h = int(hour)
        sim_m = int((hour - sim_h) * 60)
        sim_s = int(self.sim_time_sec % 60)

        return {
            "clock": f"{sim_h:02d}:{sim_m:02d}:{sim_s:02d}",
            "trains": self.trains,
            "station_crowds": self.station_crowds,
            "station_queues": self.station_queues,
            "total_passengers_served": self.total_passengers_served,
            "total_energy_kwh": round(self.total_energy_kwh, 2),
            "active_disruptions": self.active_disruptions[-3:] # Recent 3 events
        }


if __name__ == "__main__":
    topo = MetroTopology()
    twin = MetroDigitalTwin(topo)
    for _ in range(5):
        st = twin.step(15.0)
        print(f"Clock: {st['clock']} | Served: {st['total_passengers_served']} | Energy: {st['total_energy_kwh']} kWh")
