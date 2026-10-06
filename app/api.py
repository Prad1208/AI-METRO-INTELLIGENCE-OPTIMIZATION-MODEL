"""
FastAPI REST API Serving Layer for Metro AI Systems.
Exposes endpoints for crowd prediction, multi-objective route planning,
dynamic dispatching, and train energy optimization.
"""
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

from data.network_topology import MetroTopology
from models.router import MetroRouteFinder
from models.energy_optimizer import TrainTrajectoryOptimizer
from simulation.metro_sim import MetroDigitalTwin

app = FastAPI(
    title="AI Metro Intelligence & Route Optimization API",
    description="Production-grade AI endpoints for transit network routing, GNN crowding prediction, and RL dispatching.",
    version="1.0.0"
)

# Global in-memory state
topology = MetroTopology()
router = MetroRouteFinder(topology)
twin = MetroDigitalTwin(topology)
traj_opt = TrainTrajectoryOptimizer()


class RouteRequest(BaseModel):
    origin: str = Field(..., example="Rajiv Chowk (CP)")
    destination: str = Field(..., example="IGI Airport Terminal 3")


class RouteResponse(BaseModel):
    origin: str
    destination: str
    routes: Dict[str, Any]


class EventRequest(BaseModel):
    event_type: str = Field(..., example="JLN Stadium")


class EnergyRequest(BaseModel):
    distance_km: float = Field(2.5, example=2.5)
    target_time_sec: float = Field(170.0, example=170.0)
    coasting_ratio: float = Field(0.35, example=0.35)


@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "stations_loaded": topology.num_stations, "lines": list(topology.lines.keys())}


@app.get("/topology")
def get_topology():
    """
    Returns complete network topology including stations, coordinates, and track segments.
    """
    return {
        "stations": topology.stations,
        "lines": topology.lines,
        "edges_count": topology.graph.number_of_edges()
    }


@app.post("/route/optimize", response_model=RouteResponse)
def optimize_route(req: RouteRequest):
    """
    Computes Pareto-optimal transit routes (Fastest, Comfort, Direct).
    """
    if req.origin not in topology.stations:
        raise HTTPException(status_code=400, detail=f"Origin station '{req.origin}' not found.")
    if req.destination not in topology.stations:
        raise HTTPException(status_code=400, detail=f"Destination station '{req.destination}' not found.")

    routes = router.find_routes(req.origin, req.destination, crowd_predictions=twin.station_crowds)
    return RouteResponse(origin=req.origin, destination=req.destination, routes=routes)


@app.get("/simulation/state")
def get_simulation_state():
    """
    Retrieves live digital twin operational telemetry (trains, queues, crowds, clock).
    """
    return twin.step(dt_seconds=15.0)


@app.post("/simulation/inject-event")
def inject_event(req: EventRequest):
    """
    Injects disruptions: 'STADIUM_MATCH', 'MORNING_RUSH', 'SIGNAL_FAULT', 'HEAVY_RAIN'
    """
    valid_events = ["STADIUM_MATCH", "MORNING_RUSH", "SIGNAL_FAULT", "HEAVY_RAIN"]
    if req.event_type not in valid_events:
        raise HTTPException(status_code=400, detail=f"Invalid event type. Must be one of {valid_events}")
    twin.trigger_event(req.event_type)
    return {"status": "SUCCESS", "event_injected": req.event_type}


@app.post("/energy/optimize-trajectory")
def optimize_trajectory(req: EnergyRequest):
    """
    Computes physics-based energy optimal speed profile and comparison against standard aggressive driving.
    """
    comparison = traj_opt.compare_driving_strategies(distance_km=req.distance_km, target_time_sec=req.target_time_sec)
    return comparison
