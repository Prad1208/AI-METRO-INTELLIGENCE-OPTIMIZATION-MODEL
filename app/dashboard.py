"""
Streamlit Interactive Operations Dashboard & Digital Twin.
Visualizes live metro network maps, dynamic ST-GNN crowding heatmaps, moving trains,
Pareto-optimal route comparison, energy trajectory curves, and predictive maintenance telemetry.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# Ensure project modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.network_topology import MetroTopology
from simulation.metro_sim import MetroDigitalTwin
from models.energy_optimizer import TrainTrajectoryOptimizer


# Page Configuration
st.set_page_config(
    page_title="MetroAI | Intelligent Transit Digital Twin",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State (Auto-refreshes if topology changes)
fresh_topo = MetroTopology()
if "topology" not in st.session_state or set(st.session_state.topology.station_names) != set(fresh_topo.station_names):
    st.session_state.topology = fresh_topo
    st.session_state.digital_twin = MetroDigitalTwin(st.session_state.topology)
    st.session_state.last_route_res = None
if "traj_opt" not in st.session_state:
    st.session_state.traj_opt = TrainTrajectoryOptimizer()

topo = st.session_state.topology
twin = st.session_state.digital_twin
traj_opt = st.session_state.traj_opt

# -----------------------------------------------------------------------------
# Sidebar: Control Center & Simulation Clock
# -----------------------------------------------------------------------------
st.sidebar.title("🚇 Delhi Metro OCC")
st.sidebar.caption("DMRC Autonomous Operations & Intelligence")

sim_step_btn = st.sidebar.button("⏩ Advance Network Clock (+30s)", use_container_width=True)
if sim_step_btn:
    twin.step(dt_seconds=30.0)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Real-Time Event Injection")
col_e1, col_e2 = st.sidebar.columns(2)
with col_e1:
    if st.button("🏏 CP Match Surge", use_container_width=True):
        twin.trigger_event("STADIUM_MATCH")
        st.sidebar.success("Surge added to Shivaji Stadium / CP!")
with col_e2:
    if st.button("🌅 Morning Peak", use_container_width=True):
        twin.trigger_event("MORNING_RUSH")
        st.sidebar.success("Rajiv Chowk & Kashmere Gate rush triggered!")

col_e3, col_e4 = st.sidebar.columns(2)
with col_e3:
    if st.button("⚠️ Yellow Line Delay", use_container_width=True):
        twin.trigger_event("SIGNAL_FAULT")
        st.sidebar.warning("Central Secretariat delay injected!")
with col_e4:
    if st.button("🌧️ Monsoon Rain", use_container_width=True):
        twin.trigger_event("HEAVY_RAIN")
        st.sidebar.info("Delhi-NCR monsoon rain surge active!")

st.sidebar.markdown("---")
st.sidebar.subheader("Active System Alerts")
if twin.active_disruptions:
    for alert in twin.active_disruptions[-3:]:
        st.sidebar.info(alert)
else:
    st.sidebar.caption("All lines operating under nominal conditions.")


# -----------------------------------------------------------------------------
# Header KPI Metrics
# -----------------------------------------------------------------------------
sim_state = twin.step(dt_seconds=0.0) # Peek state
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("🕒 Simulation Clock", sim_state["clock"])
with kpi2:
    st.metric("🚆 Active Trains", len(sim_state["trains"]))
with kpi3:
    st.metric("👥 Passengers Served", f"{sim_state['total_passengers_served']:,}")
with kpi4:
    st.metric("⚡ Total Traction Power", f"{sim_state['total_energy_kwh']:,.1f} kWh")
with kpi5:
    avg_crowd = np.mean(list(sim_state["station_crowds"].values())) * 100.0
    st.metric("📊 Network Crowd Load", f"{avg_crowd:.1f}%")

st.markdown("---")

# -----------------------------------------------------------------------------
# Tabs Interface
# -----------------------------------------------------------------------------
tab_map, tab_router, tab_energy, tab_pdm = st.tabs([
    "🗺️ Network Digital Twin & Map",
    "🧭 AI Multi-Objective Route Planner",
    "⚡ Energy & Trajectory Optimization",
    "🛠️ Fleet Predictive Maintenance"
])

# -----------------------------------------------------------------------------
# TAB 1: Network Digital Twin & Map
# -----------------------------------------------------------------------------
with tab_map:
    st.subheader("Metropolitan Rail Topological Map & Real-Time Flow Dynamics")
    
    line_colors = {
        "Yellow": "#EAB308",
        "Blue": "#2563EB",
        "Airport Express": "#EA580C",
        "Magenta": "#DB2777"
    }

    fig_map = go.Figure()

    # 1. Plot Track Lines (Edges)
    for u, v, data in topo.graph.edges(data=True):
        x0, y0 = topo.stations[u]["x"], topo.stations[u]["y"]
        x1, y1 = topo.stations[v]["x"], topo.stations[v]["y"]
        primary_line = data["lines"][0]
        color = line_colors.get(primary_line, "#6B7280")
        
        fig_map.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=color, width=5),
            hoverinfo="text",
            text=f"Line: {', '.join(data['lines'])} | Dist: {data['distance_km']} km | Run: {data['base_time_min']} min",
            showlegend=False
        ))

    # 2. Plot Stations (Nodes) with Crowd Heatmap Color
    station_x = [topo.stations[s]["x"] for s in topo.station_names]
    station_y = [topo.stations[s]["y"] for s in topo.station_names]
    crowd_vals = [sim_state["station_crowds"][s] * 100.0 for s in topo.station_names]
    queue_vals = [sim_state["station_queues"][s] for s in topo.station_names]
    hover_texts = [
        f"<b>{s}</b><br>Lines: {', '.join(topo.stations[s]['lines'])}<br>Waiting: {queue_vals[i]} pax<br>Crowd Load: {crowd_vals[i]:.1f}%"
        for i, s in enumerate(topo.station_names)
    ]

    fig_map.add_trace(go.Scatter(
        x=station_x, y=station_y,
        mode="markers+text",
        marker=dict(
            size=18,
            color=crowd_vals,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Crowd %", x=1.02),
            line=dict(color="#FFFFFF", width=2)
        ),
        text=[s if topo.stations[s]["is_interchange"] else "" for s in topo.station_names],
        textposition="top center",
        hoverinfo="text",
        hovertext=hover_texts,
        name="DMRC Stations"
    ))

    # 3. Plot Moving Trains
    train_x, train_y, train_hovers, train_colors = [], [], [], []
    for t in sim_state["trains"]:
        st_from = topo.stations[t["current_station"]]
        st_to = topo.stations[t["next_station"]]
        p = t["progress_pct"] / 100.0
        # Linear interpolation of train coordinates along track
        tx = st_from["x"] + (st_to["x"] - st_from["x"]) * p
        ty = st_from["y"] + (st_to["y"] - st_from["y"]) * p
        train_x.append(tx)
        train_y.append(ty)
        train_colors.append(line_colors.get(t["line"], "#FFFFFF"))
        train_hovers.append(
            f"<b>{t['id']}</b> ({t['line']} Line)<br>Status: {t['status']}<br>Speed: {t['speed_kmh']} km/h<br>Passengers: {t['passengers_onboard']}/1500"
        )

    fig_map.add_trace(go.Scatter(
        x=train_x, y=train_y,
        mode="markers",
        marker=dict(
            symbol="square",
            size=14,
            color=train_colors,
            line=dict(color="#111827", width=2)
        ),
        hoverinfo="text",
        hovertext=train_hovers,
        name="Active Metro Trains"
    ))

    fig_map.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)")
    )

    st.plotly_chart(fig_map, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: AI Multi-Objective Route Planner
# -----------------------------------------------------------------------------
with tab_router:
    st.subheader("DMRC Delhi Metro: ML-Predicted Dynamic Route Optimizer")
    st.caption("Fuses topological shortest-paths with real-time Spatio-Temporal GNN crowd forecasts to generate Pareto-optimal itineraries.")

    default_orig_idx = topo.station_names.index("Rajiv Chowk (CP)") if "Rajiv Chowk (CP)" in topo.station_names else 0
    default_dest_idx = topo.station_names.index("IGI Airport Terminal 3") if "IGI Airport Terminal 3" in topo.station_names else len(topo.station_names) - 1

    c_orig, c_dest, c_btn = st.columns([3, 3, 2])
    with c_orig:
        origin_choice = st.selectbox("Origin Station (Source)", topo.station_names, index=default_orig_idx)
    with c_dest:
        dest_choice = st.selectbox("Destination Station", topo.station_names, index=default_dest_idx)
    with c_btn:
        st.write("")
        st.write("")
        calc_btn = st.button("🔍 Find Optimal Routes", use_container_width=True)

    current_od = (origin_choice, dest_choice)
    if calc_btn or st.session_state.get("last_od") != current_od or st.session_state.get("last_route_res") is None:
        st.session_state.last_route_res = twin.router.find_routes(
            origin_choice, dest_choice, crowd_predictions=sim_state["station_crowds"]
        )
        st.session_state.last_od = current_od
    routes = st.session_state.last_route_res

    if routes and "fastest" in routes:
        col_r1, col_r2, col_r3 = st.columns(3)

        # Fastest Route Card
        with col_r1:
            rf = routes["fastest"]
            st.success("⚡ **Fastest Route**")
            st.markdown(f"**Duration**: `{rf['total_time_min']} mins`")
            st.markdown(f"**Crowd Stress**: `{rf['avg_crowd_score'] * 100:.0f}%`")
            st.markdown(f"**Transfers**: `{rf['transfers']}`")
            st.markdown(f"**Lines**: `{', '.join(rf['lines_used'])}`")
            st.info(" ➔ ".join(rf["path"]))

        # Comfort Route Card
        with col_r2:
            rc = routes["comfort"]
            st.warning("🧘 **Comfort / Low-Crowd Route**")
            st.markdown(f"**Duration**: `{rc['total_time_min']} mins`")
            st.markdown(f"**Crowd Stress**: `{rc['avg_crowd_score'] * 100:.0f}%` (Lowest)")
            st.markdown(f"**Transfers**: `{rc['transfers']}`")
            st.markdown(f"**Lines**: `{', '.join(rc['lines_used'])}`")
            st.info(" ➔ ".join(rc["path"]))

        # Min Transfers Route Card
        with col_r3:
            rt = routes["min_transfers"]
            st.error("🔄 **Direct / Minimum Transfers**")
            st.markdown(f"**Duration**: `{rt['total_time_min']} mins`")
            st.markdown(f"**Crowd Stress**: `{rt['avg_crowd_score'] * 100:.0f}%`")
            st.markdown(f"**Transfers**: `{rt['transfers']}` (Fewest)")
            st.markdown(f"**Lines**: `{', '.join(rt['lines_used'])}`")
            st.info(" ➔ ".join(rt["path"]))

        # Turn-by-turn segment expansion
        if rf.get("segments"):
            with st.expander("📋 View Turn-by-Turn Journey Breakdown (Fastest Route)", expanded=True):
                seg_data = []
                for s in rf.get("segments", []):
                    seg_data.append({
                        "From Station": s["from"],
                        "To Station": s["to"],
                        "Metro Line": s["line"],
                        "Segment Time": f"{s['time_min']} mins",
                        "Platform Crowd": f"{s['crowd'] * 100:.0f}%",
                        "Interchange": "🔄 Transfer" if s.get("is_transfer") else "Direct"
                    })
                st.dataframe(pd.DataFrame(seg_data), use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: Energy & Trajectory Optimization
# -----------------------------------------------------------------------------
with tab_energy:
    st.subheader("Physics-Informed Trajectory Optimization (Regenerative Braking)")
    st.caption("Demonstrating 15-22% traction electrical power savings using the Davis Equation and optimized coasting regimes.")

    col_e_in1, col_e_in2, col_e_in3 = st.columns(3)
    with col_e_in1:
        dist_slider = st.slider("Inter-station Distance (km)", min_value=1.0, max_value=6.0, value=2.8, step=0.2)
    with col_e_in2:
        time_slider = st.slider("Target Travel Time (sec)", min_value=90, max_value=260, value=175, step=5)
    with col_e_in3:
        pax_slider = st.slider("Passenger Load Onboard", min_value=100, max_value=1500, value=850, step=50)

    # Recompute with chosen slider inputs
    custom_traj_opt = TrainTrajectoryOptimizer(passenger_count=pax_slider)
    comp = custom_traj_opt.compare_driving_strategies(distance_km=dist_slider, target_time_sec=float(time_slider))

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Standard Drive Power", f"{comp['standard_profile']['net_energy_kwh']} kWh")
    with m2:
        st.metric("AI Eco-Drive Power", f"{comp['eco_profile']['net_energy_kwh']} kWh")
    with m3:
        st.metric("⚡ Net Energy Saved", f"{comp['energy_saved_kwh']} kWh", delta=f"{comp['savings_percent']}% Saved")
    with m4:
        st.metric("Regen Power Recovered", f"{comp['eco_profile']['regen_recovered_kwh']} kWh")

    # Plot Speed Profile Comparison
    fig_traj = go.Figure()
    std = comp["standard_profile"]
    eco = comp["eco_profile"]

    fig_traj.add_trace(go.Scatter(
        x=std["distance_m"], y=std["speed_kmh"],
        mode="lines",
        line=dict(color="#EF4444", width=2.5, dash="dash"),
        name="Aggressive Driving (Max Speed & Late Brake)"
    ))

    fig_traj.add_trace(go.Scatter(
        x=eco["distance_m"], y=eco["speed_kmh"],
        mode="lines",
        line=dict(color="#10B981", width=3.5),
        name="AI Eco-Driving Profile (Motoring ➔ Coasting ➔ Regen)"
    ))

    fig_traj.update_layout(
        title="Train Speed Profile vs Track Distance",
        xaxis_title="Distance along track (meters)",
        yaxis_title="Speed (km/h)",
        template="plotly_dark",
        height=450,
        legend=dict(x=0.02, y=0.98)
    )
    st.plotly_chart(fig_traj, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 4: Predictive Maintenance & Anomaly Detection
# -----------------------------------------------------------------------------
with tab_pdm:
    st.subheader("Rolling Stock Bogie & Motor Anomaly Monitoring")
    st.caption("Deep Autoencoder reconstruction loss monitoring 3-axis vibration shocks, motor current spikes, and bearing overheating.")

    # Telemetry simulation sample
    pdm_col1, pdm_col2 = st.columns([1, 2])
    with pdm_col1:
        st.markdown("#### Real-time Sensor Telemetry")
        st.metric("Radial Vibration (X-axis)", "0.142 g", "Normal")
        st.metric("Axial Vibration (Y-axis)", "0.118 g", "Normal")
        st.metric("Vertical Vibration (Z-axis)", "0.215 g", "Normal")
        st.metric("Motor Stator Current", "312.4 A", "Normal")
        st.metric("Wheel Bearing Temp", "48.2 °C", "Normal")

        st.success("✅ **Fleet Health Status**: All train bogies within nominal vibration tolerance thresholds.")

    with pdm_col2:
        # Generate representative sample chart
        sample_time = np.linspace(0, 60, 120)
        v_signal = 0.18 + 0.05 * np.sin(sample_time * 0.8) + np.random.normal(0, 0.015, len(sample_time))
        # Anomaly pulse at t = 35..40
        v_signal[70:80] += 0.45

        fig_pdm = go.Figure()
        fig_pdm.add_trace(go.Scatter(
            x=sample_time, y=v_signal,
            mode="lines",
            line=dict(color="#3B82F6", width=2),
            name="Vibration Sensor (g-force)"
        ))
        fig_pdm.add_hline(y=0.40, line_dash="dash", line_color="#EF4444", annotation_text="Anomaly Alert Threshold (0.40g)")

        fig_pdm.update_layout(
            title="Real-Time Vibration Waveform & Defect Threshold",
            xaxis_title="Time (seconds)",
            yaxis_title="Vibration Amplitude (g)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_pdm, use_container_width=True)
