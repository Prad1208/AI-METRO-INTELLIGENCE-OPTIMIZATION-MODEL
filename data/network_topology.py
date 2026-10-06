"""
Metropolitan Rail Network Topology and Graph Representation for Delhi Metro (DMRC).
Defines stations, lines, track segments, interchange hubs, and spectral graph operators.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
import networkx as nx
import json


class MetroTopology:
    """
    Constructs and manages the Delhi Metro transit network graph G = (V, E).
    """

    def __init__(self):
        self.stations = self._define_stations()
        self.lines = self._define_lines()
        self.station_names = list(self.stations.keys())
        self.num_stations = len(self.station_names)
        self.station_to_idx = {name: i for i, name in enumerate(self.station_names)}
        self.idx_to_station = {i: name for i, name in enumerate(self.station_names)}
        
        # Build NetworkX Multi-line Graph
        self.graph = self._build_graph()
        
        # Build Adjacency Matrix
        self.adjacency_matrix = self._build_adjacency_matrix()

    def _define_stations(self) -> Dict[str, Dict[str, Any]]:
        """
        Defines 23 iconic Delhi Metro stations across NCR (Delhi, Gurugram, Noida)
        with geometric coordinates, line memberships, and platform capacities.
        """
        return {
            # --- Yellow Line Corridor (North to South Gurugram) ---
            "Samaypur Badli": {"x": 50.0, "y": 94.0, "lines": ["Yellow"], "capacity": 4000, "is_interchange": False},
            "Vishwavidyalaya (DU)": {"x": 50.0, "y": 84.0, "lines": ["Yellow"], "capacity": 5500, "is_interchange": False},
            "Kashmere Gate": {"x": 50.0, "y": 74.0, "lines": ["Yellow"], "capacity": 8500, "is_interchange": True},
            "Chandni Chowk": {"x": 50.0, "y": 66.0, "lines": ["Yellow"], "capacity": 6500, "is_interchange": False},
            "New Delhi": {"x": 50.0, "y": 58.0, "lines": ["Yellow", "Airport Express"], "capacity": 8000, "is_interchange": True},
            "Rajiv Chowk (CP)": {"x": 50.0, "y": 50.0, "lines": ["Yellow", "Blue"], "capacity": 10000, "is_interchange": True},
            "Central Secretariat": {"x": 50.0, "y": 40.0, "lines": ["Yellow"], "capacity": 6000, "is_interchange": False},
            "Dilli Haat - INA": {"x": 50.0, "y": 30.0, "lines": ["Yellow"], "capacity": 5800, "is_interchange": False},
            "Hauz Khas": {"x": 50.0, "y": 20.0, "lines": ["Yellow", "Magenta"], "capacity": 7500, "is_interchange": True},
            "Millennium City Centre (Gurugram)": {"x": 50.0, "y": 6.0, "lines": ["Yellow"], "capacity": 6200, "is_interchange": False},

            # --- Blue Line Corridor (Dwarka to Noida) ---
            "Dwarka Sector 21": {"x": 12.0, "y": 26.0, "lines": ["Blue", "Airport Express"], "capacity": 5000, "is_interchange": True},
            "Janakpuri West": {"x": 24.0, "y": 50.0, "lines": ["Blue", "Magenta"], "capacity": 6500, "is_interchange": True},
            "Kirti Nagar": {"x": 33.0, "y": 50.0, "lines": ["Blue"], "capacity": 4500, "is_interchange": False},
            "Karol Bagh": {"x": 41.0, "y": 50.0, "lines": ["Blue"], "capacity": 5500, "is_interchange": False},
            # Rajiv Chowk is shared with Yellow
            "Mandi House": {"x": 59.0, "y": 50.0, "lines": ["Blue"], "capacity": 6000, "is_interchange": False},
            "Yamuna Bank": {"x": 68.0, "y": 50.0, "lines": ["Blue"], "capacity": 4800, "is_interchange": False},
            "Mayur Vihar-1": {"x": 76.0, "y": 42.0, "lines": ["Blue"], "capacity": 5200, "is_interchange": False},
            "Botanical Garden (Noida)": {"x": 86.0, "y": 34.0, "lines": ["Blue", "Magenta"], "capacity": 7000, "is_interchange": True},
            "Noida Electronic City": {"x": 95.0, "y": 30.0, "lines": ["Blue"], "capacity": 4500, "is_interchange": False},

            # --- Airport Express / Orange Line ---
            "Shivaji Stadium": {"x": 42.0, "y": 48.0, "lines": ["Airport Express"], "capacity": 4000, "is_interchange": False},
            "Dhaula Kuan": {"x": 32.0, "y": 38.0, "lines": ["Airport Express"], "capacity": 4500, "is_interchange": False},
            "IGI Airport Terminal 3": {"x": 22.0, "y": 26.0, "lines": ["Airport Express"], "capacity": 6000, "is_interchange": False},

            # --- Magenta Line Corridor ---
            "IGI Airport Terminal 1": {"x": 34.0, "y": 24.0, "lines": ["Magenta"], "capacity": 4800, "is_interchange": False},
        }

    def _define_lines(self) -> Dict[str, List[str]]:
        """
        Sequential station list for Delhi Metro lines.
        """
        return {
            "Yellow": [
                "Samaypur Badli", "Vishwavidyalaya (DU)", "Kashmere Gate", "Chandni Chowk",
                "New Delhi", "Rajiv Chowk (CP)", "Central Secretariat", "Dilli Haat - INA",
                "Hauz Khas", "Millennium City Centre (Gurugram)"
            ],
            "Blue": [
                "Dwarka Sector 21", "Janakpuri West", "Kirti Nagar", "Karol Bagh",
                "Rajiv Chowk (CP)", "Mandi House", "Yamuna Bank", "Mayur Vihar-1",
                "Botanical Garden (Noida)", "Noida Electronic City"
            ],
            "Airport Express": [
                "New Delhi", "Shivaji Stadium", "Dhaula Kuan",
                "IGI Airport Terminal 3", "Dwarka Sector 21"
            ],
            "Magenta": [
                "Janakpuri West", "IGI Airport Terminal 1", "Hauz Khas", "Botanical Garden (Noida)"
            ]
        }

    def _build_graph(self) -> nx.Graph:
        """
        Builds a NetworkX weighted undirected graph with physical track attributes.
        """
        G = nx.Graph()
        
        # Add Nodes
        for name, data in self.stations.items():
            G.add_node(name, **data)
            
        # Add Track Segments (Edges) for each line
        for line_name, seq in self.lines.items():
            for i in range(len(seq) - 1):
                u, v = seq[i], seq[i + 1]
                pos_u = np.array([self.stations[u]["x"], self.stations[u]["y"]])
                pos_v = np.array([self.stations[v]["x"], self.stations[v]["y"]])
                dist_km = float(np.linalg.norm(pos_u - pos_v) * 0.28) # Scale factor to km
                
                # Airport express is high-speed (80 km/h avg), others 50 km/h avg
                avg_speed = 80.0 if line_name == "Airport Express" else 52.0
                base_time_min = float(round((dist_km / avg_speed) * 60.0 + 0.7, 2))
                
                if G.has_edge(u, v):
                    existing_lines = G[u][v].get("lines", [])
                    if line_name not in existing_lines:
                        existing_lines.append(line_name)
                    G[u][v]["lines"] = existing_lines
                else:
                    G.add_edge(
                        u, v,
                        distance_km=dist_km,
                        base_time_min=base_time_min,
                        lines=[line_name],
                        speed_limit_kmh=120.0 if line_name == "Airport Express" else 80.0
                    )
                    
        return G

    def _build_adjacency_matrix(self) -> np.ndarray:
        """
        Constructs the binary and distance-weighted adjacency matrix A.
        """
        A = np.zeros((self.num_stations, self.num_stations), dtype=np.float32)
        for u, v, data in self.graph.edges(data=True):
            i = self.station_to_idx[u]
            j = self.station_to_idx[v]
            dist = data.get("distance_km", 1.0)
            weight = np.exp(- (dist ** 2) / 12.0)
            A[i, j] = weight
            A[j, i] = weight
        return A

    def get_normalized_laplacian(self) -> np.ndarray:
        """
        Calculates the symmetric normalized graph Laplacian:
        L = I - D^(-1/2) * A * D^(-1/2)
        """
        A = self.adjacency_matrix + np.eye(self.num_stations, dtype=np.float32)
        d = np.sum(A, axis=1)
        d_inv_sqrt = np.power(d, -0.5, where=d > 0)
        d_inv_sqrt[d == 0] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        L = np.eye(self.num_stations, dtype=np.float32) - A_norm
        return L

    def get_chebyshev_polynomials(self, k: int = 2) -> List[np.ndarray]:
        """
        Computes Chebyshev polynomial approximations up to order k.
        """
        L = self.get_normalized_laplacian()
        eigenvals = np.linalg.eigvalsh(L)
        lambda_max = float(np.max(eigenvals))
        
        L_tilde = (2.0 / lambda_max) * L - np.eye(self.num_stations, dtype=np.float32)
        
        cheb_polys = [np.eye(self.num_stations, dtype=np.float32)]
        if k >= 1:
            cheb_polys.append(L_tilde)
        for i in range(2, k + 1):
            T_i = 2.0 * L_tilde @ cheb_polys[-1] - cheb_polys[-2]
            cheb_polys.append(T_i.astype(np.float32))
            
        return cheb_polys

    def export_topology_json(self) -> str:
        data = {
            "stations": self.stations,
            "lines": self.lines,
            "edges": [
                {
                    "from": u,
                    "to": v,
                    "distance_km": data["distance_km"],
                    "base_time_min": data["base_time_min"],
                    "lines": data["lines"]
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }
        return json.dumps(data, indent=2)


if __name__ == "__main__":
    metro = MetroTopology()
    print(f"✅ Successfully constructed Delhi Metro (DMRC) with {metro.num_stations} stations.")
    print(f"Stations: {metro.station_names}")
