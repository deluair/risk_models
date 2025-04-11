"""
Network analysis utility functions for financial risk analysis
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional, Union


def create_exposure_network(exposure_matrix: pd.DataFrame) -> nx.DiGraph:
    """Create a directed graph from an exposure matrix
    
    Args:
        exposure_matrix: DataFrame with exposures (rows = from, columns = to)
    
    Returns:
        Directed graph representing the exposure network
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node in exposure_matrix.index:
        G.add_node(node)
    
    # Add edges with weights
    for source in exposure_matrix.index:
        for target in exposure_matrix.columns:
            weight = exposure_matrix.loc[source, target]
            if weight > 0:
                G.add_edge(source, target, weight=weight)
    
    return G


def calculate_centrality_measures(G: nx.Graph) -> Dict[str, Dict[str, float]]:
    """Calculate various centrality measures for nodes in the network
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary mapping centrality measure names to dictionaries of node centralities
    """
    centrality_measures = {}
    
    # Basic centrality measures
    centrality_measures['degree'] = nx.degree_centrality(G)
    
    # For directed graphs, calculate in and out degree centrality
    if isinstance(G, nx.DiGraph):
        centrality_measures['in_degree'] = nx.in_degree_centrality(G)
        centrality_measures['out_degree'] = nx.out_degree_centrality(G)
    
    # Betweenness centrality
    centrality_measures['betweenness'] = nx.betweenness_centrality(G, weight='weight')
    
    # Eigenvector centrality
    try:
        centrality_measures['eigenvector'] = nx.eigenvector_centrality_numpy(G, weight='weight')
    except:
        # Fall back to power iteration method if numpy version fails
        try:
            centrality_measures['eigenvector'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        except:
            centrality_measures['eigenvector'] = {}
    
    # PageRank
    centrality_measures['pagerank'] = nx.pagerank(G, weight='weight')
    
    # Katz centrality (for directed graphs)
    if isinstance(G, nx.DiGraph):
        try:
            centrality_measures['katz'] = nx.katz_centrality(G, weight='weight')
        except:
            centrality_measures['katz'] = {}
    
    return centrality_measures


def calculate_systemic_importance(centrality_measures: Dict[str, Dict[str, float]], 
                                weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Calculate systemic importance scores by combining centrality measures
    
    Args:
        centrality_measures: Dictionary of centrality measures from calculate_centrality_measures
        weights: Dictionary mapping centrality measure names to weights (default: equal weights)
    
    Returns:
        Dictionary mapping nodes to systemic importance scores
    """
    if not centrality_measures:
        return {}
    
    # Get all nodes from the first centrality measure
    first_measure = next(iter(centrality_measures.values()))
    all_nodes = list(first_measure.keys())
    
    # Set default weights if not provided
    if weights is None:
        n_measures = len(centrality_measures)
        weights = {measure: 1.0/n_measures for measure in centrality_measures}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    # Calculate weighted average of centrality measures
    importance_scores = {}
    for node in all_nodes:
        score = 0.0
        for measure, measure_dict in centrality_measures.items():
            if measure in normalized_weights and node in measure_dict:
                score += normalized_weights[measure] * measure_dict[node]
        importance_scores[node] = score
    
    return importance_scores


def find_vulnerable_nodes(G: nx.Graph, importance_scores: Dict[str, float], 
                        threshold: float = 0.8) -> List[str]:
    """Find vulnerable nodes based on systemic importance and connectivity
    
    Args:
        G: NetworkX graph
        importance_scores: Dictionary mapping nodes to systemic importance scores
        threshold: Threshold for identifying vulnerable nodes (default: 0.8)
    
    Returns:
        List of vulnerable nodes
    """
    # Sort nodes by importance score
    sorted_nodes = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 20% of nodes by importance
    n_top = max(1, int(len(sorted_nodes) * 0.2))
    top_nodes = [node for node, score in sorted_nodes[:n_top]]
    
    # Calculate node vulnerability based on connections to important nodes
    vulnerability_scores = {}
    for node in G.nodes():
        if node in top_nodes:
            # Important nodes that are highly connected to other important nodes
            # are considered more vulnerable
            connections_to_top = 0
            total_weight = 0
            
            for neighbor in G.neighbors(node):
                weight = G[node][neighbor].get('weight', 1.0)
                if neighbor in top_nodes:
                    connections_to_top += 1
                    total_weight += weight
            
            # Vulnerability score based on connections and importance
            if len(top_nodes) > 1:
                connectivity_factor = connections_to_top / (len(top_nodes) - 1)
            else:
                connectivity_factor = 0
                
            vulnerability_scores[node] = importance_scores[node] * (0.5 + 0.5 * connectivity_factor)
        else:
            # Non-important nodes that are highly connected to important nodes
            # are also vulnerable
            connections_to_top = 0
            total_weight = 0
            
            for neighbor in G.neighbors(node):
                weight = G[node][neighbor].get('weight', 1.0)
                if neighbor in top_nodes:
                    connections_to_top += 1
                    total_weight += weight
            
            vulnerability_scores[node] = 0.5 * connections_to_top / len(top_nodes)
    
    # Normalize vulnerability scores
    max_score = max(vulnerability_scores.values()) if vulnerability_scores else 1.0
    vulnerability_scores = {k: v/max_score for k, v in vulnerability_scores.items()}
    
    # Return nodes with vulnerability score above threshold
    return [node for node, score in vulnerability_scores.items() if score >= threshold]


def calculate_contagion_impact(G: nx.DiGraph, initial_shock: Dict[str, float],
                             threshold: float = 0.1, max_rounds: int = 10) -> Dict[str, List[float]]:
    """Simulate financial contagion through the network
    
    Args:
        G: Directed NetworkX graph
        initial_shock: Dictionary mapping nodes to initial shock values (0-1)
        threshold: Threshold for node default (default: 0.1)
        max_rounds: Maximum number of contagion rounds (default: 10)
    
    Returns:
        Dictionary mapping nodes to lists of cumulative impact values over rounds
    """
    if not isinstance(G, nx.DiGraph):
        raise ValueError("G must be a directed graph")
    
    # Initialize impact tracking
    current_impact = {node: initial_shock.get(node, 0.0) for node in G.nodes()}
    cumulative_impact = {node: [current_impact[node]] for node in G.nodes()}
    defaulted = {node: current_impact[node] >= threshold for node in G.nodes()}
    
    # Run contagion simulation
    for _ in range(max_rounds):
        new_defaults = False
        new_impact = {node: current_impact[node] for node in G.nodes()}
        
        # Calculate new impacts based on defaulted counterparties
        for node in G.nodes():
            if not defaulted[node]:
                incoming_impact = 0.0
                
                # Calculate impact from incoming edges (exposures to this node)
                for pred in G.predecessors(node):
                    if defaulted[pred]:
                        edge_weight = G[pred][node].get('weight', 0.0)
                        # Scale impact by edge weight
                        incoming_impact += edge_weight * 0.01  # Convert to percentage impact
                
                # Update impact
                new_impact[node] += incoming_impact
                
                # Check if node defaults
                if new_impact[node] >= threshold and not defaulted[node]:
                    defaulted[node] = True
                    new_defaults = True
        
        # Update current impact
        current_impact = new_impact
        
        # Store cumulative impact for this round
        for node in G.nodes():
            cumulative_impact[node].append(current_impact[node])
        
        # Stop if no new defaults occurred
        if not new_defaults:
            break
    
    return cumulative_impact


def calculate_network_risk_metrics(G: nx.Graph) -> Dict[str, float]:
    """Calculate network-level risk metrics
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary of network risk metrics
    """
    metrics = {}
    
    # Basic network metrics
    metrics['density'] = nx.density(G)
    metrics['transitivity'] = nx.transitivity(G)
    
    # For directed graphs, calculate reciprocity
    if isinstance(G, nx.DiGraph):
        metrics['reciprocity'] = nx.reciprocity(G)
    
    # Create undirected version for some metrics
    if isinstance(G, nx.DiGraph):
        G_undir = G.to_undirected()
    else:
        G_undir = G
    
    # Check connectivity
    metrics['is_connected'] = nx.is_connected(G_undir)
    
    # Number of connected components
    metrics['n_components'] = nx.number_connected_components(G_undir)
    
    # Size of largest component
    largest_cc = max(nx.connected_components(G_undir), key=len)
    metrics['largest_component_size'] = len(largest_cc)
    metrics['largest_component_fraction'] = len(largest_cc) / G.number_of_nodes()
    
    # Average shortest path length in largest component
    largest_cc_graph = G_undir.subgraph(largest_cc)
    try:
        metrics['avg_shortest_path'] = nx.average_shortest_path_length(largest_cc_graph, weight='weight')
    except:
        metrics['avg_shortest_path'] = float('nan')
    
    # Diameter of largest component
    try:
        metrics['diameter'] = nx.diameter(largest_cc_graph, weight='weight')
    except:
        metrics['diameter'] = float('nan')
    
    # Assortativity
    try:
        metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
    except:
        metrics['assortativity'] = float('nan')
    
    # Calculate centralization (how central is the most central node compared to others)
    if len(G) > 1:
        degree_centrality = nx.degree_centrality(G)
        max_centrality = max(degree_centrality.values())
        centrality_sum = sum(max_centrality - c for c in degree_centrality.values())
        # Normalize by maximum possible centralization (star network)
        max_possible = (len(G) - 1) * (1 - 1/(len(G) - 1))
        metrics['degree_centralization'] = centrality_sum / max_possible if max_possible > 0 else 0
    else:
        metrics['degree_centralization'] = 0
    
    return metrics


def detect_communities(G: nx.Graph, method: str = 'louvain') -> Dict[str, int]:
    """Detect communities in the network
    
    Args:
        G: NetworkX graph
        method: Community detection method ('louvain', 'greedy_modularity', 'label_propagation')
    
    Returns:
        Dictionary mapping nodes to community IDs
    """
    if method == 'louvain':
        try:
            from community import best_partition
            return best_partition(G)
        except ImportError:
            # Fall back to greedy modularity if python-louvain is not installed
            method = 'greedy_modularity'
    
    if method == 'greedy_modularity':
        communities = nx.community.greedy_modularity_communities(G)
        community_dict = {}
        for i, community in enumerate(communities):
            for node in community:
                community_dict[node] = i
        return community_dict
    
    elif method == 'label_propagation':
        communities = nx.community.label_propagation_communities(G)
        community_dict = {}
        for i, community in enumerate(communities):
            for node in community:
                community_dict[node] = i
        return community_dict
    
    else:
        raise ValueError(f"Unknown community detection method: {method}")


def calculate_modularity(G: nx.Graph, communities: Dict[str, int]) -> float:
    """Calculate modularity of the community structure
    
    Args:
        G: NetworkX graph
        communities: Dictionary mapping nodes to community IDs
    
    Returns:
        Modularity score
    """
    # Convert communities dict to list of sets format
    community_sets = {}
    for node, comm_id in communities.items():
        if comm_id not in community_sets:
            community_sets[comm_id] = set()
        community_sets[comm_id].add(node)
    
    community_list = list(community_sets.values())
    
    return nx.community.modularity(G, community_list) 