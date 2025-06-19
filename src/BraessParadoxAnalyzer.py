import numpy as np
import networkx as nx 
from scipy.optimize import minimize 
import pandas as pd 

def setup_graph(network_config):
    G = nx.DiGraph()
    G.add_nodes_from(network_config['nodes'])
    for edge_data in network_config['edges']:
        u = edge_data['u']
        v = edge_data['v']
        attrs = {k: v for k, v in edge_data.items() if k not in ['u', 'v']}
        G.add_edge(u, v, **attrs)
    return G

def get_all_paths(G, source, target):
    return list(nx.all_simple_paths(G, source, target))

def calculate_edge_cost(G, edge, flow):
    edge_data = G.edges[edge]
    return edge_data['a'] * flow + edge_data['b']

def calculate_path_cost(G, path, edge_flows):
    cost = 0
    for i in range(len(path) - 1):
        edge = (path[i], path[i+1])
        if edge not in G.edges(): 
            return float('inf') 
        cost += calculate_edge_cost(G, edge, edge_flows.get(edge, 0))
    return cost

def get_highest_used_route_cost(G, path_flows, paths, edge_flows):
    used_path_costs = []
    for i, flow in enumerate(path_flows):
        if flow > 1e-6: 
            cost = calculate_path_cost(G, paths[i], edge_flows)
            if cost != float('inf'):
                used_path_costs.append(cost)
    
    if not used_path_costs:
        return 0.0 
    return max(used_path_costs) 

def calculate_total_system_cost(G, edge_flows):
    total_cost = 0
    for edge, flow in edge_flows.items():
        if flow > 1e-10: 
            total_cost += flow * calculate_edge_cost(G, edge, flow)
    return total_cost

def solve_user_equilibrium_msa(G, demand, source_node, target_node, braess_link=None, exclude_braess_link=False, verbose=False):
    local_G = G.copy()
    braess_data = None
    if exclude_braess_link and braess_link and local_G.has_edge(*braess_link):
        braess_data = local_G.edges[braess_link].copy()
        local_G.remove_edge(*braess_link)
    
    paths = get_all_paths(local_G, source_node, target_node)
    n_paths = len(paths)
    
    if n_paths == 0:
        if verbose: print(f"Warning: No paths found for UE.")
        if exclude_braess_link and braess_data is not None:
            local_G.add_edge(*braess_link, **braess_data)
        return {e: 0.0 for e in local_G.edges()}, 0.0, 0.0, np.array([])

    path_flows = np.array([demand / n_paths] * n_paths) if n_paths > 0 else np.array([])
    
    def _update_edge_flows(current_path_flows):
        flows = {edge: 0.0 for edge in local_G.edges()}
        for i, path in enumerate(paths):
            if i >= len(current_path_flows): continue 
            for j in range(len(path) - 1):
                edge = (path[j], path[j+1])
                if edge in local_G.edges(): 
                    flows[edge] += current_path_flows[i]
        return flows
    
    max_iterations = 10000 
    tolerance = 1e-6 

    for iteration in range(max_iterations):
        current_edge_flows = _update_edge_flows(path_flows)
        current_path_costs = np.array([calculate_path_cost(local_G, path, current_edge_flows) for path in paths])
        
        finite_costs = current_path_costs[current_path_costs != float('inf')]
        if len(finite_costs) == 0:
             if verbose: print(f"Warning: No finite cost paths at iter {iteration}. Breaking MSA.")
             break
        
        min_cost = np.min(finite_costs)
        min_cost_indices = np.where(current_path_costs == min_cost)[0] 
        
        auxiliary_flows = np.zeros(n_paths)
        if len(min_cost_indices) > 0:
            flow_per_shortest_path = demand / len(min_cost_indices)
            for idx in min_cost_indices:
                auxiliary_flows[idx] = flow_per_shortest_path
        
        step_size = 1.0 / (iteration + 2) 
        
        old_path_flows = path_flows.copy()
        path_flows = (1 - step_size) * path_flows + step_size * auxiliary_flows
        
        path_flows[path_flows < 0] = 0
        current_sum_path_flows = np.sum(path_flows)
        if current_sum_path_flows > 1e-9: 
            path_flows = path_flows * (demand / current_sum_path_flows)
        else: 
            path_flows = np.array([demand / n_paths] * n_paths) 
        
        flow_change = np.linalg.norm(path_flows - old_path_flows) / (np.linalg.norm(old_path_flows) + 1e-10)
        
        if flow_change < tolerance and iteration > 10: 
            if verbose: print(f"MSA converged at iteration {iteration}")
            break
    else:
        print(f"MSA did NOT converge within {max_iterations} iterations for UE (exclude_braess_link={exclude_braess_link}). Final flow change: {flow_change:.6f}")
    
    final_edge_flows = _update_edge_flows(path_flows)
    
    if exclude_braess_link and braess_data is not None:
        local_G.add_edge(*braess_link, **braess_data) 
    
    highest_used_route_cost = get_highest_used_route_cost(local_G, path_flows, paths, final_edge_flows)
    total_system_cost = calculate_total_system_cost(local_G, final_edge_flows)

    return final_edge_flows, total_system_cost, highest_used_route_cost, path_flows

def solve_system_optimal(G, demand, source_node, target_node, braess_link=None, exclude_braess_link=False, verbose=False):
    local_G = G.copy()
    braess_data = None
    if exclude_braess_link and braess_link and local_G.has_edge(*braess_link):
        braess_data = local_G.edges[braess_link].copy()
        local_G.remove_edge(*braess_link)
    
    paths = get_all_paths(local_G, source_node, target_node)
    n_paths = len(paths)
    
    if n_paths == 0:
        if verbose: print(f"Warning: No paths found for SO.")
        if exclude_braess_link and braess_data is not None:
            local_G.add_edge(*braess_link, **braess_data)
        return {e: 0.0 for e in local_G.edges()}, 0.0, 0.0, np.array([]) 

    def system_cost_objective(path_flows_arr):
        edge_flows = {edge: 0.0 for edge in local_G.edges()}
        for i, path in enumerate(paths):
            if i >= len(path_flows_arr): continue
            for j in range(len(path) - 1):
                edge = (path[j], path[j+1])
                if edge in local_G.edges():
                    edge_flows[edge] += path_flows_arr[i]
        
        total_cost = 0
        for edge, flow in edge_flows.items():
            if flow > 1e-10: 
                total_cost += flow * calculate_edge_cost(local_G, edge, flow)
        
        return total_cost
    
    def flow_conservation_constraint(path_flows_arr):
        return np.sum(path_flows_arr) - demand
    
    initial_guess = np.array([demand / n_paths] * n_paths)
    initial_guess[initial_guess < 0] = 0

    bounds = [(0, None) for _ in range(n_paths)]
    constraints = [{'type': 'eq', 'fun': flow_conservation_constraint}]
    
    result = minimize(system_cost_objective, initial_guess, 
                      method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-10, 'disp': False, 'maxiter': 10000}) 
    
    path_flows = result.x
    path_flows[path_flows < 0] = 0 
    
    edge_flows = {edge: 0.0 for edge in local_G.edges()}
    for i, path in enumerate(paths):
        if i >= len(path_flows): continue
        for j in range(len(path) - 1):
            edge = (path[j], path[j+1])
            if edge in local_G.edges():
                edge_flows[edge] += path_flows[i]
    
    total_cost = result.fun 
    highest_used_route_cost = get_highest_used_route_cost(local_G, path_flows, paths, edge_flows)
    
    if exclude_braess_link and braess_data is not None:
        local_G.add_edge(*braess_link, **braess_data)
    
    return edge_flows, total_cost, highest_used_route_cost, path_flows

def analyze_network(network_config):
    print(f"\n\n\n\n")
    print(f"Analyzing Network with Demand: {network_config['demand']}")
    
    # Initialize graph for analysis
    initial_G = setup_graph(network_config)

    # User Equilibrium (with Braess Link)
    print("Calculating User Equilibrium (With Braess Link) ...")
    ue_flows_braess, ue_total_system_cost_braess, ue_highest_used_cost_braess, ue_path_flows_braess = \
        solve_user_equilibrium_msa(initial_G.copy(), network_config['demand'], 
                                   network_config['source_node'], network_config['target_node'], 
                                   braess_link=network_config.get('braess_link'), exclude_braess_link=False)

    # System Optimal (With Braess Link)
    print("Calculating System Optimal (With Braess Link) ...")
    so_flows_braess, so_total_system_cost_braess, so_highest_used_cost_braess, so_path_flows_braess = \
        solve_system_optimal(initial_G.copy(), network_config['demand'], 
                             network_config['source_node'], network_config['target_node'], 
                             braess_link=network_config.get('braess_link'), exclude_braess_link=False)

    # User Equilibrium (Without Braess Link)
    ue_flows_no_braess = {}
    ue_total_system_cost_no_braess = 0.0
    ue_highest_used_cost_no_braess = 0.0

    if network_config.get('braess_link'):
        print(f"Calculating User Equilibrium (Without Braess Link: {network_config['braess_link']}) ...")
        ue_flows_no_braess, ue_total_system_cost_no_braess, ue_highest_used_cost_no_braess, ue_path_flows_no_braess = \
            solve_user_equilibrium_msa(initial_G.copy(), network_config['demand'], 
                                       network_config['source_node'], network_config['target_node'], 
                                       braess_link=network_config['braess_link'], exclude_braess_link=True)
    else:
        print("\nNo Braess link specified for 'Without Braess Link' analysis.")
        all_current_edges = list(initial_G.edges())
        ue_flows_no_braess = {edge: 0.0 for edge in all_current_edges}

    # Create results table with all edges
    results_data = []
    all_edges = list(initial_G.edges())
    
    for i, edge in enumerate(all_edges):
        edge_display = f"{edge[0]}->{edge[1]}"
        
        ue_flow_braess = ue_flows_braess.get(edge, 0)
        so_flow_braess = so_flows_braess.get(edge, 0)
        ue_flow_no_braess = ue_flows_no_braess.get(edge, 0) 
        
        results_data.append({
            'Edge': edge_display, 
            'Equilibrium': f"{ue_flow_braess:.2f}",
            'System Optimal': f"{so_flow_braess:.2f}",
            'Braess': f"{ue_flow_no_braess:.2f}", 
        })
    
    results_data.append({
        'Edge': 'Highest Used Route Cost', 
        'Equilibrium': f"{ue_highest_used_cost_braess:.1f}",
        'System Optimal': f"{so_highest_used_cost_braess:.1f}", 
        'Braess': f"{ue_highest_used_cost_no_braess:.1f}",
    })
    results_data.append({
        'Edge': 'System Cost',
        'Equilibrium': f"{ue_total_system_cost_braess:.1f}",
        'System Optimal': f"{so_total_system_cost_braess:.1f}",
        'Braess': f"{ue_total_system_cost_no_braess:.1f}",
    })

    df = pd.DataFrame(results_data)
    
    print("\n" + "="*100)
    print("                                   BRAESS PARADOX ANALYSIS RESULTS")
    print("="*100)
    print(f"Network demand: {network_config['demand']}")
    print("Flows and Costs:")
    print(df.to_string(index=False)) 
    
    if network_config.get('braess_link'):
        if ue_highest_used_cost_braess - ue_highest_used_cost_no_braess > 0.1:
            print("CLASSIC BRAESS PARADOX CONFIRMED!!!") 
            print(f"   Adding the Braess link increased the highest used route cost (UE travel time) by: "
                  f"{ue_highest_used_cost_braess - ue_highest_used_cost_no_braess:.2f}")
        else:
            print("CLASSIC BRAESS PARADOX NOT OBSERVED!!!")
    else:
        print("No Braess link was specified, so the Classic Braess Paradox check was skipped.")

    return df

def get_input():
    print("Enter network configuration:")
    print("Format:")
    print("Line 1: Demand")
    print("Line 2: Nodes")
    print("Line 3+: Adjacency Matrix (-1 for no link)")
    print("After matrix: cost parameters for each existing link")
    print("Final: source, target, and optional braess link")
    print()
    
    # Read demand
    demand = float(input("Demand: "))
    
    # Read nodes
    nodes_str = input("Nodes: ")
    nodes = nodes_str.split()
    num_nodes = len(nodes)
    
    print(f"\nEnter {num_nodes}x{num_nodes} adjacency matrix:")
    print("Use -1 for no link and 1 for existing link")
    
    # Read adjacency matrix
    adj_matrix = []
    for i in range(num_nodes):
        row_str = input(f"Row {i+1}: ")
        row = list(map(int, row_str.split()))
        adj_matrix.append(row)
    
    # Collect edges
    existing_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] != -1:
                existing_edges.append((nodes[i], nodes[j]))
    
    print(f"\nEnter linear cost functions for {len(existing_edges)} edges:")
    print("Format for each edge: a b (where cost = a*flow + b)")
    
    edges = []
    for u, v in existing_edges:
        params_str = input(f"Edge {u}->{v} (a b): ")
        a, b = map(float, params_str.split())
        edges.append({'u': u, 'v': v, 'a': a, 'b': b})
    
    # Read source and target
    source_node = input(f"\nSource node: ")
    target_node = input(f"Target node: ")
    
    # Read Braess Link
    braess_input = input("Braess link (u v) or press Enter for none: ")
    braess_link = None
    if braess_input.strip():
        parts = braess_input.split()
        if len(parts) == 2:
            braess_link = (parts[0], parts[1])
    
    network_config = {
        'demand': demand,
        'nodes': nodes,
        'edges': edges,
        'source_node': source_node,
        'target_node': target_node,
        'braess_link': braess_link
    }
    
    return network_config

if __name__ == "__main__":
    user_network = get_input()
    analyze_network(user_network)