import re
import networkx as nx
import matplotlib.pyplot as plt

def parse_code_to_cedg(code):
    """
    Parse Solidity-like code to create a CEDG with detailed node and edge attributes.
    Node attributes: [Category, Type, Name].
    Edge attributes: [V_s, V_e, type, order].
    """
    # Initialize directed graph for CEDG
    G = nx.DiGraph()

    # Define unique nodes for each type: I, V, F
    nodes = {'I': [], 'V': [], 'F': []}
    access_edges = set()  # Track Access edges to avoid redundancy
    edge_order = 1  # Track edge execution order

    # Identify the function name for the primary invocation node
    function_match = re.search(r'function\s+(\w+)', code)
    func_name = function_match.group(1) if function_match else "Fallback"
    if func_name not in nodes['I']:
        G.add_node(func_name, Category="Function", Type="I", Name=func_name)
        nodes['I'].append(func_name)
    last_node = func_name  # Track last node for sequential edge creation

    # Parse statements and add edges as per Table I
    statements = re.split(r';|\{|\}', code)
    for statement in statements:
        statement = statement.strip()

        # Control-flow: handle complex conditions using logical operators
        if re.match(r'if\s*\(.*?\)', statement):
            condition = re.search(r'if\s*\((.*?)\)', statement).group(1)
            condition_vars = re.split(r'\s*(?:&&|\|\|)\s*', condition)

            for cond in condition_vars:
                vars_in_cond = re.split(r'\s*(?:>=|==|>|<|!=)\s*', cond)
                for var in vars_in_cond:
                    if var not in nodes['V']:
                        G.add_node(var, Category="Variable", Type="V", Name=var)
                        nodes['V'].append(var)

            # Connect condition variables in sequence with IF edges
            for i in range(len(vars_in_cond) - 1):
                G.add_edge(vars_in_cond[i], vars_in_cond[i+1], type="IF", order=edge_order)
                edge_order += 1
            last_node = vars_in_cond[-1]

        # Invocation: handle require, transfer, etc.
        elif re.match(r'require\s*\(.*?\)', statement):
            if "require" not in nodes['I']:
                G.add_node("require", Category="Function", Type="I", Name="require")
                nodes["I"].append("require")
            G.add_edge(last_node, "require", type="TC", order=edge_order)
            edge_order += 1
            last_node = "require"

        elif re.match(r'revert\b', statement):
            if "revert" not in nodes['I']:
                G.add_node("revert", Category="Function", Type="I", Name="revert")
                nodes["I"].append("revert")
            G.add_edge(last_node, "revert", type="TC", order=edge_order)
            edge_order += 1
            last_node = "revert"

        elif re.search(r'\b(transfer|send|call|delegatecall)\b', statement):
            invocation = re.search(r'\b(transfer|send|call|delegatecall)\b', statement).group(1)
            if invocation not in nodes['I']:
                G.add_node(invocation, Category="Function", Type="I", Name=invocation)
                nodes["I"].append(invocation)
            G.add_edge(last_node, invocation, type="NS", order=edge_order)
            edge_order += 1
            last_node = invocation

        # Data-flow: handle assignments (AS) and access (AC) for variables
        assignment_match = re.match(r'(\b[_a-zA-Z][_a-zA-Z0-9]*)\s*=\s*(.*)', statement)
        if assignment_match:
            var, expr = assignment_match.groups()
            if var not in nodes['V']:
                G.add_node(var, Category="Variable", Type="V", Name=var)
                nodes['V'].append(var)
            G.add_edge(last_node, var, type="AS", order=edge_order)
            edge_order += 1
            last_node = var

        # Data-flow: accessing complex variables
        complex_vars = re.findall(r'([_a-zA-Z][_a-zA-Z0-9]*\[[^\]]+\])', statement)
        for var in complex_vars:
            if var not in nodes['V']:
                G.add_node(var, Category="Variable", Type="V", Name=var)
                nodes['V'].append(var)
            if (last_node, var) not in access_edges:
                G.add_edge(last_node, var, type="AC", order=edge_order)
                access_edges.add((last_node, var))
                edge_order += 1

    # Add Fallback node as per paper example (F)
    if "Fallback" not in nodes['F']:
        G.add_node("Fallback", Category="Function", Type="F", Name="Fallback")
        nodes['F'].append("Fallback")
    G.add_edge("Fallback", func_name, type="FB", order=edge_order)

    return G

def visualize_cedg(graph):
    """
    Visualize CEDG with node colors based on types (I, V, F) and labeled edges following edge types.
    """
    node_color_map = {"I": "lightblue", "V": "lightgreen", "F": "orange"}
    pos = nx.spring_layout(graph, seed=42)
    node_colors = [node_color_map.get(graph.nodes[node]['Type'], "grey") for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color=node_colors, edgecolors="black")
    edge_labels = nx.get_edge_attributes(graph, 'type')
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle='-|>', arrowsize=20)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="black")
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="black")
    plt.title("Contract Elements Dependency Graph (CEDG)")
    plt.axis("off")
    plt.show()

def export_graph_structure(graph):
    """
    Export graph structure with node and edge attributes for GNN input.
    """
    # Collect node information
    nodes_info = [{"id": node, **graph.nodes[node]} for node in graph.nodes()]

    # Collect edge information
    edges_info = [{"V_s": u, "V_e": v, "type": d['type'], "order": d['order']} for u, v, d in graph.edges(data=True)]

    return {"nodes": nodes_info, "edges": edges_info}
