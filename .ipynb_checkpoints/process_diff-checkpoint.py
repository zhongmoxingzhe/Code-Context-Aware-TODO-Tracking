import networkx as nx
import os
import matplotlib.pyplot as plt
from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('./Model/bert-base-uncased')#记得改成codebert编码，现在是bert编码

def bert_encode(input_lst):
    # BERT encoding
    encoded_input = tokenizer(input_lst, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return encoded_input    

def parse_diff(diff_text):
    diff_text = diff_text.replace('<nl>', '\n')  
    lines = [line for line in diff_text.split('\n') if line.strip()]  # Filter out empty lines
    added_lines = []
    removed_lines = []

    for line in lines:
        line = line.lstrip()  
        if line.startswith('+'):
            added_lines.append(line[1:].strip())  
        elif line.startswith('-'):
            removed_lines.append(line[1:].strip())  

    return added_lines, removed_lines

def build_graph(added_lines, removed_lines):
    G = nx.DiGraph()

    # Encode all lines
    removed_encodings = [bert_encode(line) for line in removed_lines]
    added_encodings = [bert_encode(line) for line in added_lines]

    for i, (line, encoding) in enumerate(zip(removed_lines, removed_encodings)):
        G.add_node(f"removed_{i}", label=line, encoding=encoding)
    

    for i, (line, encoding) in enumerate(zip(added_lines, added_encodings)):
        G.add_node(f"added_{i}", label=line, encoding=encoding)
    

    for i in range(min(len(added_lines), len(removed_lines))):
        G.add_edge(f"removed_{i}", f"added_{i}")

    for i in range(min(len(added_lines), len(removed_lines))):
        removed_line = removed_lines[i]
        added_line = added_lines[i]

        if removed_line != added_line:
            G.add_node(f"replaced_{i}", label=f"Replaced: {removed_line} -> {added_line}", encoding=bert_encode(f"Replaced: {removed_line} -> {added_line}"))
            G.add_edge(f"removed_{i}", f"replaced_{i}")
            G.add_edge(f"added_{i}", f"replaced_{i}")

    return G

def process_file(file_path):
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return []

    graphs = []  
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()  
            if not line:
                continue  
            try:
                added_lines, removed_lines = parse_diff(line)
                G = build_graph(added_lines, removed_lines)
                graphs.append(G)  
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")

    print(f"Total graphs created: {len(graphs)}")
    return graphs

def store_graphs_as_array(graphs):
    """
    Store all graphs in an array-like structure.
    """
    graph_array = []
    for G in graphs:
        graph_data = {
            "nodes": list(G.nodes(data=True)),  # List of nodes with attributes
            "edges": list(G.edges(data=True)),  # List of edges with attributes
        }
        graph_array.append(graph_data)
    return graph_array

def visualize_graph(G):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)  
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color="skyblue", font_size=10, font_color="black")
    plt.title("Graph Visualization")
    plt.show()

# Main script
file_path = "/home/database/zychuan1/TDReminder/top_repos_10000/new_python/test_fixme_python.diff"
graphs = process_file(file_path)

# Store graphs as an array
graph_array = store_graphs_as_array(graphs)

# Example: Access the first graph's data
if len(graph_array) > 0:
    first_graph_nodes = graph_array[0]["nodes"]
    first_graph_edges = graph_array[0]["edges"]


# Visualize a graph (e.g., the 5th graph if it exists)
if len(graphs) >= 5:
    visualize_graph(graphs[4])
else:
    print("Not enough graphs to visualize the fifth one.")
