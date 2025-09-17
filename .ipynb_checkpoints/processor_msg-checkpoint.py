import os
import re
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

# Function to read a file line by line
def read_file(path):
    """Read file and load lines"""
    sents = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sents.append(str(line.strip()))
    return sents

# Prepare line-level adjacency
def prepare_line_adj(line_lenth, weighted_graph=False):
    window_size = 6
    windows = []
    idx = range(line_lenth)

    if line_lenth <= window_size:
        windows.append(idx)
    else:
        for j in range(line_lenth - window_size + 1):
            window = idx[j: j + window_size]
            windows.append(window)

    word_pair_count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p_id = window[p]
                word_q_id = window[q]
                word_pair_key = (word_p_id, word_q_id)

                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
                # two orders
                word_pair_key = (word_q_id, word_p_id)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.

    row = []
    col = []
    weight = []
    for key in word_pair_count:
        p = key[0]
        q = key[1]
        row.append(p)
        col.append(q)
        weight.append(word_pair_count[key] if weighted_graph else 1.)

    adj = sp.csr_matrix((weight, (row, col)), shape=(line_lenth, line_lenth))
    edge_index = np.array(adj.nonzero())

    return edge_index

# Prepare code to 2D structure
def prepare_code2d(code_list, line_lenth, max_seq_len=100, to_lowercase=False, weighted_graph=False):
    window_size = 2
    code2d = []
    all_word_edge_index = []

    for c in code_list:
        windows = []

        c = re.sub('\\s+', ' ', c)

        if to_lowercase:
            c = c.lower()

        token_list = c.strip().split()
        total_tokens = len(token_list)

        if total_tokens > max_seq_len:
            token_list = token_list[:max_seq_len]
            total_tokens = max_seq_len

        if total_tokens < max_seq_len:
            token_list = token_list + ['<pad>'] * (max_seq_len - total_tokens)

        code2d.append(token_list)

        idx = range(0, total_tokens)

        if total_tokens <= window_size:
            windows.append(idx)
        else:
            for j in range(total_tokens - window_size + 1):
                window = idx[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p_id = window[p]
                    word_q_id = window[q]

                    word_pair_key = (word_p_id, word_q_id)

                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(p)
            col.append(q)
            weight.append(word_pair_count[key] if weighted_graph else 1.)

        adj = sp.csr_matrix((weight, (row, col)), shape=(max_seq_len, max_seq_len))
        edge_index = np.array(adj.nonzero())
        all_word_edge_index.append(edge_index)

    all_line_edge_index = prepare_line_adj(line_lenth, weighted_graph)

    return code2d, all_word_edge_index, all_line_edge_index

# Process .msg file
def process_msg_file(msg_file_path, line_lenth, max_seq_len=100, to_lowercase=False, weighted_graph=False):
    code_list = read_file(msg_file_path)
    num_lines = len(code_list)
    code2d, all_word_edge_index, all_line_edge_index = prepare_code2d(code_list, line_lenth, max_seq_len, to_lowercase, weighted_graph)
    return code2d, all_word_edge_index, all_line_edge_index, num_lines

# Visualize graph using edge_index
def visualize_edge_index(edge_index, title="Graph Visualization"):
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        source, target = edge_index[0, i], edge_index[1, i]
        G.add_edge(source, target)

    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
    plt.title(title)
    plt.show()

# Main execution
msg_file_path = "/home/database/zychuan1/TDReminder/top_repos_10000/new_python/train_fixme_python.msg"  # Replace with your file path
line_lenth = 80  # Adjust based on your file

# Process the file and visualize
code2d, all_word_edge_index, all_line_edge_index, num_lines = process_msg_file(msg_file_path, line_lenth)

# Visualize the first graph
if all_word_edge_index:
    first_edge_index = all_word_edge_index[0]
    visualize_edge_index(first_edge_index, title="Visualization of the First Word Graph")

# Print results
print("code2d:", code2d[:5])
print("all_word_edge_index:", all_word_edge_index[:1])
print("all_line_edge_index:", all_line_edge_index[:1])
print("Number of lines processed:", num_lines)
