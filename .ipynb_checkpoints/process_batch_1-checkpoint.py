import os
import re
import sys
import numpy as np
import scipy.sparse as sp
import networkx as nx
from transformers import BertTokenizer
import torch
from transformers import RobertaTokenizer
from GMN.utils_GMN import build_model
from GMN.configure import *
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from MLPClassifier import MLPClassifier  # 导入 MLPClassifier
from utils_GMN import *
from configure import *
from difflib import SequenceMatcher

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

mlp_classifier = MLPClassifier(input_dim=512).to(device)  # MLP 分类头

from TDReminder.utils import *
from TDReminder.BERT_model import codeModel, Config, load_codeBERT
from TDReminder.loss_func import Focal_loss, DiceLoss

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('./Model/codebert-base')
tokenizer_msg = BertTokenizer.from_pretrained('./Model/bert-base-uncased')


# GMN_config = get_default_config()

class Data_processor(object):
    # Initialize the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('./Model/codebert-base')
    tokenizer_msg = BertTokenizer.from_pretrained('./Model/bert-base-uncased')

    def __init__(self, set_type, batch_size, dir_path, keyword):
        self.set_type = set_type
        self.batch_size = batch_size
        self.dir_path = dir_path
        self.keyword = keyword
        print("Loading codeBERT model...")
        self.bert_model, self.tokenizer = load_codeBERT()
        print('codeBERT loaded')

        print('Loading %s dataset...' % (set_type + keyword))
        path_diff, path_msg, flag_path = load_path(self.dir_path, set_type, keyword)
        flag_file = read_file(flag_path)
        flag_file = [int(i) for i in flag_file]
        label_type = flag_file
        self.labels = np.asarray(label_type)
        print("Begin encoding...")
        self.processed_dict = self.process_files(path_diff, path_msg, self.labels)
        self.processed_dataloader = self.convert_to_tensors(self.processed_dict)
        self.processed_tensor_dataloader = self.make_load(self.processed_dataloader)
        print("%s dataset loaded.")

    def pad_tensors(self, tensor_list, max_len0, max_len1):
        """
        将张量列表填充到相同的长度（支持两个维度）。
        :param tensor_list: 张量列表
        :param max_len1: 目标长度（第一个维度）
        :param max_len2: 目标长度（第二个维度）
        :return: 填充后的张量列表
        """
        padded_tensors = []
        for tensor in tensor_list:
            # 获取当前张量的形状
            current_len0, current_len1 = tensor.size(0), tensor.size(1)

            # 计算需要填充的长度
            pad_len0 = max_len0 - current_len0
            pad_len1 = max_len1 - current_len1

            # 在第二个和第三个维度上填充
            if pad_len1 > 0 or pad_len0 > 0:
                # F.pad 的填充顺序是从最后一个维度开始向前填充
                # 格式：(padding_left, padding_right, padding_top, padding_bottom, ...)
                padded_tensor = F.pad(tensor, (0, pad_len0, 0, pad_len1), "constant", 0)
            else:
                padded_tensor = tensor

            padded_tensors.append(padded_tensor)

        return padded_tensors

    # 迫于内存的限制，填充时截取到128，多的裁掉，少的填充为0
    def pad_tensor_node_to_256(self, tensor, target_dim1=256, target_dim2=256):
        """
        将二维张量填充到 [256, 256]
        :param tensor: 输入张量，形状为 [seq_len, feature_dim]
        :return: 填充后的张量，形状为 [512, 512]
        """
        current_dim1, current_dim2 = tensor.size(0), tensor.size(1)

        # 计算需要填充的长度
        pad_dim1 = max(target_dim1 - current_dim1, 0)
        pad_dim2 = max(target_dim2 - current_dim2, 0)

        # 填充顺序：(最后一个维度左, 右, 倒数第二个维度左, 右)
        padded_tensor = F.pad(tensor, (0, pad_dim2, 0, pad_dim1), "constant", 0)

        # 截断到目标长度（二维索引）
        padded_tensor = padded_tensor[:target_dim1, :target_dim2]

        return padded_tensor

    def pad_edge_index(self, edge_index, target_edges=128):
        """
        将 edge_index 填充到 (2, target_edges) 形状
        :param edge_index: 形状 [2, num_edges]
        :param target_edges: 目标边数
        :return: 填充后的 edge_index
        """
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros(2, target_edges, dtype=torch.long)  # 返回全 0

        num_edges = edge_index.size(1)  # 获取当前边数
        if num_edges >= target_edges:
            return edge_index[:, :target_edges]  # 截断
        else:
            pad_edges = target_edges - num_edges
            padding = torch.zeros(2, pad_edges, dtype=edge_index.dtype)  # (2, pad_edges)
            return torch.cat([edge_index, padding], dim=1)  # 在列维度拼接

    def pad_edge_attr(self, edge_attr, target_edges=256):
        """
        将边属性填充到固定边数。
        :param edge_attr: 原始边属性，形状 [num_edges] 或 [num_edges, feature_dim]
        :param target_edges: 目标边数
        :return: 填充后的边属性，形状 [target_edges] 或 [target_edges, feature_dim]
        """
        current_edges = edge_attr.size(0)

        if current_edges >= target_edges:
            return edge_attr[:target_edges]  # 直接截断
        else:
            pad_edges = target_edges - current_edges

            # 根据维度创建对应形状的填充张量
            if edge_attr.dim() == 1:
                padding = torch.zeros(pad_edges, dtype=edge_attr.dtype, device=edge_attr.device)
            else:
                padding = torch.zeros((pad_edges, edge_attr.size(1)),  # 保持特征维度一致
                                      dtype=edge_attr.dtype,
                                      device=edge_attr.device)

            # 统一执行拼接操作
            padded_edge_attr = torch.cat([edge_attr, padding], dim=0)
            return padded_edge_attr

    def make_load(self, processed_dataloader, target_nodes=128, target_edges=128):
        """
        将所有张量填充到固定尺寸，并创建 DataLoader。
        :param processed_dataloader: 包含原始数据的字典
        :param target_nodes: 目标节点数（填充到该数量）
        :param target_edges: 目标边数（填充到该数量）
        :return: DataLoader
        """

        def collate_fn(batch):
            node_features_list = []
            edge_index_list = []
            edge_attr_list = []
            labels_list = []
            graph_idx = []

            n_total_nodes = 0
            n_total_edges = 0

            # 先处理所有的 diff 图
            for i, data in enumerate(batch):
                diff_node_features, diff_edge_index, diff_edge_attr, msg_node_features, msg_edge_index, msg_edge_attr, label = data

                # 处理 diff 的节点特征
                node_features_list.append(diff_node_features)

                # 处理 diff 的边索引（添加偏移量）
                diff_edge_index_offset = diff_edge_index + n_total_nodes
                edge_index_list.append(diff_edge_index_offset)

                # 处理 diff 的边特征
                edge_attr_list.append(diff_edge_attr)

                # 标记 diff 图的节点属于哪个图
                graph_idx.extend([i] * diff_node_features.shape[0])

                # 更新偏移量
                n_total_nodes += diff_node_features.shape[0]
                n_total_edges += diff_edge_index.shape[1]

            # 再处理所有的 msg 图
            for i, data in enumerate(batch):
                diff_node_features, diff_edge_index, diff_edge_attr, msg_node_features, msg_edge_index, msg_edge_attr, label = data

                # 处理 msg 的节点特征
                node_features_list.append(msg_node_features)

                # 处理 msg 的边索引（添加偏移量）
                msg_edge_index_offset = msg_edge_index + n_total_nodes
                edge_index_list.append(msg_edge_index_offset)

                # 处理 msg 的边特征
                edge_attr_list.append(msg_edge_attr)

                # 标记 msg 图的节点属于哪个图
                graph_idx.extend([i + len(batch)] * msg_node_features.shape[0])

                # 更新偏移量
                n_total_nodes += msg_node_features.shape[0]
                n_total_edges += msg_edge_index.shape[1]

                # 标签
                labels_list.append(label)

            # 拼接所有图的数据
            node_features = torch.cat(node_features_list, dim=0)
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_attr = torch.cat(edge_attr_list, dim=0)
            labels = torch.stack(labels_list, dim=0)
            graph_idx = torch.tensor(graph_idx, dtype=torch.long)

            # 返回打包后的数据
            return {
                'node_features': node_features,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'labels': labels,
                'graph_idx': graph_idx,
                'n_graphs': len(batch) * 2  # 每个输入图包含 diff 和 msg 两个图
            }

        # 初始化列表，存储填充后的张量
        diff_node_features_list = []
        diff_edge_index_list = []
        diff_edge_attr_list = []
        msg_node_features_list = []
        msg_edge_index_list = []
        msg_edge_attr_list = []
        labels_list = []  # 新增 labels 列表
        i = 0
        for key in processed_dataloader:
            data = processed_dataloader[key]

            # 处理 diff_graph
            # 1. 填充节点特征
            diff_node_features = data['diff_graph']['node_features']
            diff_node_features_padded = self.pad_tensor_node_to_256(diff_node_features)
            diff_node_features_list.append(diff_node_features_padded)

            # 2. 填充边索引和属性
            diff_edge_index = data['diff_graph']['edge_index']
            diff_edge_index_padded = self.pad_edge_index(diff_edge_index, 128)  # 适配mlp
            diff_edge_index_list.append(diff_edge_index_padded)

            diff_edge_attr = data['diff_graph']['edge_attr']
            diff_edge_attr_padded = self.pad_edge_attr(diff_edge_attr, 128)
            diff_edge_attr_padded = diff_edge_attr_padded.unsqueeze(-1)  # shape: [128, 1]
            ones = torch.ones(diff_edge_attr_padded.size(0), 3)  # [128, 3]
            diff_edge_attr_padded = torch.cat([diff_edge_attr_padded, ones], dim=-1)  # shape: [128, 4]
            diff_edge_attr_list.append(diff_edge_attr_padded)

            # 处理 msg_graph
            # 1. 填充节点特征
            msg_node_features = data['msg_graph']['node_features']
            msg_node_features_padded = self.pad_tensor_node_to_256(msg_node_features)
            msg_node_features_list.append(msg_node_features_padded)

            # 2. 填充边索引和属性
            msg_edge_index = data['msg_graph']['word_edge_index']
            msg_edge_index_padded = self.pad_edge_index(msg_edge_index, 128)
            msg_edge_index_list.append(msg_edge_index_padded)

            msg_edge_attr = data['msg_graph']['edge_weight']
            msg_edge_attr_padded = self.pad_edge_attr(msg_edge_attr, 128)  # 为内存做出妥协让步，mlp
            msg_edge_attr_padded = msg_edge_attr_padded.unsqueeze(-1)  # shape: [128, 1]
            ones = torch.ones(msg_edge_attr_padded.size(0), 3)  # [128, 3]
            msg_edge_attr_padded = torch.cat([msg_edge_attr_padded, ones], dim=-1)  # shape: [128, 4]
            msg_edge_attr_list.append(msg_edge_attr_padded)

            # 处理 labels（假设 data['label'] 存在）
            label = torch.tensor(data['label'], dtype=torch.float)  # 或者 dtype=torch.long
            labels_list.append(label)
            i += 1

        # 堆叠所有张量
        diff_node_features = torch.stack(diff_node_features_list)
        diff_edge_index = torch.stack(diff_edge_index_list)
        diff_edge_attr = torch.stack(diff_edge_attr_list)
        msg_node_features = torch.stack(msg_node_features_list)
        msg_edge_index = torch.stack(msg_edge_index_list)
        msg_edge_attr = torch.stack(msg_edge_attr_list)
        labels = torch.stack(labels_list)  # 堆叠 labels

        # 创建 TensorDataset
        tensor_dataset = TensorDataset(
            diff_node_features, diff_edge_index, diff_edge_attr,
            msg_node_features, msg_edge_index, msg_edge_attr,
            labels  # 添加 labels
        )

        dataloader = DataLoader(tensor_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        # 创建 DataLoader
        return dataloader

    def process_files(self, diff_file_path, msg_file_path, labels):
        if not os.path.isfile(diff_file_path) or not os.path.isfile(msg_file_path):
            print("One or both files not found.")
            return {}

        diff_graphs = []
        msg_graphs = []

        with open(diff_file_path, 'r', encoding='utf-8') as diff_file, open(msg_file_path, 'r',
                                                                            encoding='utf-8') as msg_file:
            for diff_line, msg_line in zip(diff_file, msg_file):
                diff_line = diff_line.strip()
                msg_line = msg_line.strip()

                if not diff_line or not msg_line:
                    continue

                try:
                    added_lines, removed_lines = get_diff_changes(diff_line)
                    diff_graph = build_graph(added_lines, removed_lines)
                    msg_code_list = [msg_line]
                    line_length = len(msg_code_list)
                    _, word_edge_index, node_feature = prepare_code2d(msg_code_list, line_length)
                    msg_graph = {
                        "word_edge_index": word_edge_index,
                        "node_feature": node_feature
                    }

                    diff_graphs.append(diff_graph)
                    msg_graphs.append(msg_graph)

                except Exception as e:
                    print(f"Error processing lines: {e}")

        # 组合 diff_graph 和 msg_graph，并添加 label
        combined_dict = {}
        for i in range(len(diff_graphs)):
            label_value = labels[i] if i < len(labels) else None
            combined_dict[i] = {
                "diff_graph": diff_graphs[i],
                "msg_graph": msg_graphs[i],
                "label": label_value
            }

        return combined_dict

    def convert_to_tensors(self, processed_dict):
        """将图数据转换为张量"""
        tensor_dict = {}
        id = 0
        for key, graphs in processed_dict.items():
            # 处理 diff_graph
            diff_graph = graphs['diff_graph']
            nodes = [
                node[1].get('encoding', {
                    'input_ids': torch.zeros(1, 1),
                    'attention_mask': torch.zeros(1, 1)
                })
                for node in diff_graph['nodes']
            ]
            if len(nodes) == 0:
                continue

            # 获取所有节点的 input_ids 长度
            max_len = max(
                node['input_ids'].size(1) if node['input_ids'].dim() >= 2
                else node['input_ids'].unsqueeze(0).size(1)
                for node in nodes
            )

            padded_input_ids = []
            padded_attention_mask = []
            for node in nodes:
                id +=1
                input_ids = node['input_ids']
                attention_mask = node['attention_mask']

                # 补齐
                pad_length = max_len - input_ids.size(1)
                if pad_length > 0:
                    padding = torch.zeros((input_ids.size(0), pad_length), dtype=input_ids.dtype)
                    input_ids = torch.cat([input_ids, padding], dim=1)
                    attention_mask = torch.cat([attention_mask, padding], dim=1)

                padded_input_ids.append(input_ids)
                padded_attention_mask.append(attention_mask)

            # 合并补齐后的张量
            input_ids = torch.cat(padded_input_ids, dim=0)
            attention_mask = torch.cat(padded_attention_mask, dim=0)
            node_features = input_ids

            # 构建 edge_index 和 edge_weights
            edge_index = []
            edge_weights = []

            for src, dst, data in diff_graph['edges']:
                src_idx = next(i for i, node in enumerate(diff_graph['nodes']) if node[0] == src)
                dst_idx = next(i for i, node in enumerate(diff_graph['nodes']) if node[0] == dst)

                edge_index.append((src_idx, dst_idx))
                weight = data.get('weight', 0.0)
                edge_weights.append(weight)

            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)

            # 处理 msg_graph
            msg_graph = graphs['msg_graph']
            word_edge_index = torch.tensor(msg_graph['word_edge_index'][0], dtype=torch.long)
            msg_edge_weights = torch.ones(word_edge_index.size(1), dtype=torch.float)

            # 将 node_features 加到 msg_graph 中
            msg_node_features = torch.tensor(msg_graph['node_feature'][0]['input_ids'], dtype=torch.float)

            # 获取 label
            label = graphs.get('label', None)
            label_tensor = torch.tensor(label, dtype=torch.long) if label is not None else None
            # 存储到 tensor_dict
            tensor_dict[key] = {
                'diff_graph': {
                    'node_features': node_features,
                    'edge_index': edge_index,
                    'edge_attr': edge_weights
                },
                'msg_graph': {
                    'word_edge_index': word_edge_index,
                    'edge_weight': msg_edge_weights,
                    'node_features': msg_node_features
                },
                'label': label_tensor  # 添加 label
            }

            # 检查 msg_graph 和 diff_graph 是否有超出 node_features 维度的 edge_index
            # diff_graph
            # msg_grap

        return tensor_dict
def similarity(a, b):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a, b).ratio()

def normalize_code(code):
    """对代码字符串进行简单的标准化（去掉空格）"""
    return "".join(code.split())

def bert_encode(input_lst):
    encoded_input = tokenizer(input_lst, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return encoded_input


def bert_case_encode(input_lst):
    encoded_input = tokenizer_msg(input_lst, padding=True, truncation=True, max_length=512, return_tensors='pt')
    return encoded_input


# 从512改成128
def parse_diff(diff_text):
    # 替换 <nl> 为换行符
    diff_text = diff_text.replace('<nl>', '\n')

    # 使用 splitlines() 分割文本并去除空行
    lines = [line.strip() for line in diff_text.splitlines() if line.strip()]

    return lines


def get_diff_changes(diff_text):
    diff_text = diff_text.replace('<nl>', '\n')
    lines = [line for line in diff_text.split('\n') if line.strip()]
    added_lines = []
    removed_lines = []

    for line in lines:
        line = line.lstrip()
        if line.startswith('+'):
            added_lines.append(line[1:].strip())
        elif line.startswith('-'):
            removed_lines.append(line[1:].strip())

    return added_lines, removed_lines


def prepare_code2d(code_list, line_length, max_seq_len=100,
                   to_lowercase=False, weighted_graph=True):
    """
    构造带有位置编码的 2D 图结构。

    参数:
        code_list: List[str]，代码段的列表。
        line_length: int，每行代码的长度（未使用，保留以兼容逻辑）。
        max_seq_len: int，最大序列长度。
        to_lowercase: bool，是否将 token 转换为小写。
        weighted_graph: bool，是否为加权图。

    返回:
        code2d: List[List[str]]，每段代码的 token。
        all_word_edge_index: List[np.ndarray]，每段代码的边索引。
        node_features: List[np.ndarray]，每段代码的节点特征（仅位置编码）。
    """
    window_size = 2
    code2d = []
    all_word_edge_index = []
    node_features = []

    for c in code_list:
        windows = []

        # 预处理代码
        c = re.sub(r'\s+', ' ', c)
        if to_lowercase:
            c = c.lower()

        # Tokenize 文本（直接用空格分割）
        token_list = c.strip().split()
        msg_embedding = bert_case_encode(token_list)
        node_features.append(msg_embedding)
        total_tokens = len(token_list)

        # 截断或填充
        if total_tokens > max_seq_len:
            token_list = token_list[:max_seq_len]
            total_tokens = max_seq_len
        else:
            token_list = token_list + ['<pad>'] * (max_seq_len - total_tokens)

        code2d.append(token_list)

        # 滑动窗口
        idx = range(0, total_tokens)
        if total_tokens <= window_size:
            windows.append(idx)
        else:
            for j in range(total_tokens - window_size + 1):
                window = idx[j: j + window_size]
                windows.append(window)

        # 统计边的权重
        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p_id = window[p]
                    word_q_id = window[q]

                    # 检查索引是否超出范围
                    if word_p_id >= max_seq_len or word_q_id >= max_seq_len:
                        continue  # 如果超出范围，跳过该边

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

        # 构造图的边索引和权重
        row = []
        col = []
        weight = []
        for key in word_pair_count:
            p = key[0]
            q = key[1]

            # 检查索引是否超出范围
            if p >= max_seq_len or q >= max_seq_len:
                continue  # 如果超出范围，跳过该边

            row.append(p)
            col.append(q)
            weight.append(word_pair_count[key] if weighted_graph else 1.)

        adj = sp.csr_matrix((weight, (row, col)), shape=(max_seq_len, max_seq_len))
        edge_index = np.array(adj.nonzero())

        # 确保 edge_index 中的索引不超过 node_features 的大小
        edge_index = edge_index[:, edge_index[0] < node_features[0]['input_ids'].size(0)]

        all_word_edge_index.append(edge_index)

    return code2d, all_word_edge_index, node_features

def build_graph(added_lines, removed_lines):
    """构建diff图，并返回节点特征、边索引和边属性"""
    G = nx.DiGraph()

    # 记录节点信息
    node_mapping = {}  # 记录节点名到索引的映射

    # 添加节点
    for i, line in enumerate(added_lines):
        node_name = f'added_{i}'
        encoding = bert_encode(line)
        G.add_node(f"added_{i}", label=line, encoding=encoding)

    for i, line in enumerate(removed_lines):
        node_name = f'removed_{i}'
        encoding = bert_encode(line)
        G.add_node(f"removed_{i}", label=line, encoding=encoding)

    # 添加上下文关系（顺序关系）
    for i in range(len(added_lines) - 1):
        src, dst = f'added_{i}', f'added_{i + 1}'
        G.add_edge(src, dst, weight=1, type='context')

    for i in range(len(removed_lines) - 1):
        src, dst = f'removed_{i}', f'removed_{i + 1}'
        G.add_edge(src, dst, weight=1, type='context')

    # 添加修改关系（相似度 > 0.5 才建立 modify 关系）
    for i, removed_line in enumerate(removed_lines):
        for j, added_line in enumerate(added_lines):
            sim_score = similarity(normalize_code(removed_line), normalize_code(added_line))
            if sim_score > 0.5:
                src, dst = f'removed_{i}', f'added_{j}'
                G.add_edge(src, dst, weight=sim_score, type='modify')
                
    graph_content = {
        "nodes": list(G.nodes(data=True)),  # 获取节点和其属性
        "edges": list(G.edges(data=True)),  # 获取边和其属性
    }


    return graph_content


def save_model(epoch, model, training_stats, base_dir):
    out_dir = os.path.join(base_dir, f"epoch_{epoch}", 'model.ckpt')
    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir))

    print('Saving model to %s' % out_dir)
    torch.save(model.state_dict(), out_dir)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats.to_json(os.path.join(base_dir, f"epoch_{epoch}", "training_stats.json"))


def train_model(dir_path, keyword, language):
    config = Config()
    config_gmn = get_default_config()
    # 创建 DataLoader
    train_dt = Data_processor('train', config.batch_size, dir_path, keyword)
    traindata_loader = train_dt.processed_tensor_dataloader
    test_dt = Data_processor('val', config.batch_size, dir_path, keyword)
    testdata_loader = test_dt.processed_tensor_dataloader

    print('Dataloader created!')

    first_batch = next(iter(traindata_loader))

    node_feature_dim = first_batch['node_features'].shape[-1]
    edge_feature_dim = first_batch['edge_attr'].shape[-1]

    print("dim:", edge_feature_dim)

    model, optimizer = build_model(config_gmn, node_feature_dim, edge_feature_dim)
    base_dir = f'./codebert_todo_new{language}_maxlen512'

    print('codeBERT model created!')

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = config.num_epochs
    total_steps = len(traindata_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    seed_val = 3407
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    total_t0 = time.time()
    model.eval()

    loss_fn = Focal_loss(alpha=0.25, gamma=2, num_classes=2)
    print("-----Using focal loss")

    progress_bar = tqdm(range(total_steps))
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        epcho_train_loss = 0
        model.train()
        i = 0
        for batch in traindata_loader:
            # if step % 100 == 0 and not step == 0:
            #    elapsed = format_time(time.time() - t0)
            #    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}'.format(step, len(traindata_loader), elapsed))

            # 提取 diff_graph 和 msg_graph 的数据
            node_features = batch['node_features'].to(config.device)  # 节点特征 (batch_size, num_nodes, feature_dim)
            edge_index = batch['edge_index'].to(config.device)  # 边索引 (batch_size, 2, num_edges)
            edge_features = batch['edge_attr'].to(config.device)  # 边权重 (batch_size, num_edges)

            batch_labels = batch['labels'].to(config.device)  # 节点特征 (batch_size, num_nodes, feature_dim)
            graph_idx = batch['graph_idx'].to(config.device)  # 边索引 (batch_size, 2, num_edges)
            training_n_graphs_in_batch = batch['n_graphs']  # 边权重 (batch_size, num_edges)

            from_idx = edge_index[0]
            to_idx = edge_index[1]

            # 将数据组织为 GMN 的输入格式
            # diff_graph = (diff_node_features, diff_edge_index, diff_edge_attr)
            # msg_graph = (msg_node_features, msg_edge_index, msg_edge_attr)
            model = model.to(device)
            print("node_feature:", node_features.shape)
            print("edge_feature:", edge_features.shape)
            print("from_idx:", from_idx.shape)
            print("to_idx:", to_idx.shape)

            graph_vectors = model(node_features.to(device).float(), edge_features.to(device).float(),
                                  from_idx.to(device),
                                  to_idx.to(device), graph_idx.to(device), training_n_graphs_in_batch)

            print("graph_vector:", graph_vectors.shape)
            diff_vectors, msg_vectors = reshape_and_split_tensor(graph_vectors, 2)
            combined_vector = torch.cat([diff_vectors, msg_vectors], dim=-1)

            outputs = mlp_classifier(combined_vector)

            print("shape:", outputs.shape)
            # 模型训练
            model.zero_grad()
            loss = loss_fn(outputs.to(device), batch_labels.to(device))
            epcho_train_loss += loss.item()
            loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()  # 更新参数
            scheduler.step()  # 更新学习率
            optimizer.zero_grad()  # 清空梯度
            progress_bar.update(1)

        avg_train_loss = epcho_train_loss / len(traindata_loader)
        training_time = format_time(time.time() - t0)

        print("")
        print("====== Average training loss: {0:.2f}".format(avg_train_loss))
        print("====== Training epoch took: {:}".format(training_time))

        print("Running Testing....")
        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        for batch in testdata_loader:
            with torch.no_grad():
                node_features = batch['node_features'].to(config.device)  # 节点特征 (batch_size, num_nodes, feature_dim)
                edge_index = batch['edge_index'].to(config.device)  # 边索引 (batch_size, 2, num_edges)
                edge_features = batch['edge_attr'].to(config.device)  # 边权重 (batch_size, num_edges)

                batch_labels = batch['labels'].to(config.device)  # 节点特征 (batch_size, num_nodes, feature_dim)
                graph_idx = batch['graph_idx'].to(config.device)  # 边索引 (batch_size, 2, num_edges)
                training_n_graphs_in_batch = batch['n_graphs']  # 边权重 (batch_size, num_edges)

                from_idx = edge_index[0]
                to_idx = edge_index[1]

                # 将数据组织为 GMN 的输入格式
                model = model.to(device)

                graph_vectors = model(node_features.to(device).float(), edge_features.to(device).float(),
                                      from_idx.to(device),
                                      to_idx.to(device), graph_idx.to(device), training_n_graphs_in_batch)

                print("graph_vector:", graph_vectors.shape)
                diff_vectors, msg_vectors = reshape_and_split_tensor(graph_vectors, 2)
                combined_vector = torch.cat([diff_vectors, msg_vectors], dim=-1)

                b_outputs = mlp_classifier(combined_vector)
                print("b_shape:", b_outputs.shape)

            loss = loss_fn(b_outputs, batch_labels)
            total_eval_loss += loss.item()
            preds = torch.max(b_outputs.data, 1)[1].cpu().numpy()
            print("preds:", preds.shape)
            labels = batch_labels.cpu().numpy()
            print("label:", batch_labels)
            total_eval_accuracy += flat_accuracy(preds, labels)
            print("f_acc:", flat_accuracy(preds, labels))

        avg_val_accuracy = total_eval_accuracy / len(testdata_loader)
        print("len:", len(testdata_loader))
        length = len(testdata_loader)
        print("t_acc:", total_eval_accuracy)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(testdata_loader)
        test_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(test_time))

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        training_stats.append(
            {'epoch': epoch_i + 1,
             'Training Loss': avg_train_loss,
             'Valid. Loss': avg_val_loss,
             'Valid. Accur.': avg_val_accuracy,
             'Training Time': training_time,
             'Validation Time': test_time,
             'len': length,
             't_acc': total_eval_accuracy
             })

        save_model(epoch_i + 1, model, training_stats, base_dir)

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

# Example: Access the first combined graph
if __name__ == '__main__':
    JAVA_TGT_DIR = "./top_repos_10000/new_java/"
    PYTHON_TGT_DIR = "./top_repos_10000/new_python/"

    # todo �޸�����
    language = "python"  # �ɸ�Ϊ "python"
    dir_path = JAVA_TGT_DIR if language == "java" else PYTHON_TGT_DIR
    train_model(dir_path, 'todo', language)
    print(dir_path)