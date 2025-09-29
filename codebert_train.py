# encoding=utf-8
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from BERT_model import TODO_Checker, Config, load_codeBERT
from loss_func import Focal_loss, DiceLoss
import re
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
import networkx as nx
from transformers import BertTokenizer
import torch
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
from configure import *
from difflib import SequenceMatcher

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class Data_processor(object):

    def __init__(self, set_type, batch_size, dir_path, keyword):
        self.set_type = set_type
        self.batch_size = batch_size
        self.dir_path = dir_path
        self.keyword = keyword
        print("Loading codeBERT model...")
        self.bert_model, self.tokenizer = load_codeBERT()
        print('codeBERT loaded')

        print('Loading %s dataset...' % (set_type + keyword))
        diff_type, msg_type, label_type = load_data(self.dir_path, set_type, keyword)
        path_diff, path_msg, flag_path = load_path(self.dir_path, set_type, keyword)
        print("Begin encoding...")
        self.encoded_diff = self.bert_encode_pair(diff_type, msg_type)
        self.encoded_msg = self.bert_encode_pair(msg_type, diff_type)
        self.labels = np.asarray(label_type)
        self.diff_graph, self.msg_graph = self.process_files(path_diff, path_msg, self.labels)
        print("%s dataset loaded.")

        print("Making data loader...")
        self.train_diff = self.make_data(self.encoded_diff)
        self.train_msg = self.make_data(self.encoded_msg)
        self.diff_graph = self.graph_to_edge_index(self.diff_graph, use_node_mapping=True)
        self.msg_graph = self.graph_to_edge_index(self.msg_graph, use_node_mapping=False)
        self.processed_dataloader = self.make_loader()
        print("Finished making data loader.")

    def process_files(self, diff_file_path, msg_file_path, labels):
        if not os.path.isfile(diff_file_path) or not os.path.isfile(msg_file_path):
            print("One or both files not found.")
            return {}

        diff_graphs = []
        msg_graphs = []

        with open(diff_file_path, 'r', encoding='utf-8') as diff_file, open(msg_file_path, 'r',
                                                                            encoding='utf-8') as msg_file:
            for idx, (diff_line, msg_line) in enumerate(zip(diff_file, msg_file)):
                diff_line = diff_line.strip()
                msg_line = msg_line.strip()

                if not diff_line or not msg_line:
                    continue

                added_lines, removed_lines = get_diff_changes(diff_line)
                diff_graph = build_graph(added_lines, removed_lines)

                msg_code_list = [msg_line]
                line_length = len(msg_code_list)
                word_edge_index = self.prepare_code2d(msg_code_list, line_length)

                # ✅ 自动检查 edge_index 的格式
                if not isinstance(word_edge_index, torch.Tensor):
                    word_edge_index = torch.tensor(word_edge_index, dtype=torch.long)

                if word_edge_index.size(1) != 2:
                    raise ValueError(
                        f"[Error] Sample {idx} has invalid msg_edge_index shape: "
                        f"expected (2, num_edges), but got {word_edge_index.shape}"
                    )

                msg_graph = {
                    "word_edge_index": word_edge_index
                }

                diff_graphs.append(diff_graph)
                msg_graphs.append(msg_graph)

        return diff_graphs, msg_graphs
        
    def prepare_code2d(self,code_list, line_lenth, max_seq_len=100, to_lowercase=False, weighted_graph=False):
        window_size = 2
        code2d = []
        all_word_edge_index = []
    
        for c in code_list:
            windows = []
    
            c = re.sub('\\s+', ' ', c)
    
            if to_lowercase:
                c = c.lower()
    
            token_list = self.tokenizer.tokenize(c)
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
    
        return all_word_edge_index

    def graph_to_edge_index(self, graph_list, use_node_mapping=True):
        """
        将图结构转换为 edge_index Tensor 列表。

        参数:
            graph_list (List[Dict]): 每个图是一个字典，包含 "edges" 和可选的 "nodes"
            use_node_mapping (bool): 是否对节点名进行编号映射（True 适用于 diff_graph，False 适用于 msg_graph）

        返回:
            List[Tensor]: 每个图对应的 edge_index Tensor（shape: [2, num_edges]）
        """
        all_edge_indices = []

        for graph in graph_list:
            if use_node_mapping:
                # 需要将节点名映射为索引
                nodes = graph["nodes"]
                edges = graph["edges"]

                node2idx = {node: idx for idx, (node, _) in enumerate(nodes)}
                edge_index_list = []

                for src, dst, _ in edges:
                    edge_index_list.append([node2idx[src], node2idx[dst]])

                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
                all_edge_indices.append(edge_index)
            else:
                # edges 是 numpy array，shape = (2, num_edges)
                edges = graph["word_edge_index"]
                edge_index = torch.tensor(edges, dtype=torch.long)  # shape: (2, num_edges)
                edge_index = edge_index.squeeze(0)  # 得到 (2, num_edges)
                print(f"[graph_to_edge_index] msg_edge_index 形状: {edge_index.shape}")
                all_edge_indices.append(edge_index)

        return all_edge_indices

    def bert_encode(self, input_lst):
        # bert encoding
        encoded_input = self.tokenizer(input_lst, padding=True, truncation=True, max_length=512, return_tensors='pt')
        return encoded_input

    def bert_encode_pair(self, input_lst1, input_lst2):
        encoded_input_pair = self.tokenizer(input_lst1, input_lst2,
                                            padding=True, truncation=True,
                                            max_length=512, return_tensors='pt')
        return encoded_input_pair

    def make_data(self, encoded_data):
        input_ids, attention_masks = encoded_data['input_ids'], encoded_data['attention_mask']
        # Convert to Pytorch Data Types
        inputs = torch.tensor(input_ids)
        masks = torch.tensor(attention_masks)
        labels = torch.tensor(self.labels)
        train_data = (inputs, masks, labels)
        return train_data

    class GraphCodeDataset(Dataset):
        def __init__(self, diff_ids, diff_masks, diff_graphs,
                     msg_ids, msg_masks, msg_graphs,
                     labels):
            self.diff_ids = diff_ids
            self.diff_masks = diff_masks
            self.diff_graphs = diff_graphs

            self.msg_ids = msg_ids
            self.msg_masks = msg_masks
            self.msg_graphs = msg_graphs

            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            if idx >= len(self.msg_graphs):
                print(f"Index {idx} is out of range for msg_graphs with length {len(self.msg_graphs)}")
                raise IndexError("Index out of range for msg_graphs")
            return {
                "diff_ids": self.diff_ids[idx],
                "diff_mask": self.diff_masks[idx],
                "msg_ids": self.msg_ids[idx],
                "msg_mask": self.msg_masks[idx],
                "label": self.labels[idx],
                "diff_edge_index": self.diff_graphs[idx],
                "msg_edge_index": self.msg_graphs[idx],
            }

    def graph_collate_fn(self, batch):
        batch_dict = {key: [item[key] for item in batch] for key in batch[0].keys()}
        batch_dict["diff_ids"] = torch.stack(batch_dict["diff_ids"])
        batch_dict["diff_mask"] = torch.stack(batch_dict["diff_mask"])
        batch_dict["msg_ids"] = torch.stack(batch_dict["msg_ids"])
        batch_dict["msg_mask"] = torch.stack(batch_dict["msg_mask"])
        batch_dict["label"] = torch.tensor(batch_dict["label"], dtype=torch.long)
        return batch_dict

    def make_loader(self):
        msg_edge_index_list = self.msg_graph
        diff_edge_index_list = self.diff_graph


        tensor_data = self.GraphCodeDataset(self.train_diff[0], self.train_diff[1], diff_edge_index_list,
                                            self.train_msg[0],
                                            self.train_msg[1], msg_edge_index_list, self.train_diff[2])
        dataloader = DataLoader(tensor_data, batch_size=self.batch_size, collate_fn=self.graph_collate_fn)

        return dataloader

    def save_loader(self):
        with open(self.dir_path + self.set_type + '_dataloader.pkl', 'wb') as hander:
            pickle.dump(self.processed_dataloader, hander)
        pass


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


def similarity(a, b):
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a, b).ratio()


def normalize_code(code):
    """对代码字符串进行简单的标准化（去掉空格）"""
    return "".join(code.split())


def build_graph(added_lines, removed_lines):
    """构建diff图，并返回节点特征、边索引和边属性"""
    G = nx.DiGraph()

    # 记录节点信息
    node_mapping = {}  # 记录节点名到索引的映射

    # 添加节点
    for i, line in enumerate(added_lines):
        # node_name = f'added_{i}'
        # encoding = bert_encode(line)
        G.add_node(f"added_{i}", label=line)

    for i, line in enumerate(removed_lines):
        # node_name = f'removed_{i}'
        # encoding = bert_encode(line)
        G.add_node(f"removed_{i}", label=line)

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


def save_model(epoch, model, training_stats):
    # todo change the model dir
    base_dir = './codebert_todo_newjava_maxlen512/epoch_' + str(epoch) + '/'
    out_dir = base_dir + 'model.ckpt'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print('Saving model to %s' % out_dir)
    torch.save(model.state_dict(), out_dir)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats.to_json(base_dir + "training_stats.json")


def train_model(dir_path, keyword):
    config = Config()
    train_dt = Data_processor('train', config.batch_size, dir_path, keyword)
    traindata_loader = train_dt.processed_dataloader
    test_dt = Data_processor('val', config.batch_size, dir_path, keyword)
    testdata_loader = test_dt.processed_dataloader
    print('Dataloader maked!')
    model = TransformerModel1(config).to(config.device)
    print('codeBERT model created!')

    # Optimizer & Learning Rate Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the training data.
    epochs = config.num_epochs
    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(traindata_loader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # We are ready to kick off the training
    # Set the seed value all over the place to make this reproducible.
    seed_val = 3407
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # We'll store a number of quantities such as training and validation loss, validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    model.eval()
    # todo change loss function
    print("-----Using focal loss")
    # loss_fn = F.cross_entropy
    loss_fn = Focal_loss(alpha=0.25, gamma=2, num_classes=2)
    # with_logits=True, ohem_ratio=0.01
    # loss_fn = DiceLoss(with_logits=True, ohem_ratio=0.1)
    print("Begin training...")
    progress_bar = tqdm(range(total_steps))
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        epcho_train_loss = 0
        model.train()

        for step, batch in enumerate(traindata_loader):
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(traindata_loader), elapsed))

            diff_input_ids = batch['diff_ids'].to(config.device)
            diff_input_mask = batch['diff_mask'].to(config.device)
            msg_input_ids = batch['msg_ids'].to(config.device)
            msg_input_mask = batch['msg_mask'].to(config.device)
            batch_labes = batch['label'].to(config.device)
            diff_edge_index = batch['diff_edge_index']  # list of edge_index tensors
            msg_edge_index = batch['msg_edge_index']

            model.zero_grad()
            diff_input = (diff_input_ids, diff_input_mask)
            msg_input = (msg_input_ids, msg_input_mask)
            batch_outputs = model(diff_input, msg_input, diff_edge_index, msg_edge_index)
            loss = loss_fn(batch_outputs, batch_labes)
            epcho_train_loss += loss.item()
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Calculate the average loss over all of the batches.
        avg_train_loss = epcho_train_loss / len(traindata_loader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("====== Average training loss: {0:.2f}".format(avg_train_loss))
        print("====== Training epcoh took: {:}".format(training_time))

        print("Running Testing....")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        for batch in testdata_loader:
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # values prior to applying an activation function like the softmax.
                diff_input_ids = batch['diff_ids'].to(config.device)
                diff_input_mask = batch['diff_mask'].to(config.device)
                msg_input_ids = batch['msg_ids'].to(config.device)
                msg_input_mask = batch['msg_mask'].to(config.device)
                batch_labes = batch['label'].to(config.device)
                diff_edge_index = batch['diff_edge_index']  # list of edge_index tensors
                msg_edge_index = batch['msg_edge_index']

                model.zero_grad()
                diff_input = (diff_input_ids, diff_input_mask)
                msg_input = (msg_input_ids, msg_input_mask)
                for index in msg_edge_index:
                    print(index.shape)
                b_outputs = model(diff_input, msg_input, diff_edge_index, msg_edge_index)

            loss = loss_fn(b_outputs, batch_labes)
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # move labels to CPU
            preds = torch.max(b_outputs.data, 1)[1].cpu().numpy()
            # print("preds:", type(preds), preds.shape)
            labels = batch_labes.to('cpu').numpy()
            # Calculate the accuracy for this batch of test sentences, and
            total_eval_accuracy += flat_accuracy(preds, labels)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(testdata_loader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(testdata_loader)

        # Measure how long the validation run took.
        test_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(test_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {'epoch': epoch_i + 1,
             'Training Loss': avg_train_loss,
             'Valid. Loss': avg_val_loss,
             'Valid. Accur.': avg_val_accuracy,
             'Training Time': training_time,
             'Validation Time': test_time
             })

        save_model(epoch_i + 1, model, training_stats)

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


if __name__ == '__main__':
    # todo change the dir path and the path of model
    # new dataset (filter out fixme xxx)
    JAVA_TGT_DIR = "./top_repos_10000/new_java/"
    PYTHON_TGT_DIR = "./top_repos_10000/new_python/"
    dir_path = JAVA_TGT_DIR
    train_model(dir_path, 'todo')
    print(dir_path)

