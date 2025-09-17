# encoding=utf-8
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from BERT_model import TransformerModel, Config, load_codeBERT
from loss_func import Focal_loss, DiceLoss
from LabelSmoothing import LabelSmoothingLoss
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
from MLPClassifier import MLPClassifier  # 导入 MLPClassifier
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
        self.diff_graph ,self.diff_types ,self.diff_weights = self.graph_to_edge_index(self.diff_graph, use_node_mapping=True)
        self.msg_graph , self.msg_types , self.msg_weights = self.graph_to_edge_index(self.msg_graph, use_node_mapping=False)
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

    def graph_to_edge_index_and_type(self, graph_list, use_node_mapping=True):
        """
        将图结构转换为 edge_index、edge_type、edge_weight Tensor 列表。
    
        返回：
            - edge_index_list: 每个图的 edge_index，shape [2, num_edges]
            - edge_type_list: 每条边的类型（整数编码），shape [num_edges]
            - edge_weight_list: 每条边的权重（float），shape [num_edges]
        """
        EDGE_TYPE_MAP = {'context': 0, 'modify': 1, 'data_flow': 2}
    
        all_edge_indices = []
        all_edge_types = []
        all_edge_weights = []
    
        for graph in graph_list:
            edges = graph["edges"]
    
            if use_node_mapping:
                nodes = graph["nodes"]
                node2idx = {node: idx for idx, (node, _) in enumerate(nodes)}
    
                edge_index_list = []
                edge_type_list = []
                edge_weight_list = []
    
                for src, dst, attr in edges:
                    src_idx = node2idx[src]
                    dst_idx = node2idx[dst]
                    edge_index_list.append([src_idx, dst_idx])
    
                    edge_type_str = attr.get('type', 'context')
                    edge_type_list.append(EDGE_TYPE_MAP.get(edge_type_str, 0))
    
                    edge_weight = attr.get('weight', 1.0)
                    edge_weight_list.append(float(edge_weight))
    
                edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
                edge_type = torch.tensor(edge_type_list, dtype=torch.long)
                edge_weight = torch.tensor(edge_weight_list, dtype=torch.float32)
    
                all_edge_indices.append(edge_index)
                all_edge_types.append(edge_type)
                all_edge_weights.append(edge_weight)
            else:
                edges = graph["word_edge_index"]
                edge_index = torch.tensor(edges, dtype=torch.long).squeeze(0)  # shape [2, num_edges]
                all_edge_indices.append(edge_index)
                # 注意：如果没有边类型/权重，这里可能需要填充默认值
                all_edge_types.append(None)
                all_edge_weights.append(None)
    
        return all_edge_indices, all_edge_types, all_edge_weights


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
        def __init__(self, diff_ids, diff_masks, diff_edge_indices, diff_edge_types, diff_edge_weights,
                     msg_ids, msg_masks, msg_edge_indices, msg_edge_types, msg_edge_weights,
                     labels):
            self.diff_ids = diff_ids
            self.diff_masks = diff_masks
            self.diff_edge_indices = diff_edge_indices
            self.diff_edge_types = diff_edge_types
            self.diff_edge_weights = diff_edge_weights
    
            self.msg_ids = msg_ids
            self.msg_masks = msg_masks
            self.msg_edge_indices = msg_edge_indices
            self.msg_edge_types = msg_edge_types
            self.msg_edge_weights = msg_edge_weights
    
            self.labels = labels
    
        def __len__(self):
            return len(self.labels)
    
        def __getitem__(self, idx):
            return {
                "diff_ids": self.diff_ids[idx],
                "diff_mask": self.diff_masks[idx],
                "diff_edge_index": self.diff_edge_indices[idx],
                "diff_edge_type": self.diff_edge_types[idx],
                "diff_edge_weight": self.diff_edge_weights[idx],
    
                "msg_ids": self.msg_ids[idx],
                "msg_mask": self.msg_masks[idx],
                "msg_edge_index": self.msg_edge_indices[idx],
                "msg_edge_type": self.msg_edge_types[idx],
                "msg_edge_weight": self.msg_edge_weights[idx],
    
                "label": self.labels[idx],
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
        msg_types = self.msg_types
        diff_types = self.diff_types
        msg_weights = self.msg_weights
        diff_weights = self.diff_weights

        tensor_data = self.GraphCodeDataset(self.train_diff[0], self.train_diff[1], diff_edge_index_list,diff_types,diff_weights,
                                            self.train_msg[0],
                                            self.train_msg[1], msg_edge_index_list,msg_types,msg_weights, self.train_diff[2])
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

def extract_variables(code_line: str) -> List[str]:
    """
    提取潜在变量名（假设变量是符合标识符规则的 token）
    你也可以用 AST 做更精细的提取
    """
    return re.findall(r'\b[_a-zA-Z][_a-zA-Z0-9]*\b', code_line)


def build_graph(added_lines: List[str], removed_lines: List[str], sim_threshold=0.5):
    G = nx.MultiDiGraph()

    for i, line in enumerate(added_lines):
        G.add_node(f"added_{i}", label=line)

    for i, line in enumerate(removed_lines):
        G.add_node(f"removed_{i}", label=line)

    # 顺序边
    for i in range(len(added_lines) - 1):
        G.add_edge(f"added_{i}", f"added_{i+1}", weight=1.0, type="context")

    for i in range(len(removed_lines) - 1):
        G.add_edge(f"removed_{i}", f"removed_{i+1}", weight=1.0, type="context")

    # 相似边（modify）
    for i, removed_line in enumerate(removed_lines):
        for j, added_line in enumerate(added_lines):
            sim_score = similarity(normalize_code(removed_line), normalize_code(added_line))
            if sim_score > sim_threshold:
                G.add_edge(f"removed_{i}", f"added_{j}", weight=sim_score, type="modify")

    # ✅ 变量依赖边（data_flow）
    for i, removed_line in enumerate(removed_lines):
        removed_vars = set(extract_variables(removed_line))
        for j, added_line in enumerate(added_lines):
            added_vars = set(extract_variables(added_line))
            common_vars = removed_vars & added_vars
            if common_vars:
                # 可选：边权设置为重合变量数量 or 1.0
                G.add_edge(f"removed_{i}", f"added_{j}", weight=len(common_vars), type="data_flow")

    return {
        "nodes": list(G.nodes(data=True)),
        "edges": list(G.edges(data=True))
    }


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
    model = TransformerModel(config).to(config.device)
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

    # We'll store a number of quantities such as training and validation loss, validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    model.eval()
    # todo change loss function
    print("-----Using focal loss")
    # loss_fn = F.cross_entropy
    loss_fn = Focal_loss(alpha=0.25, gamma=2, num_classes=2)
    #print("-----Using Label Smoothing Loss")
    #loss_fn = LabelSmoothingLoss(classes=2, smoothing=0.05)
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
            diff_types = batch['diff_edge_types']

            model.zero_grad()
            diff_input = (diff_input_ids, diff_input_mask)
            msg_input = (msg_input_ids, msg_input_mask)
            batch_outputs = model(diff_input, msg_input, diff_edge_index, diff_types,msg_edge_index)
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
                diff_types = batch['diff_edge_types']
    
                model.zero_grad()
                diff_input = (diff_input_ids, diff_input_mask)
                msg_input = (msg_input_ids, msg_input_mask)
                b_outputs = model(diff_input, msg_input, diff_edge_index, diff_types,msg_edge_index)

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
