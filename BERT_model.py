from utils import *
import transformers
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import RGCNConv
transformers.logging.set_verbosity_error()


def load_BERT():
    ## Tokenize & Input Formatting
    ## Import model/tokenizer
    ## Load the BERT model
    bert_model = BertModel.from_pretrained('./Model/bert-base-uncased')
    bert_model.cuda()
    tokenizer = BertTokenizer.from_pretrained('./Model/bert-base-uncased')
    return bert_model, tokenizer


def load_codeBERT():
    model_path = './Model/codebert-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaModel.from_pretrained(model_path)
    model.cuda()
    return model, tokenizer


class Config(object):

    def __init__(self):
        self.model_name = 'bert'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 2
        self.codebert_path = './Model/codebert-base'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.codebert_path)
        self.hidden_size = 768
        self.batch_size = 16  # 向cuda内存妥协
        # todo change the number of epochs, there is no need to have 10 epochs
        self.num_epochs = 6
        self.vocab_size = len(self.tokenizer)


class TODO_Checker(nn.Module):
    def __init__(self, config):
        super(TransformerModel1, self).__init__()

        self.codebert1 = RobertaModel.from_pretrained(config.codebert_path)
        self.codebert2 = RobertaModel.from_pretrained(config.codebert_path)

        self.graph_attention_module = GATv2Conv(config.hidden_size, config.hidden_size, heads=1)

        self.cross_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=8, batch_first=True)

        self.fc0 = nn.Linear(config.hidden_size, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, config.num_classes)

        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(config.hidden_size)

    def clip_edge_index(self, edge_index, max_node_num):
        mask = (edge_index[0] < max_node_num) & (edge_index[1] < max_node_num)
        return edge_index[:, mask]

    def forward(self, diff_input, msg_input, graph_edge_index_diff, graph_edge_index_msg):
        diff_ids, diff_mask = diff_input
        msg_ids, msg_mask = msg_input

        diff_outputs = self.codebert1(input_ids=diff_ids, attention_mask=diff_mask)
        msg_outputs = self.codebert2(input_ids=msg_ids, attention_mask=msg_mask)

        diff_encoded = diff_outputs.last_hidden_state
        msg_encoded = msg_outputs.last_hidden_state

        device = diff_encoded.device
        diff_graph_encoded_list = []
        msg_graph_encoded_list = []

        for i in range(diff_encoded.size(0)):
            if graph_edge_index_diff[i].numel() == 0:
                diff_graph_encoded = torch.zeros_like(diff_encoded[i])
            else:
                edge_index = self.clip_edge_index(graph_edge_index_diff[i], diff_encoded[i].size(0))
                diff_graph_encoded = self.graph_attention_module(diff_encoded[i], edge_index.to(device))
                diff_graph_encoded = diff_graph_encoded + diff_encoded[i]  
            diff_graph_encoded_list.append(diff_graph_encoded)
            if graph_edge_index_msg[i].numel() == 0:
                msg_graph_encoded = torch.zeros_like(msg_encoded[i])
            else:
                edge_index = self.clip_edge_index(graph_edge_index_msg[i], msg_encoded[i].size(0))
                msg_graph_encoded = self.graph_attention_module(msg_encoded[i], edge_index.to(device))
                msg_graph_encoded = msg_graph_encoded + msg_encoded[i] 
            msg_graph_encoded_list.append(msg_graph_encoded)

        diff_graph_encoded = torch.stack(diff_graph_encoded_list, dim=0)
        msg_graph_encoded = torch.stack(msg_graph_encoded_list, dim=0)

        cross_out, _ = self.cross_attention(
            query=diff_graph_encoded,
            key=msg_graph_encoded,
            value=msg_graph_encoded,
            key_padding_mask=(msg_mask == 0))

        fused_encoded = self.norm(cross_out + diff_graph_encoded)
        fused_vec = fused_encoded[:, 0, :]  # (batch_size, hidden_size)

        x = self.dropout(torch.relu(self.fc0(fused_vec)))
        x = self.dropout(torch.relu(self.fc1(x)))
        output = self.fc2(x)

        return output
        


