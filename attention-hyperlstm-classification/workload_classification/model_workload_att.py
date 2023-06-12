from torch import nn
from torch.nn import init
import sys 

sys.path.append('C:/driver_workload/attention-hyperlstm-classification')
sys.path.append('C:/driver_workload/attention-hyperlstm-classification/models')  

from models.hyperlstm import HyperLSTMCell, LSTMCell

class RecModel(nn.Module):

    def __init__(self, rnn_type, input_size, hidden_size, 
                 hyper_hidden_size, hyper_embedding_size,
                 use_layer_norm, dropout_prob, output_size):
        super().__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = dropout_prob
        self.output_size = output_size
 
        if rnn_type == 'hyperlstm':
            self.rnn_cell = HyperLSTMCell(
                input_size=input_size, 
                hidden_size=hidden_size,
                hyper_hidden_size=hyper_hidden_size,
                hyper_embedding_size=hyper_embedding_size,
                use_layer_norm=use_layer_norm, 
                dropout_prob=dropout_prob)
        elif rnn_type == 'lstm':
            self.rnn_cell = LSTMCell(
                input_size=input_size, 
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout_prob=dropout_prob)
        else:
            raise ValueError('Unknown RNN type')
        self.output_proj = nn.Linear(in_features=hidden_size,
                                     out_features=output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        self.rnn_cell.reset_parameters()
        init.xavier_uniform_(self.output_proj.weight.data)
        init.constant_(self.output_proj.bias.data, val=0)

    def forward(self, inputs, state, hyper_state=None):
        inputs_emb = inputs
        max_length = inputs.size(1)

        rnn_outputs = []
        for t in range(max_length):
            if self.rnn_type == 'hyperlstm':
                output, state, hyper_state = self.rnn_cell(
                    x=inputs_emb[:,t], state=state, hyper_state=hyper_state)
            elif self.rnn_type == 'lstm':
                output, state = self.rnn_cell.forward(
                    x=inputs_emb[:,t], state=state)
            else:
                raise ValueError('Unknown RNN type')
            rnn_outputs.append(output)
            # rnn_outputs.append(self.output_proj(output))
            
        rnn_outputs = rnn_outputs[-1]

        logits = self.output_proj(rnn_outputs)
        # logits = rnn_outputs
        
        return rnn_outputs, logits, state, hyper_state
