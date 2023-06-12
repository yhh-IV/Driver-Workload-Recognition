# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 13:45:17 2022

@author: Lenovo
"""

import torch
from torch import nn
import sys 

sys.path.append('C:/driver_workload/attention-hyperlstm-classification')
sys.path.append('C:/driver_workload/attention-hyperlstm-classification/workload_classification/models')  

from workload_classification.model_workload_att import RecModel

class Att_Model(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, 
                 hyper_hidden_size, hyper_embedding_size,
                 use_layer_norm, dropout_prob, output_size):
        super().__init__()
        
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.hyper_hidden_size = hyper_hidden_size
        self.hyper_embedding_size = hyper_embedding_size
        self.use_layer_norm = use_layer_norm
        self.dropout_prob = dropout_prob
        self.output_size = output_size
        
        self.EEG_cell = RecModel(
            rnn_type = rnn_type,
            input_size = 4,
            hidden_size = hidden_size,
            hyper_hidden_size = hyper_hidden_size,
            hyper_embedding_size = hyper_embedding_size,
            use_layer_norm = use_layer_norm,
            dropout_prob = dropout_prob,
            output_size = output_size)
        
        self.eye_cell = RecModel(
            rnn_type = rnn_type,
            input_size = 4,
            hidden_size = hidden_size,
            hyper_hidden_size = hyper_hidden_size,
            hyper_embedding_size = hyper_embedding_size,
            use_layer_norm = use_layer_norm,
            dropout_prob = dropout_prob,
            output_size = output_size)
        
        self.vehicle_cell = RecModel(
            rnn_type = rnn_type,
            input_size = 6,
            hidden_size = hidden_size,
            hyper_hidden_size = hyper_hidden_size,
            hyper_embedding_size = hyper_embedding_size,
            use_layer_norm = use_layer_norm,
            dropout_prob = dropout_prob,
            output_size = output_size)
        
        self.hidden_proj_EEG = nn.Linear(in_features=hidden_size,
                                         out_features=hidden_size,
                                         bias=False)
        
        self.hidden_proj_eye = nn.Linear(in_features=hidden_size,
                                         out_features=hidden_size,
                                         bias=False)
        
        self.hidden_proj_vehicle = nn.Linear(in_features=hidden_size,
                                         out_features=hidden_size,
                                         bias=False)
        
        self.scalar = nn.Softmax(2)
        self.pool = nn.MaxPool1d(kernel_size=3)
        
        # self.output_proj = nn.Linear(in_features=hidden_size,
        #                               out_features=output_size)
        
        self.output_proj = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=output_size),
                                          nn.ReLU(),
                                          nn.Linear(in_features=output_size, out_features=output_size))
            
    def forward(self, inputs, state, hyper_state=None):

        # print(inputs[:, 0:3].shape)
        hidden_EEG, _, _, _ = self.EEG_cell(inputs[:, :, 0:4], None)
        hidden_eye, _, _, _ = self.eye_cell(inputs[:, :, 4:8], None)
        hidden_vehicle, _, _, _ = self.vehicle_cell(inputs[:, :, 8:14], None)
        
        trans_hidden_EEG = self.hidden_proj_EEG(hidden_EEG)  # (batch size, hidden size)
        trans_hidden_eye = self.hidden_proj_eye(hidden_eye)
        trans_hidden_vehicle = self.hidden_proj_vehicle(hidden_vehicle)
        
        trans_hidden_EEG = torch.unsqueeze(trans_hidden_EEG, 2)
        trans_hidden_eye = torch.unsqueeze(trans_hidden_eye, 2)
        trans_hidden_vehicle = torch.unsqueeze(trans_hidden_vehicle, 2)
        
        trans_hidden = torch.cat((trans_hidden_EEG, trans_hidden_eye, trans_hidden_vehicle), axis=2)
        attention_scores = torch.matmul(trans_hidden.transpose(1, 2), trans_hidden)      
        attention_scores = self.scalar(attention_scores)
        
        attention_hidden = torch.matmul(attention_scores, trans_hidden.transpose(1, 2))
        attention_output = torch.squeeze(self.pool(attention_hidden.transpose(1, 2)))
        
        logits = self.output_proj(attention_output)
        
        return logits
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
