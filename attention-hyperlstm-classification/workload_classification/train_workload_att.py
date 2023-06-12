import argparse
import logging
import os
from pprint import pprint

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn.utils import clip_grad_norm

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sklearn.metrics as sm

import sys
sys.path.append('C:/driver_workload/attention-hyperlstm-classification')
sys.path.append('C:/driver_workload/attention-hyperlstm-classification/workload_classification')

from workload_classification.data_workload import load_dataset
from workload_classification.attention_workload import Att_Model

import random

log_formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_log_handler = logging.StreamHandler()
console_log_handler.setFormatter(log_formatter)
logger.addHandler(console_log_handler)

seed = 23
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

class FusionDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X_train, Y_train):
        'Initialization'
        self.X_train = X_train
        self.Y_train = Y_train

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X_train)

  def __getitem__(self, index):
        'Generates one sample of data'       
        # Load data and get label
        X = torch.FloatTensor(self.X_train[index])
        Y = torch.LongTensor(self.Y_train[index])
        X, Y = X.to(device), Y.to(device)

        return X, Y


def train(args):
    
    model = Att_Model(rnn_type=args.rnn_type,
                     input_size=args.input_size,
                     hidden_size=args.hidden_size,
                     hyper_hidden_size=args.hyper_hidden_size,
                     hyper_embedding_size=args.hyper_embedding_size,
                     use_layer_norm=args.use_layer_norm,
                     dropout_prob=args.dropout_prob,
                     output_size=args.output_size)
    # print(model)
    
    train_loader = DataLoader(FusionDataset(X_train, Y_train), batch_size=args.batch_size) 
    valid_loader = DataLoader(FusionDataset(X_test, Y_test), batch_size=args.batch_size)
    
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total parameters: {num_params}')
    
    if args.gpu > -1:
        model.cuda(args.gpu)
    optimizer = optim.Adam(params=model.parameters())

    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'log', 'train'))

    loss_fn = torch.nn.CrossEntropyLoss()
    global_step = 0
    best_valid_loss = 1e10

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        state = hyper_state = None
        for train_batch in tqdm(train_loader, desc=f'Epoch {epoch}: Training'):
            train_inputs, train_targets = train_batch   # training inputs and tragets
            train_logits = model(inputs=train_inputs, state=state, 
                                                     hyper_state=hyper_state)
            
            train_loss = loss_fn(train_logits, train_targets.view(-1))
            # train_loss = 0
            # for i in range(len(train_logits)):
            #     train_loss += loss_fn(train_logits[i], train_targets.view(-1)) * np.exp(-len(train_logits) + i)
            
            optimizer.zero_grad()
            train_loss.backward()
            clip_grad_norm(parameters=model.parameters(), max_norm=1)
            optimizer.step()

            global_step += 1
            summary_writer.add_scalar(
                tag='train_loss', scalar_value=train_loss.item(),
                global_step=global_step)
        
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        con_matrix_list = []
        
        Y_pred = [] 
        # load_prob = torch.tensor([])
        
        model.eval()
        valid_loss_sum = valid_loss_denom = 0
        state = hyper_state = None
        for valid_batch in tqdm(valid_loader,
                                desc=f'Epoch {epoch}: Validation'):
            valid_inputs, valid_targets = valid_batch
            with torch.no_grad():
                valid_logits = model(
                    inputs=valid_inputs, state=state, hyper_state=hyper_state)
            
            Y_pred = np.append(Y_pred, torch.argmax(valid_logits, axis=-1).cpu())
            # load_prob = torch.cat((load_prob, valid_logits), 0)	 #save valid_logits
            # Y_pred = np.append(Y_pred, torch.argmax(valid_logits[-1], axis=-1))

            valid_loss = loss_fn(valid_logits, valid_targets.view(-1))         
            # valid_loss = 0
            # for i in range(len(valid_logits)):
            #     valid_loss += loss_fn(valid_logits[i], valid_targets.view(-1)) * np.exp(-len(valid_logits) + i)
            
            valid_loss_sum += valid_loss.item()
            valid_loss_denom += 1
        
        # record metrics
        accuracy_list.append(accuracy_score(Y_test, Y_pred))
        precision_list.append(precision_score(Y_test, Y_pred, average='weighted'))
        recall_list.append(recall_score(Y_test, Y_pred, average='weighted'))
        f1_list.append(f1_score(Y_test, Y_pred, average='weighted'))
        con_matrix_list.append(sm.confusion_matrix(Y_test, Y_pred))
        

        # determine recogntion accuracy                  
        valid_loss = valid_loss_sum / valid_loss_denom
        valid_bpc = valid_loss 
        summary_writer.add_scalar(
            tag='valid_bpc', scalar_value=valid_bpc, global_step=global_step)
        logging.info(f'Epoch {epoch}: Valid Loss = {valid_loss:.6f}')
        logging.info(f'Epoch {epoch}: Valid BPC = {valid_bpc:.6f}')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model_filename = f'{epoch:03d}-{valid_bpc:.6f}.pt'
            model_path = os.path.join(args.save_dir, model_filename)
            torch.save(model, model_path)
            logging.info('Saved the new best checkpoint')
    

    print(accuracy_list)
    print(con_matrix_list[-1])
    accuracies.append(accuracy_list[-1])
    precisions.append(precision_list[-1])
    recalls.append(recall_list[-1])
    f1_scores.append(f1_list[-1])
    matrices.append(con_matrix_list[-1])  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/ptb_char')
    parser.add_argument('--rnn-type', default='hyperlstm',
                        choices=['hyperlstm', 'lstm'])
    parser.add_argument('--save-dir', default='log')
    parser.add_argument('--input-size', default=14, type=int)
    parser.add_argument('--hidden-size', default=128, type=int)
    parser.add_argument('--hyper-hidden-size', default=64, type=int)
    parser.add_argument('--hyper-embedding-size', default=16, type=int)
    parser.add_argument('--use-layer-norm', default=False, action='store_true')
    parser.add_argument('--dropout-prob', default=0.1, type=float)
    parser.add_argument('--output-size', default=3, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--max-epoch', default=20, type=int)
    args = parser.parse_args()

    config = {'model': {'rnn_type': args.rnn_type,
                        'input_size': args.input_size,
                        'hidden_size': args.hidden_size,
                        'hyper_hidden_size': args.hyper_hidden_size,
                        'hyper_embedding_size': args.hyper_embedding_size,
                        'use_layer_norm': args.use_layer_norm,
                        'dropout_prob': args.dropout_prob,
                        'output_size': args.output_size},
              'train': {'batch_size': args.batch_size,}}
    pprint(config)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    file_log_handler = logging.FileHandler(
        os.path.join(args.save_dir, 'train.log'))
    file_log_handler.setFormatter(log_formatter)
    logger.addHandler(file_log_handler)
    
    train(args)
    
if __name__ == '__main__':
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    matrices = []
    
    # gpu
    device = torch.device('cuda:0')

    X, Y = load_dataset()      
    kf = KFold(n_splits = 5, shuffle=False)
    
    file_num = 14 * 3
    per_length = int(X.shape[0]/file_num)

    sample = X[0: per_length]

    for per_train_index, per_test_index in kf.split(sample): 
           X_train, X_test = np.zeros((0, X.shape[1], X.shape[2])),np.zeros((0, X.shape[1], X.shape[2]))
           Y_train, Y_test = np.zeros((0, Y.shape[1])), np.zeros((0, Y.shape[1]))
           for i in range(0, X.shape[0], per_length):
               X_train = np.append(X_train, X[i: i+per_length][per_train_index], axis=0)
               X_test = np.append(X_test, X[i: i+per_length][per_test_index], axis=0)
               Y_train = np.append(Y_train, Y[i: i+per_length][per_train_index], axis=0)
               Y_test = np.append(Y_test, Y[i: i+per_length][per_test_index], axis=0)
           np.random.seed(0)
           np.random.shuffle(X_train)
           np.random.seed(0)
           np.random.shuffle(Y_train)
           np.random.seed(0)
           np.random.shuffle(X_test)
           np.random.seed(0)
           np.random.shuffle(Y_test)
            
           main()
            
    mean_matrix = (matrices[0] + matrices[1] + matrices[2] + matrices[3] + matrices[4])/5
    mean_matrix = np.around(mean_matrix, decimals=6)
    print("acc_mean:", np.mean(accuracies), "acc_std:", np.std(accuracies))
    print("pre_mean:", np.mean(precisions), "pre_std:", np.std(precisions))
    print("rec_mean:", np.mean(recalls), "rec_std:", np.std(recalls))
    print("f1_mean:", np.mean(f1_scores), "f1_std:", np.std(f1_scores))
    print(mean_matrix)

    
    
    
    
    
