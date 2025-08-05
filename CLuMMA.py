import argparse
from textwrap import dedent
from Bio import SeqIO
import numpy as np
import random
from layers import SNNModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.optim import lr_scheduler
import pandas as pd




MAX_LEN = 50

def one_hot_padding(seq_list,padding):
    """
    Generate features for aa sequences [one-hot encoding with zero padding].
    Input: seq_list: list of sequences, 
           padding: padding length, >= max sequence length.
    Output: one-hot encoding of sequences.
    """
    feat_list = []
    one_hot = {}
    aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    pos, neg, neut_charge=['K', 'R'], ['D', 'E'], ['A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    hpho, pol, neut_hydro=['F', 'I', 'W', 'C'], ['K', 'D', 'E', 'Q', 'P', 'S', 'R', 'N', 'T', 'G'], ['A', 'H', 'Y', 'M', 'L', 'V']
    for i in range(len(aa)):
        one_hot[aa[i]] = [0]*26
        one_hot[aa[i]][i] = 1 
        if aa[i] in pos:
            one_hot[aa[i]][20]=1
        elif aa[i] in neg:
            one_hot[aa[i]][21]=1
        elif aa[i] in neut_charge:
            one_hot[aa[i]][22]=1
        
        if aa[i] in hpho:
            one_hot[aa[i]][23]=1
        elif aa[i] in pol:
            one_hot[aa[i]][24]=1
        elif aa[i] in neut_hydro:
            one_hot[aa[i]][25]=1
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
            # one_hot[seq_list[i][j]][-1]=(j/len(seq_list))
            feat.append(one_hot[seq_list[i][j]])
        feat = feat + [[0]*26]*(padding-len(seq_list[i]))
        feat_list.append(feat)
    feat_list=torch.from_numpy(np.array(feat_list))
    return feat_list




def predict_by_class(scores):
    """
    Turn prediction scores into classes.
    If score > 0.5, label the sample as 1; else 0.
    Input: scores - scores predicted by the model, 1-d array.
    Output: an array of 0s and 1s.
    """
    classes = []
    for i in range(len(scores)):
        if scores[i]>0.5:
            classes.append(1)
        else:
            classes.append(0)
    return torch.from_numpy(np.array(classes))


def positional_encodings(x, D):
    # input x a vector of positions
    encodings = torch.zeros((x.size(0), x.size(1), D))
    if x.is_cuda:
        encodings = encodings.cuda(x.get_device())

    # encodings = torch.autograd.Variable(encodings)

    for j in range(x.size(0)):
        for channel in range(x.size(1)):
          if channel % 2 == 0:
              encodings[j][channel] = torch.sin(
                  x[j][channel] / 10000 ** (channel / D))
          else:
              encodings[j][channel] = torch.cos(
                  x[j][channel] / 10000 ** ((channel - 1) / D))
    return encodings


def main():
    parser = argparse.ArgumentParser(description=dedent('''
        CLuMMA v1.1.0 training
        ------------------------------------------------------
        Given training sets with two labels: AMP and non-AMP,
        train the AMP prediction model.    
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-amp_tr', help="Training AMP set, fasta file", required=True)
    parser.add_argument('-non_amp_tr', help="Training non-AMP set, fasta file", required=True)
    parser.add_argument('-amp_te', help="Test AMP set, fasta file (optional)", default=None, required=False)
    parser.add_argument('-non_amp_te', help="Test non-AMP set, fasta file (optional)", default=None, required=False)
    parser.add_argument('-sample_ratio', 
                        help="Whether the training set is balanced or not (balanced by default, optional)", 
                        choices=['balanced', 'imbalanced'], default='balanced', required=False)
    parser.add_argument('-out_dir', help="Output directory", required=True)
    parser.add_argument('-model_name', help="File name of trained model weights", required=True)
    
    args = parser.parse_args()
    
    # load training sets
    AMP_train = []
    non_AMP_train = []
    for seq_record in SeqIO.parse(args.amp_tr, 'fasta'):
        # "../data/AMPlify_AMP_train_common.fa"
        AMP_train.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(args.non_amp_tr, 'fasta'):
        # "../data/AMPlify_non_AMP_train_balanced.fa"
        non_AMP_train.append(str(seq_record.seq))
        
    # sequences for training sets
    train_seq = AMP_train + non_AMP_train    
    # set labels for training sequences
    y_train = torch.from_numpy(np.array([1]*len(AMP_train) + [0]*len(non_AMP_train)))
    
    # shuffle training set
    train = list(zip(train_seq, y_train))
    random.Random(123).shuffle(train)
    train_seq, y_train = zip(*train)
    train_seq = list(train_seq)
    y_train = torch.from_numpy(np.array((y_train)))
    
    # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_train = one_hot_padding(train_seq, MAX_LEN) 

    
    indv_pred_train = [] # list of predicted scores for individual models on the training set
    
    # if test sets specified, process the test data
    if args.amp_te is not None and args.non_amp_te is not None:
        # load test sets
        AMP_test = []
        non_AMP_test = []      
        for seq_record in SeqIO.parse(args.amp_te, 'fasta'):
            # "../data/AMPlify_AMP_test_common.fa"
            AMP_test.append(str(seq_record.seq))
        for seq_record in SeqIO.parse(args.non_amp_te, 'fasta'):
            # "../data/AMPlify_non_AMP_test_balanced.fa"
            non_AMP_test.append(str(seq_record.seq))
        
        # sequences for test sets
        test_seq = AMP_test + non_AMP_test
        # set labels for test sequences
        y_test = torch.from_numpy(np.array([1]*len(AMP_test) + [0]*len(non_AMP_test)))
        # generate one-hot encoding input and pad sequences into MAX_LEN long
        X_test = one_hot_padding(test_seq, MAX_LEN)
        indv_pred_test = [] # list of predicted scores for individual models on the test set       

   
    te_ens=[i for i in range(len(non_AMP_test))]
    model = SNNModel(X_train[te_ens].shape, 64, 2, 32)
    model.eval()

    #loading best model
    model.load_state_dict(torch.load('../Models/Model_weight_{}.h5'.format(0)))

    y_pred = model(X_train[te_ens].float())
    # print(test_seq[-1], y_test[-1])

    for i in range(2):
        df = pd.read_csv('Attention_{}.csv'.format(i))
        df = df.drop(['Unnamed: 0'], axis=1)
        df.insert(0, 'seq', non_AMP_test)
        df.to_csv('Attention_{}.csv'.format(i))
        

        
if __name__ == "__main__":
    main()
    	