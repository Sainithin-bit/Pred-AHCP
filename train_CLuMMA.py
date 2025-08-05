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
from sklearn.model_selection import train_test_split
import datetime




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
    g1_p , g2_p, g3_p = ['G', 'A', 'S', 'D', 'T'], ['C', 'P', 'N', 'V', 'E', 'Q', 'I', 'L'], ['K', 'M', 'H', 'F', 'R', 'Y', 'W']
    g1_v, g2_v, g3_v = ['G', 'A', 'S', 'T', 'P', 'D'], ['N', 'V', 'E', 'Q', 'I', 'L'], ['M', 'H', 'K', 'F', 'R', 'Y', 'W']
    H,S, C =['E','A', 'L', 'M', 'Q', 'K', 'R', 'H'], ['V','I','Y','C','W','F','T'], ['G', 'N', 'P', 'S', 'D']

    
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

        # if aa[i] in H:
        #     one_hot[aa[i]][26]=1
        # elif aa[i] in S:
        #     one_hot[aa[i]][27]=1
        # elif aa[i] in C:
        #     one_hot[aa[i]][28]=1

        # if aa[i] in g1_p:
        #     one_hot[aa[i]][29]=1
        # elif aa[i] in g2_p:
        #     one_hot[aa[i]][30]=1
        # elif aa[i] in g3_p:
        #     one_hot[aa[i]][31]=1
        
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
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
        # "../data/CLuMMA_AMP_train_common.fa"
        AMP_train.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(args.non_amp_tr, 'fasta'):
        # "../data/CLuMMA_non_AMP_train_balanced.fa"
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
            # "../data/CLuMMA_AMP_test_common.fa"
            AMP_test.append(str(seq_record.seq))
        for seq_record in SeqIO.parse(args.non_amp_te, 'fasta'):
            # "../data/CLuMMA_non_AMP_test_balanced.fa"
            non_AMP_test.append(str(seq_record.seq))
        
        # sequences for test sets
        test_seq = AMP_test + non_AMP_test
        # set labels for test sequences
        y_test = torch.from_numpy(np.array([1]*len(AMP_test) + [0]*len(non_AMP_test)))
        # generate one-hot encoding input and pad sequences into MAX_LEN long
        X_test = one_hot_padding(test_seq, MAX_LEN)
        indv_pred_test = [] # list of predicted scores for individual models on the test set       

    X_train, X_test, y_train,  y_test = train_test_split(torch.concat((X_train, X_test), dim=0), torch.concat((y_train, y_test), dim=0), test_size=0.2, random_state=42)



    ensemble_number = 5 # number of training subsets for ensemble
    ensemble = StratifiedKFold(n_splits=ensemble_number, shuffle=True, random_state=50)
    save_file_num = 0
    in_channels = 64
    heads=2
    out_channels=32

    log_file = open('logs/Log_{}_{}_{}_{}'.format(in_channels, heads, out_channels, datetime.datetime.now().time()), 'w')

    for i, (tr_ens, te_ens) in enumerate(ensemble.split(X_train, y_train)):

        print(len(tr_ens), len(te_ens))
        print(X_train[tr_ens].shape)

        best_loss=float('inf')
        model = SNNModel(X_train[tr_ens].shape, in_channels, heads, out_channels)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-07)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                               patience=10,
                                               verbose=True)
        
        valid_loss=[]
        total = np.concatenate((tr_ens, te_ens))
        for epoch in range(45):

            optimizer.zero_grad()


            y_pred = model(X_train[total].float())

            loss = torch.nn.functional.binary_cross_entropy(torch.squeeze(y_pred), y_train[total].float())

            loss.backward()
            optimizer.step()

            valid_pred = model(X_train[te_ens].float())
            valid_loss.append(torch.nn.functional.binary_cross_entropy(torch.squeeze(valid_pred), y_train[te_ens].float()).data.item())

            print('Epoch : {}  Training loss :{}, Training accu : {},  Validation acc: {}'.format(epoch, loss.data.item(), 
                                                                                               accuracy_score(y_train[total],predict_by_class(torch.squeeze(y_pred))), 
                                                                                               accuracy_score(y_train[te_ens], predict_by_class(torch.squeeze(valid_pred)))))
            log_file.write('Epoch : {}  Training loss :{}, Training accu : {},  Validation acc: {}'.format(epoch, loss.data.item(), 
                                                                                               accuracy_score(y_train[total],predict_by_class(torch.squeeze(y_pred))), 
                                                                                               accuracy_score(y_train[te_ens], predict_by_class(torch.squeeze(valid_pred)))))
            log_file.write('\n')
            

            if valid_loss[-1]< best_loss:

                print("Saving Best model params....")
                #saving model weights 
                torch.save(model.state_dict(), '../Models/Model_weight_{}.h5'.format(i))

                best_loss=valid_loss[-1]


            scheduler.step(valid_loss[-1])



        #predciting the entire training set with best model
        temp_pred_train = torch.squeeze(model(X_train.float())).detach().numpy() # predicted scores on the [whole] training set from the current model
        indv_pred_train.append(temp_pred_train)




        # training and validation accuracy for the current model
        temp_pred_class_train_curr = predict_by_class(model(X_train[total].float()).flatten())
        temp_pred_class_val = predict_by_class(model(X_train[te_ens].float()).flatten())

        print('*************************** current model ***************************')
        print('current train acc: ', accuracy_score(y_train[total], temp_pred_class_train_curr))
        print('current val acc: ', accuracy_score(y_train[te_ens], temp_pred_class_val)) 
        
        # if test sets specified, output the test performance for the current model
        if args.amp_te is not None and args.non_amp_te is not None:
            temp_pred_test = torch.squeeze(model(X_test.float())).detach().numpy()# predicted scores on the test set from the current model
            indv_pred_test.append(temp_pred_test)
            temp_pred_class_test = predict_by_class(temp_pred_test)
            tn_indv, fp_indv, fn_indv, tp_indv = confusion_matrix(y_test, temp_pred_class_test).ravel()
            #print(confusion_matrix(y_test, temp_pred_class_test))
            print('test acc: ', accuracy_score(y_test, temp_pred_class_test))  
            print('test sens: ', tp_indv/(tp_indv+fn_indv))
            print('test spec: ', tn_indv/(tn_indv+fp_indv))
            print('test f1: ', f1_score(y_test, temp_pred_class_test))
            print('test roc_auc: ', roc_auc_score(y_test, temp_pred_test))
            
        print('*********************************************************************')

        log_file.write('*************************** current model ***************************')
        log_file.write('\n')
        log_file.write('current train acc: {}'.format(accuracy_score(y_train[total], temp_pred_class_train_curr)))
        log_file.write('\n')
        log_file.write('current val acc: {} '.format(accuracy_score(y_train[te_ens], temp_pred_class_val)))
        log_file.write('\n')
        
        # if test sets specified, output the test performance for the current model
        if args.amp_te is not None and args.non_amp_te is not None:
            temp_pred_test = torch.squeeze(model(X_test.float())).detach().numpy()# predicted scores on the test set from the current model
            indv_pred_test.append(temp_pred_test)
            temp_pred_class_test = predict_by_class(temp_pred_test)
            tn_indv, fp_indv, fn_indv, tp_indv = confusion_matrix(y_test, temp_pred_class_test).ravel()
            #print(confusion_matrix(y_test, temp_pred_class_test))
            log_file.write('test acc: {}'.format(accuracy_score(y_test, temp_pred_class_test)))
            log_file.write('\n')
            log_file.write('test sens: {}'.format(tp_indv/(tp_indv+fn_indv)))
            log_file.write('\n')
            log_file.write('test spec: {}'.format(tn_indv/(tn_indv+fp_indv)))
            log_file.write('\n')
            log_file.write('test f1: {}'.format(f1_score(y_test, temp_pred_class_test)))
            log_file.write('\n')
            log_file.write('test roc_auc: {}'.format(roc_auc_score(y_test, temp_pred_test)))
            log_file.write('\n')
        log_file.write('*********************************************************************')
        log_file.write('\n')


    # if test sets specified, output the test performance for the ensemble model
    if args.amp_te is not None and args.non_amp_te is not None:
        y_pred_prob_test = np.mean(np.array(indv_pred_test), axis=0) # prediction for the test set after ensemble
        y_pred_class_test = predict_by_class(y_pred_prob_test)
                
        y_pred_prob_train = np.mean(np.array(indv_pred_train), axis=0) # prediction for the training set after ensemble
        y_pred_class_train = predict_by_class(y_pred_prob_train)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class_test).ravel()
        print('**************************** final model ****************************')
        print('overall train acc: ', accuracy_score(y_train, y_pred_class_train))
        #print(confusion_matrix(y_train, y_pred_class_train))
        print('overall test acc: ', accuracy_score(y_test, y_pred_class_test))
        print(confusion_matrix(y_test, y_pred_class_test))
        print('overall test sens: ', tp/(tp+fn))
        print('overall test spec: ', tn/(tn+fp))
        print('overall test f1: ', f1_score(y_test, y_pred_class_test))
        print('overall test roc_auc: ', roc_auc_score(y_test, y_pred_prob_test))
        print('*********************************************************************')
        log_file.write('**************************** final model ****************************')
        log_file.write('\n')
        log_file.write('overall train acc: {}'.format(accuracy_score(y_train, y_pred_class_train)))
        log_file.write('\n')
        #print(confusion_matrix(y_train, y_pred_class_train))
        log_file.write('overall test acc: {}'.format(accuracy_score(y_test, y_pred_class_test)))
        log_file.write('\n')
        log_file.write('{}'.format(confusion_matrix(y_test, y_pred_class_test)))
        log_file.write('\n')
        log_file.write('overall test sens: {}'.format(tp/(tp+fn)))
        log_file.write('\n')
        log_file.write('overall test spec: {}'.format(tn/(tn+fp)))
        log_file.write('\n')
        log_file.write('overall test f1: {}'.format(f1_score(y_test, y_pred_class_test)))
        log_file.write('\n')
        log_file.write('overall test roc_auc: {}'.format(roc_auc_score(y_test, y_pred_prob_test)))
        log_file.write('\n')
        log_file.write('*********************************************************************')
        log_file.write('\n')
        
if __name__ == "__main__":
    main()
    