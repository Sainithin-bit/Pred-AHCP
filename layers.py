import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.utils.rnn as rnn_utils
import pandas as pd

class Attention(nn.Module):
    def __init__(self,
                 activation='relu',
                 input_shape=None,
                 return_attention=False,
                 W_regularizer=None,
                 u_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 u_constraint=None,
                 b_constraint=None,
                 out_channels=8,
                 bias=True):
        super(Attention, self).__init__()

        
        self.W_regularizer = W_regularizer
        self.u_regularizer = u_regularizer
        self.b_regularizer = b_regularizer
        
        self.W_constraint = W_constraint
        self.u_constraint = u_constraint
        self.b_constraint = b_constraint
        
        self.bias = bias
        self.supports_masking = True
        self.return_attention = return_attention

        self.activation=torch.nn.ReLU()

        amount_features = out_channels
        attention_size  = out_channels
        self.W = torch.empty((amount_features, attention_size))
        torch.nn.init.xavier_uniform_(self.W,  gain=nn.init.calculate_gain('relu'))

        if self.bias:
            self.b = torch.zeros(attention_size)

        else:
            self.register_parameter('b', None)

        self.context = torch.empty((attention_size,))
        torch.nn.init.uniform_(self.context)

        

    def forward(self, x, mask=None):       



        ui = x @ self.W             # (b, t, a)
        if self.bias is not None:
            ui += self.b
        ui = self.activation(ui)           # (b, t, a)


        # Z = U * us (eq. 9)
        us = self.context.unsqueeze(0)   # (1, a)
        ui_us = ui @ us.transpose(0, 1)              # (b, t, a) * (a, 1) = (b, t, 1)
        ui_us = ui_us.squeeze(-1)  # (b, t, 1) -> (b, t)

        
        # alpha = softmax(Z) (eq. 9)
        alpha = self._masked_softmax(ui_us, mask) # (b, t)
        alpha = alpha.unsqueeze(-1)     # (b, t, 1)


        if self.return_attention:
            return alpha
        else:
            # v = alpha_i * x_i (eq. 10)
            return torch.sum(x * alpha, dim=1), alpha

    def _masked_softmax(self, logits, mask):

        """PyTorch's default implementation of softmax allows masking through the use of
        `torch.where`. This method handles masking if `mask` is not `None`."""

        b, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - b

        exped = torch.exp(logits)

        # ignoring masked inputs
        if mask is not None:
            mask = mask.float()
            exped *= mask

        partition = torch.sum(exped, dim=-1, keepdim=True)

        # if all timesteps are masked, the partition will be zero. To avoid this
        # issue we use the following trick:
        partition = torch.max(partition, torch.tensor(torch.finfo(logits.dtype).eps))

        return exped / partition



class SNNModel(nn.Module):
    def __init__(self, inp, in_channels = 26 , no_of_head=1, out_chan=16):

        super(SNNModel, self).__init__()

        self.Linear = nn.Linear(27, in_channels)

        self.conv1 = nn.Conv1d(in_channels, out_chan, 1, stride=1)
        self.conv2 = nn.Conv1d(in_channels, out_chan, 3, stride=1)
        self.conv3 = nn.Conv1d(in_channels, out_chan, 5, stride=1)
        self.attention_heads=[Attention(input_shape=inp, out_channels=out_chan) for i in range(no_of_head)]
        self.Dense1=nn.Linear(no_of_head*out_chan, 512)
        self.Output=nn.Linear(512, 1)
        self.sigm=torch.nn.Sigmoid()
        self.activation=nn.LeakyReLU()
        self.dropout=nn.Dropout(p=0.5)
        self.bacthnorm=nn.BatchNorm1d(512)


    def forward(self, x):

        
        x = self.activation(self.Linear(x))
        x = x.transpose(1, 2) #This line is tranforming Batch, Seq_length, Embedding ---> Batch, Embedding, Seq_length


        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        x = torch.cat((x1, x2, x3), dim=2)
        x = x.transpose(1, 2)  #This line is tranforming Batch, Embedding, Seq_length ---> Batch, Seq_length, Embedding
        
        
        x_l=[]
        for ind, fun in enumerate(self.attention_heads):
            x_pred, scores = fun(x) 
            x_l.append(x_pred)

            #The below code is useful for visualizing the attention values each motif is getting
            # scores = torch.squeeze(x_pred)
            # df = pd.DataFrame(scores.detach().numpy())
            # df.to_csv('Attention_{}.csv'.format(ind))

        x=torch.cat(tuple(x_l), dim=1)

        x=self.bacthnorm(self.Dense1(x))
        x=self.activation(x)
        x=self.dropout(x)
        x=self.Output(x)
        out = self.sigm(x)

        return out

	