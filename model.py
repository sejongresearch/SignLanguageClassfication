import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
       
        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        

    def forward(self, image):
       

        out = self.relu(self.conv1(image))
        out = self.pool1(out)
       

        out = self.relu(self.conv2(out))
        out = self.pool2(out)
        
        out = self.relu(self.conv3a(out))
        out = self.relu(self.conv3b(out))
        out = self.pool3(out)
    
        
        out = self.relu(self.conv4a(out))
        out = self.relu(self.conv4b(out))
        out = self.pool4(out)
       
        out = self.relu(self.conv5a(out))
        out = self.relu(self.conv5b(out))
        
        out = self.pool5(out)

     
        out = out.view(-1, 8192)
        
        out = self.relu(self.fc6(out))

        out = self.dropout(out)

        result = self.relu(self.fc7(out))
     
        return result
        


class LSTM_anno(nn.Module):
  def __init__(self):
    super(LSTM_anno, self).__init__()
    
    self.features = 4096
    self.num_classes = 20
    self.clips = 16
    
   
    
   
    self.lstm = nn.LSTM(input_size=self.features,
                       hidden_size=256,
                       num_layers=1,
                       bias=True,
                       batch_first=False,
                       bidirectional=False)
    
   
    
    #self.FC1bn = nn.InstanceNorm1d(4096)

    self.FC1 = nn.Linear(256,20)

    self.dropout = nn.Dropout(p=0.5)

    self.relu = nn.ReLU()

    self.logsoftmax = nn.LogSoftmax(dim=1)


    self.init_lstm()
  
  def init_lstm(self):

    #LSTM._all_weights

    for ci in self.children():
        
        if isinstance(ci, nn.LSTM):
          
          nn.init.xavier_uniform_(ci._parameters['weight_ih_l0'])
          nn.init.xavier_uniform_(ci._parameters['weight_hh_l0'])
          nn.init.constant_(ci._parameters['bias_hh_l0'], 0.)
          nn.init.constant_(ci._parameters['bias_ih_l0'], 0.)
        elif isinstance(ci, nn.Linear):
          nn.init.xavier_uniform_(ci.weight)
          nn.init.constant_(ci.bias, 0.)
        
    
    
  def forward(self, x):

    
    state = None

    lstm_output, _ = self.lstm(x, state)

    lstm_sfeature = self.FC1(lstm_output[-1])

    final_score = self.logsoftmax(lstm_sfeature)







    #####################################################
    """featout = self.relu(self.FC3(featout))

    featsize = featout.size(0)
    import pdb
    pdb.set_trace()
    result_feat = 0.125*featout[0:int(featsize/4)].item() + 0.75*featout[int(featsize/4):int(3*featsize/4)]+0.125*featout[int(3*featsize/4):]
    
    
    
    
    last_output = lstm_output[-1]

  
    featout =  self.dropout(self.relu(self.FC1(last_output)))

    featout =  self.dropout(self.relu(self.FC2(featout)))

    featout = self.relu(self.FC3(featout))

    featout=((featout - featout.mean())/featout.std())

    #last_outputs = self.softmax(featout)

    last_outputs = self.logsoftmax(featout)"""
   
    #####################################################
    
    
    return final_score



"""
class LSTM_anno(nn.Module):
  def __init__(self):
    super(LSTM_anno, self).__init__()
    
    
    
   
    
   
    self.lstm = BNLSTM(input_size=4096,
                       hidden_size=256
                       )
    

    self.FC1 = nn.Linear(256,20)
    self.dropout = nn.Dropout(p=0.5)
    self.relu = nn.ReLU()
    self.logsoftmax = nn.LogSoftmax(dim=1)

    
    
  def forward(self, x):

    
    state = None

    lstm_output, _ = self.lstm(x, state)

    lstm_sfeature = self.FC1(lstm_output[-1])

    final_score = self.logsoftmax(lstm_sfeature)


    
    return final_score


class BNLSTMCell(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(BNLSTMCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.weight_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
    self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
    self.bias = nn.Parameter(torch.zeros(4 * hidden_size))          
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.weight_ih = nn.Parameter(torch.Tensor(input_size, 4 * hidden_size))
    self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, 4 * hidden_size))
    self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

    self.bn_ih = nn.InstanceNorm1d(4 * self.hidden_size, affine=False)
    self.bn_hh = nn.InstanceNorm1d(4 * self.hidden_size, affine=False)
    self.bn_c = nn.InstanceNorm1d(self.hidden_size)

    self.reset_parameters()
      
  def reset_parameters(self):
    nn.init.orthogonal_(self.weight_ih.data)
    nn.init.orthogonal_(self.weight_hh.data[:, :self.hidden_size])
    nn.init.orthogonal_(self.weight_hh.data[:, self.hidden_size:2 * self.hidden_size])
    nn.init.orthogonal_(self.weight_hh.data[:, 2 * self.hidden_size:3 * self.hidden_size])
    nn.init.eye_(self.weight_hh.data[:, 3 * self.hidden_size:])
    self.weight_hh.data[:, 3 * self.hidden_size:] *= 0.95
          

  def forward(self, input, hx):
    h, c = hx
    ih = torch.matmul(input, self.weight_ih)
    hh = torch.matmul(h, self.weight_hh)
    bn_ih = self.bn_ih(ih)
    bn_hh = self.bn_hh(hh)
    hidden = bn_ih + bn_hh + self.bias

    i, f, o, g = torch.split(hidden, split_size_or_sections=self.hidden_size, dim=1)
    new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    new_h = torch.sigmoid(o) * torch.tanh(self.bn_c(new_c))
    return (new_h, new_c)


class BNLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, batch_first=False, bidirectional=False):
    super(BNLSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.batch_first = batch_first
    self.bidirectional = bidirectional

    self.lstm_f = BNLSTMCell(input_size, hidden_size)
    if bidirectional:
      self.lstm_b = BNLSTMCell(input_size, hidden_size)
    self.h0 = nn.Parameter(torch.Tensor(2 if self.bidirectional else 1, 1, self.hidden_size))
    self.c0 = nn.Parameter(torch.Tensor(2 if self.bidirectional else 1, 1, self.hidden_size))
    nn.init.normal_(self.h0, mean=0, std=0.1)
    nn.init.normal_(self.c0, mean=0, std=0.1)
    
  def forward(self, input, hx=None):
    if not self.batch_first:
      input = input.transpose(0, 1)
    batch_size, seq_len, dim = input.size()
    if hx: init_state = hx
    else: init_state = (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1))
        
    hiddens_f = []
    final_hx_f = None
    hx = (init_state[0][0], init_state[1][0])
    for i in range(seq_len):
      
      hx = self.lstm_f(input[:, i, :], hx)
      hiddens_f.append(hx[0])
      final_hx_f = hx
    hiddens_f = torch.stack(hiddens_f, 1)
        
    if self.bidirectional:
      hiddens_b = []
      final_hx_b = None
      hx = (init_state[0][1], init_state[1][1])
      for i in range(seq_len-1, -1, -1):
        
        hx = self.lstm_b(input[:, i, :], hx)
        hiddens_b.append(hx[0])
        final_hx_b = hx
      hiddens_b.reverse()
      hiddens_b = torch.stack(hiddens_b, 1)
        
    if self.bidirectional:
      hiddens = torch.cat([hiddens_f, hiddens_b], -1)
      hx = (torch.stack([final_hx_f[0], final_hx_b[0]], 0), torch.stack([final_hx_f[1], final_hx_b[1]], 0))
    else:
      hiddens = hiddens_f
      hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(1))
    if not self.batch_first:
      hiddens = hiddens.transpose(0, 1)
    return hiddens, hx
"""