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







    
    
    
    return final_score



