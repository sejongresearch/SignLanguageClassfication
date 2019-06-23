# *C3D_LSTM_SignLanguage_Pytorch*
This is term project of Sejong 2019 AI<br> 
[by JunHee.Park](https://github.com/joooooonie) / 
[by KyuWon.Lee](https://github.com/KYUUUW) /
[by MiYeon.Lee](https://github.com/iammiori) /
[by Won.Jo](https://github.com/Jo-won) /
[by DaeChan.Han](https://github.com/big-chan)

![image](https://user-images.githubusercontent.com/46413594/59977686-65511800-960f-11e9-98c2-47c73d41a460.png)

- # Our submit
  - [인공지능_ppt](https://github.com/Jo-won/Me-Mo/files/3317990/_final_.pptx)
  - [인공준희.pdf](https://github.com/Jo-won/Me-Mo/files/3317994/default.pdf)
  

- # Summary

![image](https://user-images.githubusercontent.com/46413594/59973642-6a956f00-95dd-11e9-80a6-cb659f663980.png)


- # Dataset

  - Dataset은 20개의 class, class 당 40개의 video, video당 110 ~ 200 frames (약 3 ~ 7초, 1초당 30frame)
  
    0.What do you want<br>
    1.i love you<br>
    2.your are welcome<br>
    3.nice to meet you<br>
    4.time is up<br>
    5.i don't understand<br>
    6.see you later<br>
    7.im sorry<br>
    8.no thank you<br>
    9.how old ar you<br>
    10.are you hurt<br>
    11.how are you<br>
    12.im fine<br>
    13.what happened<br>
    14.i have not eaten yet<br>
    15.do you need help<br>
    16.how much does it cost<br>
    17.hello<br>
    18.i miss you<br>
    19.what's your name<br>
  
  - Dataset은 임의로 train 600개 (75%), test 200개 (25%) 로 나눴습니다.
  
  - 16 frame씩 앞뒤로 8 frame씩 overlap 시킨 것을 한 clip으로 하고 한 video 를 여러 clip으로 나누어 clip 을 하나씩 model에 넣었습니다.
  
  - 분류의 성능을 높이기 위한 양 손에 대한 Local정보를 Openpose를 이용해 뽑아냈습니다. 그리고 이를 Local영역을 추가했을 때와 안했을 때를 비교하여 실험 하였습니다.
  
  
- # Model
  
  - Model은 C3D(conv3d)와 LSTM으로 이루어진 model을 사용했습니다.
  
  - C3D model을 사용한 이유 : <br>분류하려는 것이 image가 아닌 video인데 conv2d로만 이루어진 model에서 feature를 뽑아내게 되면 spatial한 정보는 얻을 수 있지만 temporal한 정보는 얻을 우 없어서 temporal한 정보 또한 얻을 수 있는 C3D를 선택했습니다.
  
  - LSTM model을 사용한 이유 : <br> C3D만을 사용했을 시 영상에서 수화를 하는 중간에만 잘 맞추고 대부분의 부분에서는 잘 맞추지 못하느 경향이 있습니다. 즉, Short-term에 대한 영상정보는 잘 인식하는 부분도 있지만 Long-term 에 대해서는 정보가 전달이 되지 못하는 것 같아서 LSTM을 쓰게 되었습니다.<br><br><br>
  아래 video가 C3D만을 사용했을 때이고 오른쪽 위 부분이 해당 class라고 예측하고 정확도를 나타내는 부분입니다.
  그리고 이 video의 class는 "Do you need help?" 입니다.<br>
![gitgif](https://user-images.githubusercontent.com/46413594/59976843-db03b680-9604-11e9-9962-d1646ff92895.gif)

  
  - C3D 부분의 parameter들은 C3D저자가 제공해준 pretrained parameter에 저희 Dataset으로 추가적인 pretrain을 한 parameter를 사용했습니다.
  
  - C3D에서 나온 4096 dimension의 feature를 LSTM에 넣고 256 dimension으로 줄인 후 한개의 FC Layer를 거쳐 20개의 dimension, 즉 class의 갯수만큼의 dimension으로 줄였습니다.
  
  - LSTM과 FC layer를 거쳐 지나온 feature에 대한 Activation function으로 Softmax와 Logsoftmax를 비교하여 실험해 성능이 더 잘나온 Logsoftmax를 사용하였습니다.
  
  - Loss function은 CrossEntropy 와 MSELoss를 비교하여 실험한 결과 더 성능이 잘 나온 CrossEntropy를 사용했습니다.
  
  - Learning rate 는 1e-3으로 initialize 하였고 10 epoch 마다 10분의 1로 줄어들 수 있도록 LR scheduler를 사용했습니다.
  
  
- # Performance
  
  1. Only_global + Logsoftmax + CrossEntropy : 손에 대한 Local dataset을 추가하지 않고 LSTM FC layer 후에 activation function을 Logsoftmax로 쓰고 Loss function을 CrossEntropy로 쓴 그래프 입니다.
  
  2. Global_local + Logsoftmax + CrossEntropy : 1번에서 Local dataset만 추가한 그래프 입니다.
  
  3. Only_global + Logsoftmax + MSE : 1번에서 loss function을 CrossEntropy 대신 MSE를 사용한 그래프 입니다.
  
  4. Only_global + Softmax + CrossEntropy : 1번에 Logsoftmax대신 Softmax를 사용한 그래프 입니다.
  
  ![image](https://user-images.githubusercontent.com/46413594/59975099-2317de80-95ef-11e9-90dd-f530d8080f62.png)
  _가로축 = percent , 세로축 = epoch_
  
  Local 정보를 포함한 2번이 아닌 1번이 가장 좋은 성능을 보였는데 이는 Local정보를 video에서 손이 움직이는 위치가 아닌 단지 hand detecting을 하여 손의 모양의 변화만 뽑아내어 model에 넣었기 때문에 1번보다 성능이 덜 나왔습니다.
  
  아래사진은 frame 수에 따른 분류 성공/실패 그래프입니다.
  ![image](https://user-images.githubusercontent.com/46413594/59975742-60339f00-95f6-11e9-8d01-58be2bf21eca.png)
  _세로축에서 성공했을 땐 1, 실패 했을 땐 0, 가로축은 frame의 수 입니다._
  
  
- # Code

  - version
    
    python 3.7.1<br>
    cuda version 10.0<br>
    torch version 1.0.1<br>
    torchvision version 0.2.2<br>
  
  - Practice
    
    [main.py](https://github.com/Jo-won/C3D_LSTM_SignLanguage_Pytorch/blob/master/main.py) <- train과 test를 하는 code<br>
    [model.py](https://github.com/Jo-won/C3D_LSTM_SignLanguage_Pytorch/blob/master/model.py) <- model<br>
    [make_txt.py](https://github.com/Jo-won/C3D_LSTM_SignLanguage_Pytorch/blob/master/make_txt.py) <- 16frames 8overlap 된 1clip을 만들기 위해 참고<br>
    [trainindex.txt](https://github.com/Jo-won/C3D_LSTM_SignLanguage_Pytorch/blob/master/trainindex.txt)  [testindex.txt](https://github.com/Jo-won/C3D_LSTM_SignLanguage_Pytorch/blob/master/testindex.txt) <- main.py안에서 임의로 나눈 train과 test에 대해 제공 (1~800)<br>
    [txtpath.txt ](https://github.com/Jo-won/C3D_LSTM_SignLanguage_Pytorch/blob/master/testindex.txt) [csvpath.txt]()<- video의 실제 경로<br>
    [utils.py](https://github.com/Jo-won/C3D_LSTM_SignLanguage_Pytorch/blob/master/utils.py) <- pytorch의 utils function<br>
    [dcdatasets.py](https://github.com/Jo-won/C3D_LSTM_SignLanguage_Pytorch/blob/master/dcdatasets.py) <-local 경로에서 저장된 video를 불러와 clip으로 만들어주는 코드<br>
    
    
    
  
