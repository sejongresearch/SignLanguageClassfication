import os

#text_path = './txt/traintext_16frame.txt'
text_path = './txt/testtext_16frame.txt'
text = open(text_path,'wt',newline='\n')

frames = 16
overlap = 8

base_path = './trainframedata'
file_list = os.listdir(base_path)
file_list.sort()

data = []

for classPath in file_list:
    vidNum_list = os.listdir(base_path+'/'+classPath)
    vidNum_list.sort()
    for vid in vidNum_list:
        frame_list = os.listdir(base_path+'/'+classPath + '/' + vid)
        frame_list.sort()

        ind = len(frame_list)

        i=0
        k=0
        
        all_frame = (int((ind-16)/8)+1)*frames

        for _ in range(all_frame):
            #import pdb
            #pdb.set_trace()
            data.append(base_path + '/' + classPath +'/'+ vid+ '/' + frame_list[i])
            i+=1
            k+=1
            if(k%frames==0):
                data.append('\n')
                i = i - 8
                k = 0
            

text.write('\n'.join(data))

text.close()       

print('Finish!')