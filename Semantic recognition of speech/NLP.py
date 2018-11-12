# -*- coding: utf-8 -*-

############################### Train part below 

import os 
import label_wav
import shutil

cwd = os.getcwd()      

def train():
    os.system("'python3 train.py ")
    print('Finish training.')
    print('Start creating frozen model ...')
    ## 删掉旧的模型
    os.remove(cwd+'/graph/my_frozen_graph.pb')
    ## 创建新的模型 （根据新的数据集）
    os.system("'python3 freeze.py ")
    print('Done.')

    
#train()


############################### Infer part below

                 
nlp_path = cwd


def infer(wav):  
    ## wav should be a name in string form that is under recorder 
    ## 将需要识别结果的 wav 文件放到 recorder 里面
    ## 要求 必须是 无压缩 16000Hz 的单英文单词的 1s音频
    ## e.g. wav = '0b40aa8e_nohash_0.wav'  
    labels = nlp_path + '/speech_commands_train/conv_labels.txt'   #xxx改变label生成位置
    graph = nlp_path + '/graph/my_frozen_graph.pb'   #freeze里面改变graph生成位置
    input_name = 'wav_data:0'
    output_name = 'labels_softmax:0'
    how_many_labels = 1
    
    wav = nlp_path + "/recorder/" + wav
    
    return label_wav.label_wav(wav, labels, graph, input_name, output_name, how_many_labels)

#wav = '0ab3b47d_nohash_0.wav'  #改放data的位置
#infer(wav)

############################### Collected part here

recorder_path = cwd+"/recorder/"    ##  用来存放测试或者更新数据集的 wav ##
collected_path = cwd +'/collected_dataset/'   ## 用来存放已经infer出结果并且改好名称的 wav ##

def collected_rename(wav):
    ## 将 recorder里 未知的语音 wav 识别出结果后 将格式（名称） 修改为与原数据集相同格式 new_name.wav
    print('Inferring and renaming ...')
    infer_result = infer(wav)
    new_name = infer_result+'_' + wav
    wav = nlp_path + "/recorder/" + wav
    new_name = nlp_path + "/recorder/" + new_name
    shutil.move(wav,new_name)
    
#wav = '0b40aa8e_nohash_0.wav'
#collected_rename(wav)

def collected_move():
    ## 将已经全部是同样格式的 新数据 new_name.wav 从recorder转移到 collected_dataset里面
    print('Start transferring ...')
    if not os.path.exists(recorder_path):
         print('No such path')
         return 
    if not os.path.exists(collected_path):
         os.mkdir(collected_path)

    allfiles = os.listdir(recorder_path) 
    for fi in allfiles:
        if fi.endswith(".wav"):
            meaning = fi.split('_')[0]
			
            if not os.path.exists(collected_path + '/' +meaning ):
                os.mkdir(collected_path+'/'+meaning)
                
            shutil.move(recorder_path + '/' + fi,collected_path + '/' + meaning)
    print('Done.')             
#collected_move()

################################# Reframe part here

def reframe():
    reframed_path = cwd +'/reframed_dataset/'
    if not os.path.exists(reframed_path):
        os.mkdir(reframed_path)
        shutil.copytree(cwd +'/'+ 'speech_dataset', reframed_path)   ##最一开始的 speech_dataset 即原始自带的 dataset
         
         
    ## 将 原先的 reframed_dataset 备份到 former_dataset里面
    shutil.rmtree(cwd +'/'+ 'former_dataset')
    shutil.copytree(reframed_path,cwd +'/'+ 'former_dataset')
    
    ## 将 collected_dataset 已经收集整理好的数据 复制到 reframed_dataset里面
     
    allfiles = os.listdir(collected_path) 
    for wanted_words in allfiles:
        #if os.path.isdir(fi):  
        # wanted_words = fi
        if wanted_words not in os.listdir(reframed_path):
            shutil.copytree(collected_path +'/'+ wanted_words, reframed_path+'/'+wanted_words)
            
        else:	
            cp_path = os.path.join(reframed_path, wanted_words)
            for file in os.listdir(collected_path+ '/' + wanted_words):
                if file not in os.listdir(cp_path):
                   shutil.copyfile(collected_path+ '/' + wanted_words , cp_path)     	
#reframe()	
def collect(wav):
    collected_rename(wav)
    collected_move()
def retrain(str):
    if str=="collect":
        os.system("python3 train2.py")
    elif str=="reframe":
        os.system("python3 train3.py")
    else:
        print("Wrong parameter")


    
    
    
    
    
    
    
    
    
    
