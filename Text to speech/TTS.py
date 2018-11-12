#!/usr/bin/env pyth2
#-*- coding: utf-8 -*-
"""

"""

from synthesize import synthesize
import os 
import pandas as pd
import csv

path = "./former_dataset/private/voice/LJSpeech-1.0"
recorder_path = "./recorder/temp"
collected_path = './collected_dataset'
reframed_path='./reframed_dataset'

class sound:
    def __init__(self,path):
        self.path=path
    def play(self):
        os.system("play "+self.path)
        
def infer(text):
    text = ' '+text
    pathh = 'Examples.csv'
    if not os.path.exists(pathh):
        df=pd.DataFrame(columns=["text"])
        df.to_csv ("Examples.csv", encoding = "utf-8") 
    else:
        f = open(pathh,'rb')
        df = pd.read_csv(f,index_col=0, encoding = "utf-8")
        df = pd.DataFrame(df)
    row_num = df.iloc[:,0].size
    df.loc[row_num] = [text]
    df.to_csv ("./Examples.csv", encoding = "utf-8")
    f = open("harvard_sentences.txt","w")
    f.write("1."+text)
    f.close()
    synthesize(row_num)
    print("Done")

'''
    if os.path.exists("./samples/%d.wav" % (row_num+1)):
          rnt  = ("./samples/%d.wav" % (row_num+1)) 
          errorFile = open(os.path.join(path,'transcript.csv'), 'a+') 
          writeCSV = csv.writer(errorFile)
          row = []
          con = ("RUN_%.5d" % (row_num+1))+'|'+text
          row.append(con)
          writeCSV.writerow(row)
          os.system('cp '+rnt+' '+path+'/wavs/'+("RUN_%.5d.wav" % (row_num+1)))
          return sound(rnt)
    else: print("No this file!")
'''

def collect():  
     if not os.path.exists(recorder_path):
         print('No such path')
         return 
     if not os.path.exists(collected_path):
         os.mkdir(collected_path)
         os.mkdir(os.path.join(collected_path,'wav'))
         #df=pd.DataFrame(columns=["Text"])
         #df=pd.DataFrame()
         #df.to_csv (os.path.join(collected_path,'transcript.csv'), encoding = "utf-8") 
         os.mknod(collected_path+'/'+"transcript.csv")

        
     a = open(os.path.join(collected_path,'transcript.csv'), "r")
     row_num  = len(a.readlines())
     a.close()
     
     errorFile = open(os.path.join(collected_path,'transcript.csv'), 'a+') 
     writeCSV = csv.writer(errorFile)
     
     allfiles = os.listdir(recorder_path)
     for fi in allfiles:       
        if fi.endswith(".wav"):
            meaning  = fi.replace('_',' ')
            row = []
            name = ('MY_%.5d' % (row_num+1))
            row.append(name) 
            #print(name)
            row.append('|')
            row.append(meaning[:-4])
            row.append('|')
            row.append(meaning[:-4])
            writeCSV.writerow(row)
            cmd = ''
            cmd +=( 'cp '+recorder_path+'/'+fi+' '+collected_path+'/wav'+
                      ('/%s.wav' % name))
            os.system(cmd)
            row_num+=1
     
def reframe():  
     if not os.path.exists(recorder_path):
         print('No such path')
         return 
     if not os.path.exists(reframed_path):
         os.mkdir(reframed_path)
         #os.mkdir(os.path.join(reframed_path,'wav'))
         #df=pd.DataFrame(columns=["Text"])
         #df=pd.DataFrame()
         #df.to_csv (os.path.join(collected_path,'transcript.csv'), encoding = "utf-8") 
         #os.mknod(collected_path+'/'+"transcript.csv")
         cmd=''
         cmd+=('cp -R '+path+'/* '+reframed_path+'/')
         os.system(cmd)

  
     a = open(os.path.join(reframed_path,'transcript.csv'), "r")
     row_num  = len(a.readlines())
     a.close()
     errorFile = open(os.path.join(reframed_path,'transcript.csv'), 'a+') 
     writeCSV = csv.writer(errorFile)
     
     allfiles = os.listdir(recorder_path)
     for fi in allfiles:       
        if fi.endswith(".wav"):
            meaning  = fi.replace('_',' ')
            row = []
            name = ('MY_%.5d' % (row_num+1))
            row.append(name) 
            #print(name)
            row.append('|')
            row.append(meaning[:-4])
            row.append('|')
            row.append(meaning[:-4])
            writeCSV.writerow(row)
            cmd = ''
            cmd +=( 'cp '+recorder_path+'/'+fi+' '+reframed_path+'/wavs'+
                      ('/%s.wav' % name))
            os.system(cmd)
            row_num+=1
            #print("done"+str(row_num))
        
'''        
def reframe():
     
    a = open(os.path.join(path,'transcript.csv'), "r")
    row_num  = (a.readlines())
    print(len(row_num))
    a.close()
    
    
    errorFile = open(os.path.join(path,'transcript.csv'), 'a+') 
    writeCSV = csv.writer(errorFile)
    
    if not os.path.exists(os.path.join(collected_path,'transcript.csv')):
        print('no collected!')
    else:       
        a = open(os.path.join(collected_path,'transcript.csv'), "r")
        print(a.readline())
        content = a.readlines()
        tot  = len(content)     
        for i in range(tot):
            line = content[i]
            tmp = line.split(',')
            file_name = tmp[0]+'.wav'
            meaning = tmp[1][:-1]
            print(meaning)
            cp_path = os.path.join(path,'wavs')
            if not os.path.exists(os.path.join(cp_path,file_name)):
                row = []
                row.append(tmp[0]+'|'+meaning+'|'+meaning)
                writeCSV.writerow(row) 
                os.system('cp '+collected_path+'/wav/'+file_name+' '+cp_path+'/'+file_name)
'''    

     
def train():
    os.system("python3 train.py")

def retrain(str):
    if str=="collect":
        os.system("python3 retrain.py")
    elif str=="reframe":
        os.system("python3 retrain2.py")
    else:
        print("Wrong parameter")

