##Developing a toolkit for prototyping machine learning-empowered products: The design and evaluation of ML-Rapid.
Rapid-ML is a toolkit for designers.  
In Workspace.py, you can use several function.  

For example, in Text to speech(TTS):  
TTS.train() Train the default neural networks with initial dataset  
TTS.collect() Organize the marked data based on the suitable format  
TTS.reframe() Reframe the existing data set with collected data  
TTS.retrain("collect") Retrain the neural networks with collected dataset  
TTS.retrain("reframe") Retrain the neural networks with reframed dataset  
TTS.infer() Make inference based on the trained model and new data  
  
There are six models in this toolkit, the functions of collect, reframe, train and retrain are same, but the "infer" is not same.  
The function of "infer":   
Facial based emotion recognition, recognize the emotion of the human in the picture.  
Facial based Identity recognition, recognize the human's identity in the picture.  
Image generation, generate pictures by the test pictures.  
Object Recognition Based on Image, recognize the kind of object in the picture.  
Semantic recognition of speech, recognize the words of an audio.  
Text to speech, generate the audio of words.  
