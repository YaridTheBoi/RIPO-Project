WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATIONS_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'                         
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config' 
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

import os

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

#Tworzy plik z mapowaniem etykiet rzeczy do wykrywania
def createLabelMap():
    labels = [{'name': 'door', 'id': 1},{'name': 'step', 'id': 2}, {'name': 'window', 'id': 3} ]
    with open(ANNOTATIONS_PATH + '/label_map.pbtxt', 'w') as f:
        for label in labels:
            f.write('item{\n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

#Tworzy TFRecords potrzebne do nauki 
#(z tym sa takie fikoly. generalnie trzeba doinstalowac paczki i zmienic kod w paru plikach. google dobrze na to odpowiada jak bedzie czas to opisze proces)
def createTFRecords():
    os.system('python {} -x {} -l {} -o {}'.format(SCRIPTS_PATH+'/generate_tfrecord.py',    #odpal skrypt od TF
                                                    IMAGE_PATH+'/train',                     #przekaz mu dane treningowe
                                                    ANNOTATIONS_PATH+'/label_map.pbtxt',     #powiedz gdzie ma mapy etykiet
                                                    ANNOTATIONS_PATH + '/train.record'))     #tutaj daj wynik


    os.system('python {} -x {} -l {} -o {}'.format(SCRIPTS_PATH+'/generate_tfrecord.py',    #odpal skrypt od TF
                                                    IMAGE_PATH+'/test',                     #przekaz mu dane testowe
                                                    ANNOTATIONS_PATH+'/label_map.pbtxt',     #powiedz gdzie ma mapy etykiet
                                                    ANNOTATIONS_PATH + '/test.record'))     #tutaj daj wynik
    


#Pobieranie pretrained modelu
'''
w /Tensorflow skopiuj to repo: https://github.com/tensorflow/models (nie wiem jeszcze czy przejdzie na moim pushu wiec daje info co zrobic)
'''


#Kopiowanie configa modelu
'''
Przekopiuj pipline.config z /RIPO-Project/Tensorflow/workspace/pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
                        do /RIPO-Project/Tensorflow/workspace/models/my_ssd_mobnet
'''
def copyModelConfig():
    
    os.system('mkdir Tensorflow/workspace/models/{}'.format(CUSTOM_MODEL_NAME))
    os.system('cp {} {}'.format(PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config',
                                MODEL_PATH +'/' +CUSTOM_MODEL_NAME))
    

def updateModelConfig():
    pass

if __name__ == "__main__":
    #createLabelMap()
    createTFRecords()
    #copyModelConfig()
    