#NIE MASZ PATH Z MODELEM W WORK SPACE. NAPISAC SKRYPT KTORY WSZYSTKO TWORZY CO POTRZEBNE

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
tf.compat.v1.enable_v2_behavior()
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# #Tworzy plik z mapowaniem etykiet rzeczy do wykrywania
# def createLabelMap():
#     labels = [{'name': 'door', 'id': 1},
#             {'name': 'step', 'id': 2}, 
#             {'name': 'window', 'id': 3} ]
#     with open(ANNOTATIONS_PATH + '/label_map.pbtxt', 'w') as f:
#         for label in labels:
#             f.write('item{\n')
#             f.write('\tname:\'{}\'\n'.format(label['name']))
#             f.write('\tid:{}\n'.format(label['id']))
#             f.write('}\n')

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
    
    # if(os.path.exists(MODEL_PATH +'/' +CUSTOM_MODEL_NAME +'/pipeline.config')):
    #     print("\nConfig already exists\n")
    #     return()

    os.system('mkdir Tensorflow/workspace/models/{}'.format(CUSTOM_MODEL_NAME))
    os.system('cp {} {}'.format(PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config',
                                MODEL_PATH +'/' +CUSTOM_MODEL_NAME))
    print("\nSuccesfully copied config\n")
    '''

    NIE ROBIC TEGO

    Po tym jak sie skopiuje to w decelowym miejscu:


    Remove fine_tune_checkpoint_version line (line 172 according to what you posted) from the pipeline.config file and try again

    
    '''

def updateModelConfig():
    CONFIG_PATH = MODEL_PATH +'/' +CUSTOM_MODEL_NAME + '/pipeline.config'
    lines = []

    # with open(CONFIG_PATH, 'r') as config:
    #     with open('temp.config', 'w') as new_config:
    #         for line in config:
    #             if "fine_tune_checkpoint_version: V2" not in line.strip('\n'):
    #                 new_config.write(line)          #usun linijke co robi blad

    # os.replace('temp.config', CONFIG_PATH)

    config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(CONFIG_PATH, 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

        pipeline_config.model.ssd.num_classes = 3           #TUTAJ ILE MASZ KLAS
        pipeline_config.train_config.batch_size = 4
        pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH + '/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
        pipeline_config.train_config.fine_tune_checkpoint_type = 'detection'        #tutaj czy chcesz detection
        pipeline_config.train_input_reader.label_map_path = ANNOTATIONS_PATH + '/label_map.pbtxt'
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATIONS_PATH+'/train.record']
        pipeline_config.eval_input_reader[0].label_map_path = ANNOTATIONS_PATH + '/label_map.pbtxt'
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATIONS_PATH+'/test.record']

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(CONFIG_PATH, 'wb') as f:
        f.write(config_text)


def generateTrainCommand():
    print("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=5000""".format(APIMODEL_PATH, MODEL_PATH, CUSTOM_MODEL_NAME, MODEL_PATH, CUSTOM_MODEL_NAME))

if __name__ == "__main__":
    if(not os.path.exists(ANNOTATIONS_PATH + '/label_map.pbtxt')):
        print("\nYou have to create labelmap first! Use createLabelMap.py\n")
        quit()
    if not any(os.scandir(IMAGE_PATH+'/train')):
        print("\nYou have to select train group first. Chceck description in workspace/images\n")
        quit()

    if not any(os.scandir(IMAGE_PATH+'/test')):
        print("\nYou have to select test group first. Chceck description in workspace/images\n")
        quit()


    createTFRecords()
    copyModelConfig()
    updateModelConfig()
    generateTrainCommand()


#fine_tune_checkpoint_version: V2