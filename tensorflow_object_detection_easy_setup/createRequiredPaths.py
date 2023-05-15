import os
import requests
import subprocess
import shutil

BASE = 'Tensorflow'
WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATIONS_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'          
TRAIN_PATH = IMAGE_PATH + '/train'               
TEST_PATH = IMAGE_PATH + '/test'      
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'


paths= [
BASE,
WORKSPACE_PATH,
SCRIPTS_PATH ,
ANNOTATIONS_PATH,
IMAGE_PATH ,                   
MODEL_PATH,
PRETRAINED_MODEL_PATH,
CHECKPOINT_PATH,
TRAIN_PATH,
TEST_PATH
]
current_dir = os.getcwd()


# Create necessary folders
for path in paths:
    if not os.path.exists(path):
        os.mkdir(path)




# Copy generate_tfrecord.py to desired folder
github_url = "https://raw.githubusercontent.com/YaridTheBoi/tensorflow_generate_tfrecord/main/generate_tfrecord.py"
response = requests.get(github_url)
with open(SCRIPTS_PATH+'/generate_tfrecord.py', "w") as f:
    f.write(response.text)
print("Copied generate_tfrecord.py file ")

# Install object detection api https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

#remove other venvs
if os.path.exists('venv'):
    shutil.rmtree('venv')
    print("Removed previous venv")

# Create new venv and install tensorflow
venv_path = os.path.join(current_dir, 'venv', 'bin', 'activate')
os.system("python3.8 -m venv venv")
os.system('venv/bin/python -m pip install tensorflow')
os.system('venv/bin/python -m pip install cython')

# Copy models from tensorflow git
if not os.path.exists(APIMODEL_PATH):
    os.system('cd {} && git clone https://github.com/tensorflow/models'.format(BASE))



os.system('cd {} && protoc object_detection/protos/*.proto --python_out=.'.format(APIMODEL_PATH+'/research'))
os.system('cd {} && git clone https://github.com/cocodataset/cocoapi.git && cd cocoapi/PythonAPI && make && cp -r pycocotools {}'.format(APIMODEL_PATH+'/research', APIMODEL_PATH+'/research'))
os.system('cd {} && cp object_detection/packages/tf2/setup.py . && python -m pip install .'.format(APIMODEL_PATH+'/research', APIMODEL_PATH+'/research'))

# Default pretrained model http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
# If you want to use different one go to https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
# Remember to change name of folder in trainer.py
if not os.path.exists('Tensorflow/workspace/pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'):
    print("Downloading pretrianed model")
    os.system('cd {} && wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz && tar -xvzf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'.format(PRETRAINED_MODEL_PATH))
    print("Downloaded pretrianed model")


# Create description in folders
#models
github_url = "https://raw.githubusercontent.com/YaridTheBoi/tensorflow_easy_setup_files/main/models_description.txt"
response = requests.get(github_url)
with open(APIMODEL_PATH+'/models_description.txt', "w") as f:
    f.write(response.text)

#scripts
github_url = "https://raw.githubusercontent.com/YaridTheBoi/tensorflow_easy_setup_files/main/scripts_description.txt"
response = requests.get(github_url)
with open(SCRIPTS_PATH+'/scripts_description.txt.txt', "w") as f:
    f.write(response.text)

#workspace/annotations
github_url = "https://raw.githubusercontent.com/YaridTheBoi/tensorflow_easy_setup_files/main/workspace_annotation_description.txt"
response = requests.get(github_url)
with open(ANNOTATIONS_PATH+'/workspace_annotation_description.txt', "w") as f:
    f.write(response.text)


#workspace/images
github_url = "https://raw.githubusercontent.com/YaridTheBoi/tensorflow_easy_setup_files/main/workspace_images_description.txt"
response = requests.get(github_url)
with open(IMAGE_PATH+'/workspace_images_description.txt', "w") as f:
    f.write(response.text)

#workspace/images/test
github_url = "https://raw.githubusercontent.com/YaridTheBoi/tensorflow_easy_setup_files/main/workspace_image_test_description.txt"
response = requests.get(github_url)
with open(TEST_PATH+'/workspace_image_test_description.txt', "w") as f:
    f.write(response.text)

#workspace/images/train
github_url = "https://raw.githubusercontent.com/YaridTheBoi/tensorflow_easy_setup_files/main/workspace_image_train_description.txt"
response = requests.get(github_url)
with open(TRAIN_PATH+'/workspace_image_train_description.txt', "w") as f:
    f.write(response.text)


#workspace/models/my_model
github_url = "https://raw.githubusercontent.com/YaridTheBoi/tensorflow_easy_setup_files/main/workspace_models_my_ssd_mobnet_description.txt"
response = requests.get(github_url)
with open(CHECKPOINT_PATH+'workspace_models_my_ssd_mobnet_description.txt', "w") as f:
    f.write(response.text)



print("Created Descriptions")