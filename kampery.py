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

import cv2
import numpy as np
import random
import string
import time
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_utils
from object_detection.builders import model_builder


import tensorflow as tf
from object_detection.utils import config_util




#zaladowanie modelu i zbuildowanie go wg configu
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs["model"], is_training=False)

#zaladowanie ostatniego checkpointa (najnowszy stan wiedzy modelu)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH , 'ckpt-6')).expect_partial()



@tf.function
def detect_fn(image):
    image,shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


window_areas = {"out_front" : [(542,196), (565,227)], "out_back" : [(297,231), (412,369)], "inside" : [(0,0), (0,0)]}
door_areas = {"out_front" : [(477,295), (572,396)], "out_back" : [(459,245), (594,320)], "inside" : [(0,0), (0,0)]}
stairs_areas = {"out_front" : [(543,456), (563,480)], "out_back" : [(639,266), (659,298)], "inside" : [(0,0), (0,0)]}

window_area = [(), ()]
door_area = [(), ()]
stairs_area = [(), ()]


def selectData():
    #filename= input("Podaj nazwe pliku zrodlowego(z rozszerzeniem): ")
    filename = "out_front_1.MOV"
    cap = cv2.VideoCapture("Kampery/" +filename)

    if(not cap.isOpened()):
        print("Nie ma takiego pliku\n")
        cap.release()
        selectData()
    else:
        area_identifer = filename[:-6]
        global window_area, door_area, stairs_area
        window_area= window_areas[area_identifer]
        #door_area = door_areas[area_identifer]
        door_area = [(468,77),(593,500)]
        stairs_area = stairs_areas[area_identifer]
        # print(window_area)
        display(cap)
        #collectData(cap)


def display(cap):
    category_index = label_map_util.create_category_index_from_labelmap(ANNOTATIONS_PATH + '/label_map.pbtxt')
    start = time.time()
    
    max_detection_delay = 0.2
    min_detection_delay = 0.02
    detection_delay = (max_detection_delay + min_detection_delay)/2
    delay_delta = 0.02
    prev_best_score = 0.25
    while(True):
        flag, frame = cap.read()

        frame = cv2.resize(frame, (int(frame.shape[1] * 0.5) , int(frame.shape[0] * 0.5)), interpolation= cv2.INTER_AREA)
        


        current_time = time.time()

        if current_time - start >= detection_delay:

            
            input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype =tf.float32)
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))

            detections = {key:value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            label_id_offset =1

            prev_best_score= max(detections['detection_scores'])

            vis_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw= 3,
                min_score_thresh=0.3,
                agnostic_mode=False
            )

            if prev_best_score > 0.27 and detection_delay - delay_delta > min_detection_delay:
                detection_delay -= delay_delta
            
            if prev_best_score < 0.27 and detection_delay + delay_delta <=max_detection_delay:
                detection_delay += delay_delta
            
            start = current_time

            print(detection_delay)



        cv2.imshow("Detections", frame)
        #cv2.imshow("Whole Frame", frame)
        key = cv2.waitKey(1)
        if key== ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()



def collectData(cap):
    data_path = "data"
    while(True):
        flag, frame = cap.read()
        if(flag):
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.5) , int(frame.shape[0] * 0.5)), interpolation= cv2.INTER_AREA)


            cv2.imshow("Whole Frame", frame)
        key = cv2.waitKey(1)
        if key== ord('q'):
            break
        elif key == ord('d'):
            cv2.imwrite(data_path+ f"/door/door_{(''.join(random.choices(string.ascii_lowercase, k=10)))}.jpg".format(), frame)
        elif key == ord('s'):
            cv2.imwrite(data_path+ f"/step/step_{(''.join(random.choices(string.ascii_lowercase, k=10)))}.jpg".format(), frame)
        elif key == ord('w'):
            cv2.imwrite(data_path+ f"/window/window_{(''.join(random.choices(string.ascii_lowercase, k=10)))}.jpg".format(), frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    selectData()

    #2.12