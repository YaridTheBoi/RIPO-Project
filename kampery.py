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
import sys
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


# wykrywanie obiektu
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

# wybor pliku
def selectData(mode):
    filename= input("Podaj nazwe pliku zrodlowego(z rozszerzeniem): ")
    #filename = "out_front_2.MOV"
    cap = cv2.VideoCapture("Kampery/" +filename)

    global is_back

    is_back = "back" in filename


    if(not cap.isOpened()):
        print("Nie ma takiego pliku\n")
        cap.release()
        selectData(mode)
    else:
        if mode == 1:
            display(cap)
        elif mode == 2:
            collectData(cap)

# analiza filmiku i detekcja
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

        frame = cv2.resize(frame, (frame.shape[1] //2 , frame.shape[0] //2), interpolation= cv2.INTER_AREA)
        

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
            print(prev_best_score)
            vis_utils.visualize_boxes_and_labels_on_image_array(
                frame,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw= 3,
                min_score_thresh=0.2,
                agnostic_mode=False
            )

            if prev_best_score > 0.27 and detection_delay - delay_delta > min_detection_delay:
                detection_delay -= delay_delta
            
            if prev_best_score < 0.27 and detection_delay + delay_delta <=max_detection_delay:
                detection_delay += delay_delta
            
            start = current_time

            #print(detection_delay)



        cv2.imshow("Detections", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    startApp()


# pobieranie klatek do nauki
def collectData(cap):
    data_path = "data"

    print("""
S - aby pobrac probke do folderu ze schodkiem
W - aby pobrac probke do folderu z oknami
D - aby pobrac probke do foldreu z drzwiami
    """)

    while(True):
        flag, frame = cap.read()
        if(flag):
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.5) , int(frame.shape[0] * 0.5)), interpolation= cv2.INTER_AREA)


            cv2.imshow("Whole Frame", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('d'):
            if not is_back:
                cv2.imwrite(data_path+ f"/door/door_{(''.join(random.choices(string.ascii_lowercase, k=10)))}.jpg".format(), frame)
            else:
                cv2.imwrite(data_path+ f"/door_back/door_back_{(''.join(random.choices(string.ascii_lowercase, k=10)))}.jpg".format(), frame)
        elif key == ord('s'):
            if not is_back:
                cv2.imwrite(data_path+ f"/step/step_{(''.join(random.choices(string.ascii_lowercase, k=10)))}.jpg".format(), frame)
            else:
                cv2.imwrite(data_path+ f"/step_back/step_back_{(''.join(random.choices(string.ascii_lowercase, k=10)))}.jpg".format(), frame)
        elif key == ord('w'):
            if not is_back:
                cv2.imwrite(data_path+ f"/window/window_{(''.join(random.choices(string.ascii_lowercase, k=10)))}.jpg".format(), frame)
            else:
                cv2.imwrite(data_path+ f"/window_back/window_back_{(''.join(random.choices(string.ascii_lowercase, k=10)))}.jpg".format(), frame)
    cap.release()
    cv2.destroyAllWindows()
    startApp()

#uruchamianie aplikacji
def startApp():
    mode = ""
    mode = input("""
Wybierz co chcesz zrobic:
1. Wykrywaj obiekty
2. Zbieraj dane
3. Wyjdz
""")
    while (int(mode) > 3 or int(mode) < 1):
        mode = input("""
Nie ma takiej opcji
1. Wykrywaj obiekty
2. Zbieraj dane
3. Wyjdz
""")
    
    if(int(mode) == 3):
        quit()

    selectData(int(mode))


if __name__ == "__main__":
    startApp()

