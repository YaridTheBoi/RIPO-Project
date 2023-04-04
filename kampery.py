import cv2
import numpy as np
import random
import string




window_areas = {"out_front" : [(542,196), (565,227)], "out_back" : [(297,231), (412,369)], "inside" : [(0,0), (0,0)]}
door_areas = {"out_front" : [(477,295), (572,396)], "out_back" : [(459,245), (594,320)], "inside" : [(0,0), (0,0)]}
stairs_areas = {"out_front" : [(543,456), (563,480)], "out_back" : [(639,266), (659,298)], "inside" : [(0,0), (0,0)]}

window_area = [(), ()]
door_area = [(), ()]
stairs_area = [(), ()]


def selectData():
    #filename= input("Podaj nazwe pliku zrodlowego(z rozszerzeniem): ")
    filename = "out_back_3.MOV"
    cap = cv2.VideoCapture("Kampery/" +filename)

    if(not cap.isOpened()):
        print("Nie ma takiego pliku\n")
        cap.release()
        selectData()
    else:
        # area_identifer = filename[:-6]
        # global window_area, door_area, stairs_area
        # window_area= window_areas[area_identifer]
        # door_area = door_areas[area_identifer]
        # stairs_area = stairs_areas[area_identifer]
        # print(window_area)
        #display(cap)
        collectData(cap)

def display(cap):

    while(True):
        flag, frame = cap.read()
        if(flag):
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.5) , int(frame.shape[0] * 0.5)), interpolation= cv2.INTER_AREA)


            cv2.imshow("Whole Frame", frame)
        key = cv2.waitKey(1)
        if key== ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()



def collectData(cap):
        #zamienic tutaj obiekt dla ktorego zbierasz dane
    data_path = "data/door"
    while(True):
        flag, frame = cap.read()
        if(flag):
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.5) , int(frame.shape[0] * 0.5)), interpolation= cv2.INTER_AREA)


            cv2.imshow("Whole Frame", frame)
        key = cv2.waitKey(1)
        if key== ord('q'):
            break
        elif key == ord('f'):
            cv2.imwrite(data_path+ f"/positive/{('p'.join(random.choices(string.ascii_lowercase, k=10)))}.jpg".format(), frame)
        elif key == ord('d'):
            cv2.imwrite(data_path+ f"/negative/{('n'.join(random.choices(string.ascii_lowercase, k=10)))}.jpg".format(), frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    selectData()