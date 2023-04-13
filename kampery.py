import cv2
import numpy as np
import random
import string


cascade_door = cv2.CascadeClassifier("data/door/cascade.xml")

window_areas = {"out_front" : [(542,196), (565,227)], "out_back" : [(297,231), (412,369)], "inside" : [(0,0), (0,0)]}
door_areas = {"out_front" : [(477,295), (572,396)], "out_back" : [(459,245), (594,320)], "inside" : [(0,0), (0,0)]}
stairs_areas = {"out_front" : [(543,456), (563,480)], "out_back" : [(639,266), (659,298)], "inside" : [(0,0), (0,0)]}

window_area = [(), ()]
door_area = [(), ()]
stairs_area = [(), ()]


def selectData():
    #filename= input("Podaj nazwe pliku zrodlowego(z rozszerzeniem): ")
    filename = "out_front_3.MOV"
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

    while(True):
        flag, frame = cap.read()
        if(flag):
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.5) , int(frame.shape[0] * 0.5)), interpolation= cv2.INTER_AREA)
            frame_door = frame[door_area[0][1]-10: door_area[1][1]+10, door_area[0][0]-10: door_area[1][0]+10]

            door_rectangles = cascade_door.detectMultiScale(frame_door)


            for i in door_rectangles:
                frame_door = cv2.rectangle(frame_door,(i[0], i[1]), (i[2], i[3]), (255,0,0), 1)


            cv2.imshow("Whole Frame", frame)
            cv2.imshow("Door Frame", frame_door)
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



'''
zaznaczanie danych:
opencv_annotation --anotations <nazwa>.txt --images <path to images>

generowanie wektora:
opencv_createsamples -info pos.txt -w <szerokosc_ob> -h <wysokosc_ob> -num <ile_masz_poz_probek(moze byc wiecej niz masz)> -vec <nazwa_pliku(pos)>.vec


trenowanie modelu:
opencv_traincascade -data cascade_door/ -vec pos.vec -bg bg.txt -w <szerokosc_ob> -h <wysokosc_ob> -numPos <ilosc_poz> -numNeg <ilosc_neg> -minHitRate <minimalne_trafianie> -maxFalseAlarmRate <maksymalne_false_alarm> -numStages <ilosc_rundek_nauki> -precalcValBufSize <ram> -precalcIdxBufSize <ram>





#za duzo falszywych alarmow: +neg probki +stages

#za duzo misow: -stages

#narazie najlepsze(tylko front):
opencv_traincascade -data cascade_door/ -vec pos_front.vec -bg bg.txt -w 24 -h 24 -numPos 300 -numNeg 1000 -minHitRate 0.999 -maxFalseAlarmRate 0.1 -numStages 6 -precalcValBufSize 6000 -precalcIdxBufSize 6000



'''