import cv2
import numpy as np

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
        door_area = door_areas[area_identifer]
        stairs_area = stairs_areas[area_identifer]
        print(window_area)
        display(cap)

def display(cap):
    while(True):
        flag, frame = cap.read()
        if(flag):
            frame = cv2.resize(frame, (int(frame.shape[1] * 0.5) , int(frame.shape[0] * 0.5)), interpolation= cv2.INTER_AREA)

            #klatka z oknem
            frame_window = frame[window_area[0][1]: window_area[1][1]+1, window_area[0][0]: window_area[1][0]+1]

            frame_window_edges = cv2.cvtColor(frame_window, cv2.COLOR_BGR2GRAY)
            frame_window_edges = cv2.Canny(frame_window_edges,150,200)

            white_pixel_window = cv2.countNonZero(frame_window_edges)
            black_pixel_window = (frame_window_edges.shape[0] * frame_window_edges.shape[1]) - white_pixel_window
            percentage_window = str(white_pixel_window/black_pixel_window)

            frame = cv2.putText(frame, percentage_window[0:7], window_area[0], cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,0,0), 1, cv2.LINE_AA)
            cv2.imshow("Window Frame", frame_window_edges)

            #klatka z drzwiami
            frame_door = frame[door_area[0][1]: door_area[1][1]+1, door_area[0][0]: door_area[1][0]+1]
            frame_door_edges = cv2.cvtColor(frame_door, cv2.COLOR_BGR2GRAY)
            frame_door_edges = cv2.Canny(frame_door_edges,150,200)

            white_pixel_door = cv2.countNonZero(frame_door_edges)
            black_pixel_door = (frame_door_edges.shape[0] * frame_door_edges.shape[1]) - white_pixel_door
            percentage_door = str(white_pixel_door/black_pixel_door)

            frame = cv2.putText(frame, percentage_door[0:7], door_area[0], cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,255,0), 1, cv2.LINE_AA)


            cv2.imshow("Door Frame", frame_door_edges)

            #klatka z schodem
            frame_stair = frame[stairs_area[0][1]: stairs_area[1][1]+1, stairs_area[0][0]: stairs_area[1][0]+1]
            frame_stair_edges = cv2.cvtColor(frame_stair, cv2.COLOR_BGR2GRAY)
            frame_stair_edges = cv2.Canny(frame_stair_edges,150,200)


            white_pixel_stair = cv2.countNonZero(frame_stair_edges)
            black_pixel_stair = (frame_stair_edges.shape[0] * frame_stair_edges.shape[1]) - white_pixel_stair
            percentage_stair = str(white_pixel_stair/black_pixel_stair)

            frame = cv2.putText(frame, percentage_stair[0:7], stairs_area[0], cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255), 1, cv2.LINE_AA)


            cv2.imshow("Step Frame", frame_stair_edges)


            #ramka wokol okien
            frame = cv2.rectangle(frame, window_area[0], window_area[1], (255,0,0), 1)

            #ramka wokol drzwi
            frame = cv2.rectangle(frame, door_area[0], door_area[1], (0,255,0), 1)

            #ramka wokol schodow
            frame = cv2.rectangle(frame, stairs_area[0], stairs_area[1], (0,0,255), 1)




            cv2.imshow("Whole Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    selectData()