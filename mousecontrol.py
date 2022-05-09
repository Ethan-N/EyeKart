#Eye detection credit to https://pysource.com/2019/01/07/eye-detection-gaze-controlled-keyboard-with-python-and-opencv-p-1/
#Blinking credit to https://pysource.com/2019/01/10/eye-blinking-detection-gaze-controlled-keyboard-with-python-and-opencv-p-2

import cv2
import numpy as np
import dlib
from math import hypot
from datetime import datetime
from keras.layers import Input, Dense
from keras.models import Model
import pickle
import pynput
import socket
import pyautogui
import os
import sys

width, height = pyautogui.size()

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.QT_FONT_NORMAL
start_time = datetime.now()

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

cv2.namedWindow("EyeTrack", cv2.WINDOW_NORMAL)

eye_width = 20
eye_height = 12
data_point = np.zeros((eye_width*2, eye_height))
total_captures = 0
capturing = True
eye_data = []
calibration_points = []
training = False
trained = False
model = []
circle_coors = [width/2, height/2]
instructions_given = False
mouse = pynput.mouse.Controller()
game_start = False
escaped = False
count = 0

ip = "127.0.0.1"
port = 11000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while not escaped:
    cv2.resizeWindow('EyeTrack', width, height)
    _, capture = cap.read()
    capture = cv2.flip(capture, 1)
    gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    boundary = 5

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        #Draws rectangle around detected face
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        landmarks = predictor(gray, face)

        #Left Eye
        left_outer_point = (landmarks.part(36).x, landmarks.part(36).y)
        left_inner_point = (landmarks.part(39).x, landmarks.part(39).y)
        left_center_top = midpoint(landmarks.part(37), landmarks.part(38))
        left_center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        #Closed Eye Calculations
        left_hor_length = hypot((left_outer_point[0] - left_inner_point[0]), (left_outer_point[1] - left_inner_point[1]))
        left_vert_length = hypot((left_center_top[0] - left_center_bottom[0]), (left_center_top[1] - left_center_bottom[1]))

        left_closed_ratio = 0
        if left_vert_length != 0:
            left_closed_ratio = left_hor_length / left_vert_length

        eye1 = cv2.equalizeHist(gray[left_center_top[1]-boundary:left_center_bottom[1]+boundary,
               left_outer_point[0]-boundary:left_inner_point[0]+boundary])

        #Right Eye
        right_outer_point = (landmarks.part(45).x, landmarks.part(45).y)
        right_inner_point = (landmarks.part(42).x, landmarks.part(42).y)
        right_center_top = midpoint(landmarks.part(43), landmarks.part(44))
        right_center_bottom = midpoint(landmarks.part(46), landmarks.part(47))

        #Closed Eye Calculations
        right_hor_length = hypot((right_outer_point[0] - right_inner_point[0]), (right_outer_point[1] - right_inner_point[1]))
        right_vert_length = hypot((right_center_top[0] - right_center_bottom[0]), (right_center_top[1] - right_center_bottom[1]))

        right_closed_ratio = 0
        if right_vert_length != 0:
            right_closed_ratio = right_hor_length / right_vert_length

        eye2 = cv2.equalizeHist(gray[right_center_top[1]-boundary:right_center_bottom[1]+boundary,
               right_inner_point[0]-boundary:right_outer_point[0]+boundary])

        #Closed Eye Detection (May need to tweak values)
        #if right_closed_ratio > 5.5 and left_closed_ratio > 5.5:
        #   DO SOMETHING

        eye_regions = np.zeros((eye_width*2, eye_height))
        for x in range(0, eye_width):
            for y in range(0, eye_height):
                eye_regions[x][y] = cv2.mean(eye1[int(x*eye1.shape[0]/eye_width):int((x+1)*eye1.shape[0]/eye_width),
                                             int(y*eye1.shape[1]/eye_height):int((y+1)*eye1.shape[1]/eye_height)])[0]
                #Outline eye on screen
                #cv2.rectangle(frame, (int(left_outer_point[0]+y*eye1.shape[1]/eye_width*2), int(left_center_top[1]+x*eye1.shape[0]/eye_width*2)),
                #              (int(left_outer_point[0]+(y+1)*eye1.shape[1]/eye_height*2), int(left_center_top[1]+(x+1)*eye1.shape[0]/eye_height*2)),
                #              (eye_regions[x][y], eye_regions[x][y], eye_regions[x][y]), -1)

        for x in range(0, eye_width):
            for y in range(0, eye_height):
                eye_regions[x+eye_width][y] = cv2.mean(eye2[int(x*eye2.shape[0]/eye_width):int((x+1)*eye2.shape[0]/eye_width),
                             int(y*eye2.shape[1]/eye_height):int((y+1)*eye2.shape[1]/eye_height)])[0]
                #Outline eye on screen
                #cv2.rectangle(frame, (int(right_inner_point[0]+y*eye2.shape[1]/eye_width*2), int(right_center_top[1]+x*eye2.shape[0]/eye_width*2)),
                #              (int(right_inner_point[0]+(y+1)*eye2.shape[1]/eye_height*2), int(right_center_top[1]+(x+1)*eye2.shape[0]/eye_height*2)),
                #              (eye_regions[x+eye_width][y], eye_regions[x+eye_width][y], eye_regions[x+eye_width][y]), -1)

    diff = (datetime.now() - start_time).seconds
    microsecs = (datetime.now() - start_time).microseconds
    duration = 5

    x_spacing = 1.16
    y_spacing = 4.3

    if count == 0:
        start_time = datetime.now()

        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        cv2.putText(capture, "CALIBRATION", (int(width/2)-610, 200), font, 5, (255, 0, 0))
        cv2.putText(capture, "Please look at the center of each dot as they appear", (int(width/2)-550, 300), font, 1, (255, 0, 0))
        cv2.putText(capture, "The first calibration point will be in the top left", (int(width/2)-500, 380), font, 1, (255, 0, 0))
        cv2.putText(capture, "Try not to move your head while using the app", (int(width/2)-500, 460), font, 1, (255, 0, 0))
        cv2.putText(capture, "Press Enter to begin calibration process", (int(width/2)-450, 540), font, 1, (255, 0, 0))
        cv2.putText(capture, "Press Escape at any time to quit the app", (int(width/2)-450, 620), font, 1, (255, 0, 0))

        cv2.imshow("EyeTrack", capture)

    elif diff < 1:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50), int(height/50)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration + 1:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= 1
        x = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width/x_spacing*x), int(height/50)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration + 2:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50), int(height/50+height/y_spacing)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 2 + 2:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= 2
        x = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width/x_spacing*x), int(height/50+height/y_spacing)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 2 + 3:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50), int(height/50+height/y_spacing*2)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 3 + 3:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= 3
        x = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width/x_spacing*x), int(height/50+height/y_spacing*2)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 3 + 4:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50), int(height/50+height/y_spacing*3)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 4 + 4:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= 4
        x = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width/x_spacing*x), int(height/50+height/y_spacing*3)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 4 + 5:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50), int(height/50)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 5 + 5:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= 5
        y = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50), int(height/50+height/1.3*y)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 5 + 6:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50+width*2/4.7), int(height/50)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 6 + 6:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= 6
        y = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width*2/4.7), int(height/50+height/1.3*y)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 6 + 7:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50+width*4/4.7), int(height/50)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 7 + 7:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= 7
        y = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width*4/4.7), int(height/50+height/1.3*y)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)


    elif diff >= duration * 7 + 7 and diff < duration * 7 + 8:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        cv2.putText(capture, "TRAINING", (int(width/2)-450, 200), font, 5, (255, 0, 0))
        cv2.putText(capture, "Please wait a moment as the system learns your gaze", (int(width/2)-550, 300), font, 1, (255, 0, 0))
        cv2.putText(capture, "After, you will be able to control the mouse with your eyes", (int(width/2)-640, 460), font, 1, (255, 0, 0))
        cv2.imshow("EyeTrack", capture)

    elif diff > duration * 7 + 8 and not trained:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        cv2.putText(capture, "TRAINING", (int(width/2)-450, 200), font, 5, (255, 0, 0))
        cv2.putText(capture, "Please wait a moment as the system learns your gaze", (int(width/2)-550, 300), font, 1, (255, 0, 0))
        cv2.putText(capture, "After, you will be able to control the mouse with your eyes", (int(width/2)-640, 460), font, 1, (255, 0, 0))
        cv2.imshow("EyeTrack", capture)

        training = True

    if training:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        cv2.putText(capture, "TRAINING", (int(width/2)-450, 200), font, 5, (255, 0, 0))
        cv2.putText(capture, "Please wait a moment as the system learns your gaze", (int(width/2)-550, 300), font, 1, (255, 0, 0))
        cv2.putText(capture, "After, you will be able to control the mouse with your eyes", (int(width/2)-640, 460), font, 1, (255, 0, 0))
        cv2.imshow("EyeTrack", capture)

        #Pickles data for offline training if needed
        with open("eye_data", "wb") as ed:
            pickle.dump(eye_data, ed)
        with open("calibration_points", "wb") as cp:
            pickle.dump(calibration_points, cp)

        eye_data = np.reshape(np.array(eye_data), (len(eye_data), 480))
        calibration_points = np.array(calibration_points)

        #Linear Model
        def regressor_model():
            input_x = Input(shape = (480,))
            x = Dense(40, activation = 'relu')(input_x)
            x = Dense(40, activation = 'relu')(x)
            x = Dense(40, activation = 'relu')(x)
            x = Dense(40, activation = 'relu')(x)
            output = Dense(2)(x)
            model = Model(inputs = input_x, outputs = output)
            model.compile(loss = 'mean_squared_error', optimizer = 'adam')

            return model

        model = regressor_model()
        log_train = model.fit(eye_data, calibration_points, epochs = 1000,  verbose = 1)

        training = False
        trained = True

    if trained:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)

        values = model.predict(np.reshape(np.array(eye_regions), (1, 480)))
        circle_coors = [circle_coors[0]*.3+values[0][0]*.7, circle_coors[1]*.3+values[0][1]*.7]
        mouse.position = (int(circle_coors[0]), int(circle_coors[1]))

        msg = str(int(circle_coors[0])) + ' ' + str(int(circle_coors[1]))
        sock.sendto(msg.encode(), ("127.0.0.1",11000))
        cv2.imshow("EyeTrack", capture)



    key = cv2.waitKey(1)
    if key == 13:
        count += 1
    if key == 27:
        escaped = True

cap.release()
cv2.destroyAllWindows()