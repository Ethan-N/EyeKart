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
evaluated = False
average_dist = 0
model = []
circle_coors = [width/2, height/2]
instructions_given = False
mouse = pynput.mouse.Controller()
post_training = datetime.now()
game_start = False
escaped = False

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
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        landmarks = predictor(gray, face)
        #Left Eye
        left_outer_point = (landmarks.part(36).x, landmarks.part(36).y)
        left_inner_point = (landmarks.part(39).x, landmarks.part(39).y)
        left_center_top = midpoint(landmarks.part(37), landmarks.part(38))
        left_center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        #Blink Detection
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
        right_hor_length = hypot((right_outer_point[0] - right_inner_point[0]), (right_outer_point[1] - right_inner_point[1]))
        right_vert_length = hypot((right_center_top[0] - right_center_bottom[0]), (right_center_top[1] - right_center_bottom[1]))

        right_closed_ratio = 0
        if right_vert_length != 0:
            right_closed_ratio = right_hor_length / right_vert_length

        eye2 = cv2.equalizeHist(gray[right_center_top[1]-boundary:right_center_bottom[1]+boundary,
               right_inner_point[0]-boundary:right_outer_point[0]+boundary])

        #Closed Eye Detection (Tweak values before using)
        #if right_closed_ratio > 5.5 and left_closed_ratio > 5.5:
        #    DO SOMETHING

        eye_regions = np.zeros((eye_width*2, eye_height))
        for x in range(0, eye_width):
            for y in range(0, eye_height):
                eye_regions[x][y] = cv2.mean(eye1[int(x*eye1.shape[0]/eye_width):int((x+1)*eye1.shape[0]/eye_width),
                                             int(y*eye1.shape[1]/eye_height):int((y+1)*eye1.shape[1]/eye_height)])[0]
                #cv2.rectangle(frame, (int(left_outer_point[0]+y*eye1.shape[1]/eye_width*2), int(left_center_top[1]+x*eye1.shape[0]/eye_width*2)),
                #              (int(left_outer_point[0]+(y+1)*eye1.shape[1]/eye_height*2), int(left_center_top[1]+(x+1)*eye1.shape[0]/eye_height*2)),
                #              (eye_regions[x][y], eye_regions[x][y], eye_regions[x][y]), -1)

        for x in range(0, eye_width):
            for y in range(0, eye_height):
                eye_regions[x+eye_width][y] = cv2.mean(eye2[int(x*eye2.shape[0]/eye_width):int((x+1)*eye2.shape[0]/eye_width),
                             int(y*eye2.shape[1]/eye_height):int((y+1)*eye2.shape[1]/eye_height)])[0]
                #cv2.rectangle(frame, (int(right_inner_point[0]+y*eye2.shape[1]/eye_width*2), int(right_center_top[1]+x*eye2.shape[0]/eye_width*2)),
                #              (int(right_inner_point[0]+(y+1)*eye2.shape[1]/eye_height*2), int(right_center_top[1]+(x+1)*eye2.shape[0]/eye_height*2)),
                #              (eye_regions[x+eye_width][y], eye_regions[x+eye_width][y], eye_regions[x+eye_width][y]), -1)

    diff = (datetime.now() - start_time).seconds
    microsecs = (datetime.now() - start_time).microseconds
    instruction_time = 5
    duration = 5

    y_spacing = 4.3

    if diff < instruction_time:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        cv2.putText(capture, "CALIBRATION", (int(width/2)-610, 200), font, 5, (255, 0, 0))
        cv2.putText(capture, "Please look at the center of each dot as they appear", (int(width/2)-550, 300), font, 1, (255, 0, 0))
        cv2.putText(capture, "The first calibration point will be in the top left", (int(width/2)-500, 380), font, 1, (255, 0, 0))
        cv2.putText(capture, "Try not to move your head while using the app", (int(width/2)-500, 460), font, 1, (255, 0, 0))


        cv2.imshow("EyeTrack", capture)

    elif diff < instruction_time:
        cv2.rectangle(capture, (int(width/2-250), int(height/2-100)), (int(width/2+50),int(height/2-20)), (255,255,255), 5)
        cv2.imshow("EyeTrack", capture)

    elif diff < 1 + instruction_time:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50), int(height/50)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration + instruction_time + 1:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= instruction_time +  + 1
        x = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width/1.16*x), int(height/50)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration + instruction_time  + 2:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50), int(height/50+height/y_spacing)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 2 + instruction_time + 2:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= instruction_time + 2
        x = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width/1.16*x), int(height/50+height/y_spacing)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 2 + instruction_time + 3:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50), int(height/50+height/y_spacing*2)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 3 + instruction_time + 3:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= instruction_time + 3
        x = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width/1.16*x), int(height/50+height/y_spacing*2)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 3 + instruction_time + 4:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50), int(height/50+height/y_spacing*3)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 4 + instruction_time + 4:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= instruction_time + 4
        x = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width/1.16*x), int(height/50+height/y_spacing*3)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 4 + instruction_time + 5:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50), int(height/50)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 5 + instruction_time + 5:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= instruction_time + 5
        y = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50), int(height/50+height/1.3*y)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 5 + instruction_time + 6:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50+width*2/4.7), int(height/50)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 6 + instruction_time + 6:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= instruction_time + 6
        y = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width*2/4.7), int(height/50+height/1.3*y)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 6 + instruction_time + 7:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        center = [int(width/50+width*4/4.7), int(height/50)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        cv2.imshow("EyeTrack", capture)

    elif diff < duration * 7 + instruction_time + 7:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        diff -= instruction_time + 7
        y = ((diff + microsecs/1e6) / duration) % 1
        center = [int(width/50+width*4/4.7), int(height/50+height/1.3*y)]
        cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
        cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
        cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)

        eye_data.append(eye_regions)
        calibration_points.append((center[0],center[1]))

        cv2.imshow("EyeTrack", capture)


    elif diff >= duration * 7 + instruction_time + 7 and diff < duration * 7 + instruction_time + 8:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        cv2.putText(capture, "TRAINING", (int(width/2)-450, 200), font, 5, (255, 0, 0))
        cv2.putText(capture, "Please wait a moment as the system learns your gaze", (int(width/2)-550, 300), font, 1, (255, 0, 0))
        cv2.putText(capture, "Next, the system will show you more points", (int(width/2)-460, 380), font, 1, (255, 0, 0))
        cv2.putText(capture, "Look at each point as they pop up in order to evaluate accuracy", (int(width/2)-620, 460), font, 1, (255, 0, 0))
        cv2.imshow("EyeTrack", capture)

    elif diff > duration * 7 + instruction_time + 8 and not trained:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        cv2.putText(capture, "TRAINING", (int(width/2)-450, 200), font, 5, (255, 0, 0))
        cv2.putText(capture, "Please wait a moment as the system learns your gaze", (int(width/2)-550, 300), font, 1, (255, 0, 0))
        cv2.putText(capture, "Next, the system will show you more points", (int(width/2)-460, 380), font, 1, (255, 0, 0))
        cv2.putText(capture, "Look at each point as they pop up in order to evaluate accuracy", (int(width/2)-620, 460), font, 1, (255, 0, 0))
        cv2.imshow("EyeTrack", capture)

        training = True

    if training:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        cv2.putText(capture, "TRAINING", (int(width/2)-450, 200), font, 5, (255, 0, 0))
        cv2.putText(capture, "Please wait a moment as the system learns your gaze", (int(width/2)-550, 300), font, 1, (255, 0, 0))
        cv2.putText(capture, "Next, the system will show you more points", (int(width/2)-460, 380), font, 1, (255, 0, 0))
        cv2.putText(capture, "Look at each point as they pop up in order to evaluate accuracy", (int(width/2)-620, 460), font, 1, (255, 0, 0))
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
        post_training = datetime.now()
        capturing = False
        data_point = [0, 0]

    if trained and not evaluated:
        diff = (datetime.now() - post_training).seconds
        microsecs = (datetime.now() - start_time).microseconds
        duration = 2

        if diff < duration * 9:
            x = int(diff / duration) % 3
            y = int(diff / (duration * 3))
            #Fully spread points on screen
            center = [int(width/15+width/3.0*x), int(height/15+height/3.4*y)]
            cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
            cv2.circle(capture, (center[0], center[1]), 20, (0, 0, 255), -1)
            cv2.line(capture, (center[0]-5, center[1]-5), (center[0]+5, center[1]+5), (255, 255, 255), 2)
            cv2.line(capture, (center[0]+5, center[1]-5), (center[0]-5, center[1]+5), (255, 255, 255), 2)
            cv2.imshow("EyeTrack", capture)
            if diff % duration == 0 and microsecs > 500000:
                capturing = True
                total_captures = 0

            if capturing:
                values = model.predict(np.reshape(np.array(eye_regions), (1, 480)))
                circle_coors = [circle_coors[0]*.3+values[0][0]*.7, circle_coors[1]*.3+values[0][1]*.7]
                data_point[0] += circle_coors[0]
                data_point[1] += circle_coors[1]
                total_captures += 1

            if diff % duration == 1 and microsecs > 800000 and capturing:
                capturing = False
                data_point[0] = data_point[0] / total_captures
                data_point[1] = data_point[1] / total_captures
                print("Compared Data: ")
                print("Prediction: ", data_point)
                print("Actual: ", center)

                distance = ((data_point[0] - center[0])**2 + (data_point[1] - center[1])**2)**0.5
                print("Distance: ", distance)
                print("")
                average_dist += distance / 9

                data_point = [0, 0]
        else:
            evaluated = True

    if evaluated:
        cv2.rectangle(capture, (0,0), (width, height), (255,255,255), -1)
        cv2.putText(capture, "EVALUATION", (int(width/2)-600, 200), font, 5, (255, 0, 0))
        cv2.putText(capture, "The system's average distance error is %.2f" % average_dist, (int(width/2)-530, 300), font, 1, (255, 0, 0))

        values = model.predict(np.reshape(np.array(eye_regions), (1, 480)))
        circle_coors = [circle_coors[0]*.3+values[0][0]*.7, circle_coors[1]*.3+values[0][1]*.7]
        cv2.circle(capture, (int(circle_coors[0]), int(circle_coors[1])), 20, (0, 255, 255), -1)

        cv2.imshow("EyeTrack", capture)


    key = cv2.waitKey(1)
    if key == 27:
        escaped = True

cap.release()
cv2.destroyAllWindows()