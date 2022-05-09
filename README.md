##File Overview

`eyekart.py` - The primary file that runs the calibration sequence and opens the Unity karting game

`evaluation.py` - Runs the calibration sequence and then shows a few known points to look at, calculating the error between the prediction and final points

`mousecontrol.py` - Runs the calibration sequence and uses predicted gaze to control the mouse

`testtraining.py` - Runs training on the last set of calibration data gathered, then shows loss of the model 
* Uses `calibration_points` and `eye_data`

`shape_predictor_68_face_landmarks.dat` - Provides the shape predictor for facial landmark recognition

`EyeKart` - The standalone Unity app, built for MacOS


##How to Run

I am using Python 3.7, and in order to run the python code you need to have all of the imported packages installed. The Unity program is also built for MacOS, so you must have MacOS installed. I don't believe it is version specific, but I have Catalina installed in case the game is throwing errors.