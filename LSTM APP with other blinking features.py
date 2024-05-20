import tkinter as tk
from tkinter import filedialog
import cv2
import os
import numpy as np
import pandas as pd
import torch
from LSTM_Pressure_RGB_model import LSTMModel
from Accessory_func import histogram_grayworld_whitebalance, large_small_diff, new_smallROI, lstm_input_prep, RollingBuffer

# loading the pressure RGB neural network model
input_size = 3
hidden_size = 128
num_layers = 3
output_size = 1
Pressure_RGB_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
#current_dir = os.getcwd()
current_file_directory = os.path.dirname(os.path.abspath(__file__))

##### Loading LSTM Model Pretrained Weights #####
state_dict = torch.load(f'{current_file_directory}/LSTM_P_RGB_e200.pt')
Pressure_RGB_model.load_state_dict(state_dict)
Pressure_RGB_model.eval()

# points = [] # storing the vertices of the ROI (global)
# LSTM_data = torch.tensor(np.zeros(shape = (9, 3)), dtype = torch.float32)

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi")])
    points = []
    LSTM_data = torch.tensor(np.zeros(shape = (9, 3)), dtype = torch.float32)
    if file_path:
        video_cap = cv2.VideoCapture(file_path)
        # store blink (bool) information for blink tracking (e.g. blinking rate per minute, blink interval)
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        # set the number of element to store in the blink_rate_storage to the number of frames per minute
        # then use the number of boolean values (np.sum(bool_blink)/len(bool_blink)) to determine blink rate
        blink_time_store = RollingBuffer(int(fps*60)) # storing the time stamp of each frame
        blink_status_store = RollingBuffer(int(fps*60)) # storing the status (boolean open/close) of each frame
        ret, first_frame = video_cap.read()
        if ret:
            large_roi = cv2.selectROI(first_frame)
            x, y, w, h = large_roi
            cv2.rectangle(first_frame, (int(large_roi[0]), int(large_roi[1])),
                    (int(large_roi[0] + large_roi[2]), int(large_roi[1] + large_roi[3])),
                    (0, 255, 0), 2)

            # Initialize the KCF tracker
            tracker = cv2.TrackerKCF_create()
            tracker.init(first_frame, (x, y, w, h))

            ROIselect_frame = first_frame.copy()
            def select_point(event, x, y, flags, param):
                #global points
                if event == cv2.EVENT_LBUTTONDOWN:
                    cv2.circle(ROIselect_frame, (x, y), 2, (0, 0, 255), -1)
                    cv2.imshow('Select ROI on lens', ROIselect_frame)
                    points.append([x, y])
            cv2.namedWindow('Select ROI on lens')
            cv2.setMouseCallback('Select ROI on lens', select_point)
            while True:
                cv2.imshow('Select ROI on lens', ROIselect_frame)
                if cv2.waitKey(0) & 0xFF == 13:  # return key
                    break
            pts = np.array(points, np.int32)
            diff = large_small_diff(pts, large_roi)

            # reference frame value
            mask = np.zeros(ROIselect_frame.shape[:2], dtype = np.uint8)
            cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], (255, 255, 255))
            reference_RGB = []
            for i in range(3):
                std_frame = histogram_grayworld_whitebalance(ROIselect_frame)
                channel_values = std_frame[:, :, i][mask == 255]
                reference_RGB.append(np.mean(channel_values))
            reference_RGB = torch.tensor(np.array(reference_RGB), dtype = torch.float32)
            # LSTM data prep
            LSTM_data = torch.cat((LSTM_data, reference_RGB.unsqueeze(0)), dim = 0)

        ## realtime ROI tracking and pressure prediction
        while True:
            ret, frame = video_cap.read()
            if not ret: # quit algorithm when video reaches the end
                print("End of video")
                break
            tracking_success, roi_coords = tracker.update(frame)
            blink_status_store.add(not tracking_success)
            # blinking rate per minute
            blink_rate = np.sum(blink_status_store.get())/len(blink_status_store.get())
            """
                1. create an empty (some easily manipulable datatype) to store the sequential time series data
                2. create a helper to dump the storage with specified length of memory (e.g. dumping the series data storage after 10 times steps)
                3. change input type to take into account of n timesteps of RGB value and use the model
            """

            if tracking_success:
                # Convert ROI coordinates to integers
                roi_coords = tuple(map(int, roi_coords))
                # need to insert new ROI coordinate with tracker information about the lens
                # current coordinates are stored in array form (nrow, ncol)
                # loop here!
                for row in range(pts.shape[0]):
                    pts[row, :] = new_smallROI(diff[row, :], roi_coords)
            
                #pts = pts.reshape((-1, 1, 2))
                mask = np.zeros(frame.shape[:2], dtype = np.uint8)
                cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], (255, 255, 255))

                # manipulate the real-time frames
                RGB_mean = []
                for i in range(3):
                    std_frame = histogram_grayworld_whitebalance(frame)
                    channel_values = std_frame[:, :, i][mask == 255]
                    #channel_values = frame[:, :, i][mask == 255]
                    RGB_mean.append(np.mean(channel_values))
                blue, green, red = RGB_mean
                
                rgb_tensor = torch.tensor([red, green, blue], dtype = torch.float32)
                # relative RGB value to the first/selected frame
                ref_rgb_tensor = rgb_tensor - reference_RGB
                LSTM_data = torch.cat((LSTM_data, ref_rgb_tensor.unsqueeze(0)), axis = 0)
                LSTM_data = LSTM_data[-10:]
                LSTM_input = lstm_input_prep(LSTM_data)

                ##### blinking features #####
                # eyelid pressure classification
                with torch.no_grad():
                    pressure_pred = Pressure_RGB_model(LSTM_input).item()
                
                text = (f"Red channel is {red}", 
                        f"Green channel is {green}", 
                        f"Blue channel is {blue}", 
                        f"Predicted Pressure is {pressure_pred}", 
                        #f"Blink rate of past minute {blink_rate}",
                        f"There are {blink_rate * 60} blinks in the past minute",
                        "Press q to end session")
                
                ##### update tracked regions & display real-time information #####
                # draw the lens ROI
                cv2.rectangle(frame, (int(roi_coords[0]), int(roi_coords[1])),
                    (int(roi_coords[0] + roi_coords[2]), int(roi_coords[1] + roi_coords[3])),
                    (0, 255, 0), 2)  # (0, 255, 0) is the color (green), and 2 is the thickness
                # draw the color changing ROI
                for vertex_id in range(pts.reshape((-1, 1, 2)).shape[0]):
                    vertex = tuple(pts.reshape((-1, 1, 2))[vertex_id, :, :][0])
                    cv2.circle(frame, vertex, 2, (0, 0, 255), -1)
                cv2.polylines(frame, [pts], isClosed = True, color = (0, 0, 255), thickness = 2)
                y0 = 50
                dy = 40
                for i, line in enumerate(text):
                    y = y0 + i*dy
                    cv2.putText(frame, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, lineType = cv2.LINE_AA)
                cv2.imshow('Image', frame)
            else: # when the lens was not detected
                error_frame = frame.copy()
                cv2.putText(error_frame, "Tracking failed!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Image', error_frame)
            print(blink_status_store.get())
            if cv2.waitKey(1) == ord('q'):
                break

        video_cap.release()
        cv2.destroyAllWindows()

def webcam_capture():
    video_cap = cv2.VideoCapture(0)
    LSTM_data = torch.tensor(np.zeros(shape = (9, 3)), dtype = torch.float32)
    points = []
    # keep showing the video until find good frame to use
    exit_while = False
    while True:
         # store blink (bool) information for blink tracking (e.g. blinking rate per minute, blink interval)
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        # set the number of element to store in the blink_rate_storage to the number of frames per minute
        # then use the number of boolean values (np.sum(bool_blink)/len(bool_blink)) to determine blink rate
        blink_time_store = RollingBuffer(int(fps*60)) # storing the time stamp of each frame
        blink_status_store = RollingBuffer(int(fps*60)) # storing the status (boolean open/close) of each frame
        ret, first_frame = video_cap.read()
        selection_frame = first_frame.copy()
        guide_selectstartframe = 'Press the "y" key on keyboard to select the first frame to get started'
        cv2.putText(selection_frame, guide_selectstartframe, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, lineType = cv2.LINE_AA)
        cv2.imshow('Select Starting Frame', selection_frame)
        # if press the "y" key will capture that corresponding frame
        if cv2.waitKey(1) & 0xFF == 121:
            ## selecting region of the video frame for RGB value extraction
            if ret:
                # guide_selectROI = ("1. Use mouse to drag a square around the lens of interest for tracking.",
                # "2. Press the enter key on keyboard to confirm selection.",
                # "3. Press the 'c' key on keyboard to exit the program.")
                # Select the ROI
                large_roi = cv2.selectROI(first_frame)
                x, y, w, h = large_roi
                cv2.rectangle(first_frame, (int(large_roi[0]), int(large_roi[1])),
                        (int(large_roi[0] + large_roi[2]), int(large_roi[1] + large_roi[3])),
                        (0, 255, 0), 2)  # Here, (0, 255, 0) is the color (green), and 2 is the thickness
                # for i, line in enumerate(guide_selectROI):
                #     y = 50 + i*40
                #     cv2.putText(first_frame, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, lineType = cv2.LINE_AA)

                # Initialize the KCF tracker
                tracker = cv2.TrackerKCF_create()
                tracker.init(first_frame, (x, y, w, h))

                #points = []
                ROIselect_frame = first_frame.copy()
                def select_point(event, x, y, flags, param):
                    #global points
                    if event == cv2.EVENT_LBUTTONDOWN:
                        cv2.circle(ROIselect_frame, (x, y), 2, (0, 0, 255), -1)
                        cv2.imshow('Select ROI on lens', ROIselect_frame)
                        points.append([x, y])
                cv2.namedWindow('Select ROI on lens')
                cv2.setMouseCallback('Select ROI on lens', select_point)
                while True:
                    cv2.imshow('Select ROI on lens', ROIselect_frame)
                    if cv2.waitKey(0) & 0xFF == 13:  # return key
                        exit_while = True
                        break
                # Convert points to numpy array
                pts = np.array(points, np.int32)
                diff = large_small_diff(pts, large_roi)

                """
                add reference (1st frame) frame RGB here
                """
                mask = np.zeros(ROIselect_frame.shape[:2], dtype = np.uint8)
                cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], (255, 255, 255))
                reference_RGB = []
                for i in range(3):
                    # standardized RGB value in frame

                    #applying RGB standardization WILL slow down the tracking speed
                    std_frame = histogram_grayworld_whitebalance(ROIselect_frame)
                    channel_values = std_frame[:, :, i][mask == 255]
                    #channel_values = frame[:, :, i][mask == 255]
                    reference_RGB.append(np.mean(channel_values))
                reference_RGB = torch.tensor(reference_RGB, dtype = torch.float32)
                # LSTM data prep
                LSTM_data = torch.cat((LSTM_data, reference_RGB.unsqueeze(0)), dim = 0)

                if exit_while == True:
                    break

    ## realtime ROI tracking and pressure prediction
    while True:
        _, frame = video_cap.read()
        tracking_success, roi_coords = tracker.update(frame)
        blink_status_store.add(not tracking_success)
        # blinking rate per minute
        blink_rate = np.sum(blink_status_store.get())/len(blink_status_store.get())
        """
            1. create an empty (some easily manipulable datatype) to store the sequential time series data
            2. create a helper to dump the storage with specified length of memory (e.g. dumping the series data storage after 10 times steps)
            3. change input type to take into account of n timesteps of RGB value and use the model
        """

        if tracking_success:
            roi_coords = tuple(map(int, roi_coords))
            for row in range(pts.shape[0]):
                pts[row, :] = new_smallROI(diff[row, :], roi_coords)
            mask = np.zeros(frame.shape[:2], dtype = np.uint8)
            cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], (255, 255, 255))

            # manipulate the real-time frames
            RGB_mean = []
            for i in range(3):
                # standardized RGB value in frame

                #applying RGB standardization WILL slow down the tracking speed
                std_frame = histogram_grayworld_whitebalance(frame)
                channel_values = std_frame[:, :, i][mask == 255]
                #channel_values = frame[:, :, i][mask == 255]
                RGB_mean.append(np.mean(channel_values))
            blue, green, red = RGB_mean
            
            rgb_tensor = torch.tensor([red, green, blue], dtype = torch.float32)
            # relative RGB value to the first/selected frame
            ref_rgb_tensor = rgb_tensor - reference_RGB
            LSTM_data = torch.cat((LSTM_data, ref_rgb_tensor.unsqueeze(0)), axis = 0)
            LSTM_data = LSTM_data[-10:]
            LSTM_input = lstm_input_prep(LSTM_data)
            with torch.no_grad():
                pressure_pred = Pressure_RGB_model(LSTM_input).item()
            
            text = (f"Red channel is {red}", 
                    f"Green channel is {green}", 
                    f"Blue channel is {blue}", 
                    f"Predicted Pressure is {pressure_pred}",
                    f"There are {blink_rate * 60} blinks in the past minute", 
                    "Press q to exit session")
            # draw the lens ROI
            cv2.rectangle(frame, (int(roi_coords[0]), int(roi_coords[1])),
                (int(roi_coords[0] + roi_coords[2]), int(roi_coords[1] + roi_coords[3])),
                (0, 255, 0), 2)  # (0, 255, 0) is the color (green), and 2 is the thickness

            # draw the color changing ROI
            for vertex_id in range(pts.reshape((-1, 1, 2)).shape[0]):
                vertex = tuple(pts.reshape((-1, 1, 2))[vertex_id, :, :][0])
                cv2.circle(frame, vertex, 2, (0, 0, 255), -1)
            cv2.polylines(frame, [pts], isClosed = True, color = (0, 0, 255), thickness = 2)
            #cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y0 = 50
            dy = 40
            for i, line in enumerate(text):
                y = y0 + i*dy
                cv2.putText(frame, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, lineType = cv2.LINE_AA)
            cv2.imshow('Image', frame)
        else:
            error_frame = frame.copy()
            cv2.putText(error_frame, "Tracking failed!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Image', error_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    video_cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Real-time DES Diagnositic Assistance")
# Set the window size
window_width = 800
window_height = 425
root.geometry(f"{window_width}x{window_height}")

# Add background image
bg_image = tk.PhotoImage(file=f"{current_file_directory}/Background_App.png")  # Change "background.png" to your image file
bg_label = tk.Label(root, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

# Define a custom font for title
custom_font = ("Broadway", 20)  # Font family and size
APP_title = tk.Label(root, text = 'Real-time DES Diagnostic Assistance', font = custom_font)
#APP_title.pack(side = 'top', anchor = 'n')
APP_title.place(x = 250, y = 150)

upload_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_button.place(x = 350, y = 200)

webcam_button = tk.Button(root, text="Use Webcam", command=webcam_capture)
#webcam_button.pack(pady=10)
#webcam_button.grid(row = 5, pady=10)
webcam_button.place(x = 350, y = 230)

root.mainloop()
