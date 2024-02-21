import cv2
import numpy as np
import pandas as pd
import torch

'''For smaller ROI to follow larger ROI'''

def large_small_diff(coord, init_largeROI_coord):
    # retrieve the init_largeROI_coord
    init_Lx, init_Ly, _, _ = init_largeROI_coord
    init_Large_coord = np.array([init_Lx, init_Ly])
    
    # find the distance between the coord and init_largeROI_coord
    # should be 1D numpy arrays
    diff = init_Large_coord - coord
    return diff

def new_smallROI(diff, new_largeROI_coord):
    # find the new coord according to new_largeROI_coord (1D array)
    new_Lx, new_Ly, _, _ = new_largeROI_coord
    new_large_coord = np.array([new_Lx, new_Ly])
    new_coord = new_large_coord - diff
    return new_coord

'''For standardizing RGB value in video under different lighting conditions'''

def gray_world_assumption(image):
    # Compute the average color in each channel
    avg_color = np.mean(image, axis=(0, 1))
    # Calculate the scaling factor for each channel
    scaling_factors = 128 / avg_color
    # Apply the scaling factors to each channel
    balanced_image = image * scaling_factors.reshape(1, 1, 3)
    # Clip the values to be in the valid range [0, 255]
    balanced_image = np.clip(balanced_image, 0, 255).astype(np.uint8)
    return balanced_image

def histogram_equalization(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab_image)
    # Convert the list to a mutable type (list)
    lab_planes_list = list(lab_planes)
    # Apply histogram equalization to the L channel
    lab_planes_list[0] = cv2.equalizeHist(lab_planes_list[0])
    # Convert the list back to a tuple
    lab_planes = tuple(lab_planes_list)
    equalized_lab = cv2.merge(lab_planes)
    equalized_image = cv2.cvtColor(equalized_lab, cv2.COLOR_LAB2BGR)
    return equalized_image

def histogram_grayworld_whitebalance(image):
    hist_img = histogram_equalization(image)
    hist_gray_img = gray_world_assumption(hist_img)
    return hist_gray_img

'''Data preparation for modeling architecture'''
def lstm_input_prep(tensor):
    batch_size = 1
    sequence_length = tensor.shape[0]
    input_size = tensor.shape[1]
    lstm_input_tensor = tensor.view(batch_size, sequence_length, input_size)
    return lstm_input_tensor