import os
import time
import sys
import csv
import cv2
import numpy as np
import argparse
import glob
import torch
from PIL import Image
import matplotlib.pyplot as plt
from threading import Thread
from djitellopy import Tello

sys.path.append('RAFT/core')  # This is in raft folder
from raft import RAFT  # in RAFT folder
from utils import flow_viz  # in RAFT folder
from utils.utils import InputPadder  # in RAFT folder


DEVICE = 'cuda' # Global variable to set device to cuda. Use cuda torch to get good inference times of 0.5s per flow.

###### -NN Related functions#######################################
def load_image(imfile):
    """
    Loads the read frames in appropriate format for the network
    Args:
        image file (here the the frames we want to use for optical flow)
    """
    img = torch.from_numpy(imfile).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

#########Stream capture thread#####################################
def recordWorker():
    """
    Thread to capture the images from the drone camera
    """
    j = 0
    while True:
        frame = drone.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        filename = current_path + str("/frames_thread/") + f"frame{j}.png"
        cv2.imwrite(filename, frame)
        j = j+1

# -################--Post processing-#######################
def contour_area(contour):
    """
    Finds the contour area
    Args:
        contour
    Returns:
        Area
    """
    return cv2.contourArea(contour)

def postprocess(i, current_path, image_path):
    """
    Identifies the gap in the flow image and returns the center of our gap.
    Args:
        i: counter used to name the images being saved
        current_path: Path for the current file directory
        image_path: path to the flow image that we want to use
    Returns:
        cX, cY: Gap center on the image in pixel coordinates
    """
    # Load the flow image
    image = cv2.imread(image_path)

    # If the image has an alpha channel, remove it
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    # Load the real frame image
    real_path = current_path + f"/frames/frame{2*i+1:03d}.png"
    real_image = cv2.imread(real_path)
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Step 1: Noise reduction with Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
    # Use adaptive thresholding to get a binary image
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 2)
    # Use Canny edge detection
    edges = cv2.Canny(adaptive_thresh, 50, 150)
    # Dilate the edges to close the gaps
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    # print("statement 4")
    # Apply closing to fill in gaps
    closed_edges = cv2.morphologyEx(
        dilated_edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # Step 4: Find contours of the remaining objects (gaps)
    contours, hierarchy = cv2.findContours(
        closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    newlist_contours = contours
    newlist_contours = sorted(
        newlist_contours, key=cv2.contourArea, reverse=True)[:1]
    max_contour = newlist_contours[0]
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)

    # Sort contours by area and get the largest one (Which usually is the gap)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    hulls = sorted(hull_list, key=cv2.contourArea, reverse=True)[:1]

    # Draw the largest contour and centroid if it exists
    for contour in contours:
        # Threshold to filter small contours (tunable)
        if cv2.contourArea(contour) > 100:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
            cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 3)
            cv2.drawContours(image, hulls, -1, (0, 0, 255), 3)
            cv2.drawContours(real_image, [contour], -1, (0, 255, 0), 3)
            cv2.drawContours(real_image, [max_contour], -1, (0, 255, 0), 3)
            cv2.drawContours(real_image, hulls, -1, (0, 0, 255), 3)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
                cv2.circle(real_image, (cX, cY), 7, (255, 255, 255), -1)
                print("The centroid of the largest contour detected is:", cX, ",", cY)

    filepath = current_path + f"/center/frame{i:03d}.png"
    cv2.imwrite(filepath, image)
    filepath2 = current_path + f"/center_real/frame{i:03d}.png"
    cv2.imwrite(filepath2, real_image)
    

    return cX, cY
##########################################################################################


try:
    # Get the folder paths and create the necessary folders for saving outputs
    current_path = os.path.abspath(__file__)
    current_path = current_path.replace("Wrapper.py", "")

    folders_list = ["center", "flow", "frames",
                    "frames_thread", "center_real"]
    for folder in folders_list:
        folder_path = os.path.join(current_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Load the NN model
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision',
                        action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficent correlation implementation')
    args = parser.parse_args()
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # Connect to the drone
    drone = Tello()
    drone.connect()
    # CHECK SENSOR READINGS------------------------------------------------------------------
    print('Altitude ', drone.get_distance_tof())
    print('Battery, ', drone.get_battery())
    print('Temperature, ', drone.get_temperature())

    drone.streamon()

    Thread(target=recordWorker).start()
    drone.takeoff()
    time.sleep(2)
    drone.go_xyz_speed(0, 0, 20, 45)
    time.sleep(2)
    for j in range(0, 7):  # Ignores black frames we get initially
        frame1 = drone.get_frame_read().frame

    image_center = np.array([480, 360]) # (h/2,w/2) is fixed as drone captures images in (960,720) sizes
    tol = 200  # 200, 250 TUNABLE PARAMETER (Tolerance allowed between the image center and gap center for the drone to safely go through window)
    runs = 0
    framei = 0
    flowi = 0

    centers_dict = {}

    while True:
        frame_no = 0
        center_list = []
        while frame_no < 1: # Perform visual servo and get the flow

            try:
                # ------Servoing------------------
                drone.move_up(20)
                time.sleep(2)
                drone.move_down(20)
                time.sleep(1.3)
                # ----Reading two frames---------
                print("Reading frame 1")
                frame1 = drone.get_frame_read().frame
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
                H, W, _ = frame1.shape
                print("shape", (H, W))
                filename = current_path + \
                    str("/frames/") + f"frame{framei:03d}.png"
                cv2.imwrite(filename, frame1)
                frame1 = cv2.imread(filename)
                framei += 1
                time.sleep(0.3)  # keep it low before 0.2

                frame2 = drone.get_frame_read().frame
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
                H, W, _ = frame2.shape
                print("shape", (H, W))
                filename = current_path + \
                    str("/frames/") + f"frame{framei:03d}.png"
                cv2.imwrite(filename, frame2)
                frame2 = cv2.imread(filename)
                # -----------------------------------------
                # -------Get optical flow and do post-processing------------------
                with torch.no_grad():
                    print("starting nn")
                    image1 = load_image(frame1)
                    image2 = load_image(frame2)
                    print("loaded images")
                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)
                    print("padding done")
                    flow_low, flow_up = model(
                        image1, image2, iters=20, test_mode=True)
                    print("model ran")

                    img = image1[0].permute(1, 2, 0).cpu().numpy()
                    flo = flow_up[0].permute(1, 2, 0).cpu().numpy()

                    # map flow to rgb image
                    flo = flow_viz.flow_to_image(flo)
                    img_flo = np.concatenate([img, flo], axis=0)
                    image_path = current_path + \
                        str("/flow/")+f"frame{flowi:03d}.png"
                    cv2.imwrite(image_path, flo)
                    print("saved the flow", flowi)
                    # drone.send_keepalive()

                    cX, cY = postprocess(flowi, current_path, image_path)
                    center_list.append([cX, cY])
                    frame_no += 1
                    flowi += 1
                    framei += 1
            except Exception as error:
                print(f"An error occurred:{type(error).__name__} - {error}")
                continue
        runs += 1
        centers_dict[f"run{runs}"] = center_list

        # Find average center
        center_list = np.array(center_list)
        average_center = np.mean(center_list, axis=0)

        # -----Once we get the window center do one of the following------------------
        if np.linalg.norm(average_center-image_center) <= tol: # If dist b/w image center and gap center within tolerance send the drone through
            drone.go_xyz_speed(400, 0, 0, 90)
            time.sleep(3)
            drone.land()
            print(centers_dict)
            break
        # Otherwise convert the pixel distance between the centers to real world distance through trail and error to send the drone to that spot.
        conversion_factor = 0.20  # 0.20, 0.18, 0.15
        if image_center[0] - average_center[0] > 0:
            y_command = int(conversion_factor *
                            (abs(image_center[0] - average_center[0])))
            if y_command < 10:
                drone.go_xyz_speed(400, 0, 0, 95)
                drone.land()
            elif 10 < y_command < 20:
                drone.go_xyz_speed(0, 20, 0, 45)

            else:
                drone.go_xyz_speed(0, y_command, 0, 45)
        else:
            y_command = -int(conversion_factor *
                             (abs(image_center[0] - average_center[0])))
            if y_command > 10:
                drone.go_xyz_speed(500, 0, 0, 95)
                drone.land()
            elif 10 < y_command < 20:
                drone.go_xyz_speed(0, -20, 0, 45)
            else:
                drone.go_xyz_speed(0, y_command, 0, 45)
        if image_center[1] - average_center[1] > 0:
            z_command = int(conversion_factor *
                            (abs(image_center[1] - average_center[1])))
        else:
            z_command = -int(conversion_factor *
                             (abs(image_center[1] - average_center[1])))
        print(
            f"y command is: {int(conversion_factor*(abs(image_center[0] - average_center[0])))}")
        time.sleep(3)
except KeyboardInterrupt:
    # HANDLE KEYBOARD INTERRUPT AND STOP THE DRONE COMMANDS
    print('keyboard interrupt')
    drone.emergency()
    drone.land()
    drone.emergency()
    drone.end()
