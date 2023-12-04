import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import argparse
import glob
import cv2
# from sklearn.metrics import jaccard_score

sys.path.append('RAFT/core')  # This is in raft folder
from raft import RAFT  # in RAFT folder
from utils import flow_viz  # in RAFT folder
from utils.utils import InputPadder  # in RAFT folder

DEVICE = 'cuda'
###### -NN Related functions#######################################
def load_image(imfile):
   img = torch.from_numpy(imfile).permute(2, 0, 1).float()
   return img[None].to(DEVICE)

###### Post Processing##################################
def postprocess(curr_folder,flow_path, drone_frame,groundtruth):
   # Real frame
   frame = drone_frame
   # Load the flow image
   flow_image = cv2.imread(flow_path)

   # If the image has an alpha channel, remove it
   if flow_image.shape[-1] == 4:
       flow_image = cv2.cvtColor(flow_image, cv2.COLOR_BGRA2BGR)
   
   # Convert flow_image to grayscale
   gray_image = cv2.cvtColor(flow_image, cv2.COLOR_BGR2GRAY)
   gray_mask = cv2.cvtColor(groundtruth, cv2.COLOR_BGR2GRAY)
   gray_mask = cv2.bitwise_not(gray_mask)
   
   # Step 1: Noise reduction with Gaussian blur
   blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
   # Use adaptive thresholding to get a binary flow_image
   adaptive_thresh = cv2.adaptiveThreshold(
       blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 2)
   
   # Use Canny edge detection
   edges = cv2.Canny(adaptive_thresh, 50, 150)
   # Dilate the edges to close the gaps
   dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
   
   # Apply closing to fill in gaps
   closed_edges = cv2.morphologyEx(
       dilated_edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

   # Step 4: Find contours of the remaining objects (gaps)
   contours, hierarchy = cv2.findContours(
       closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
   # Gound truth contours
   contours_gt, hierarchy_gt = cv2.findContours(
       gray_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   contours_gt = sorted(contours_gt, key=cv2.contourArea, reverse=True)[:1]

   
#    newlist_contours = contours
#    newlist_contours = sorted(
#        newlist_contours, key=cv2.contourArea, reverse=True)[:1]
#    max_contour = newlist_contours[0]
   # Find the convex hull object for each contour
   hull_list = []
   for i in range(len(contours)):
       hull = cv2.convexHull(contours[i])
       hull_list.append(hull)

   # Sort contours by area and get the largest one
   contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
   hulls = sorted(hull_list, key=cv2.contourArea, reverse=True)[:1]

   (thresh1, prediction) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   (thresh2, target) = cv2.threshold(gray_mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

   intersection = np.logical_and(target, prediction)
   union = np.logical_or(target, prediction)
   iou_score = np.sum(intersection) / np.sum(union)
   print("iou is",iou_score)
   

   # Draw the largest contour and centroid if it exists
   frame_copy = frame.copy()
   for contour in contours:
       # Threshold to filter small contours (tunable)
       if cv2.contourArea(contour) > 100:
           cv2.drawContours(flow_image, [contour], -1, (255, 0, 255), 3)
        #    cv2.drawContours(flow_image, [max_contour], -1, (0, 255, 0), 3)
        #    cv2.drawContours(flow_image, hulls, -1, (255, 0, 0), 3) ###################
           cv2.drawContours(frame, [contour], -1, (0, 0, 255), 3)
           cv2.drawContours(frame_copy, [contour], -1, (0, 0, 255), 3)
        #    cv2.drawContours(frame, [max_contour], -1, (0, 0, 255), 3)
        #    cv2.drawContours(frame, hulls, -1, (255, 0, 0), 3) ######################
           M = cv2.moments(contour)
           if M["m00"] != 0:
               cX = int(M["m10"] / M["m00"])
               cY = int(M["m01"] / M["m00"])
               cv2.circle(flow_image, (cX, cY), 7, (255, 255, 255), -1)
               cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
               print("The centroid of the largest contour detected is:", cX, ",", cY)

   for contour in contours_gt:
       cv2.drawContours(frame_copy,[contour], -1, (0, 255, 0), 3)

   cv2.putText(frame_copy, "IoU: {:.4f}".format(iou_score), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
   cv2.putText(frame_copy, "Green: groundtruth", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
   cv2.putText(frame_copy, "Red: prediction", (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

   filepath1 = curr_folder + f"/Outputs/processed_flow/frame{0:03d}.png"
   cv2.imwrite(filepath1, flow_image)
   filepath2 = curr_folder + f"/Outputs/processed_frame/frame{0:03d}.png"
   cv2.imwrite(filepath2, frame)
   filepath3 = curr_folder + f"/Outputs/iou/frame{0:03d}.png"
   cv2.imwrite(filepath3, frame_copy)
   return contours,cX, cY

def main():
    curr_folder = os.path.dirname(__file__)
    # Loading the model
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
    # Get the frames
    frame1_path = curr_folder + "/Outputs/frames/frame10001.png"
    frame2_path = curr_folder + "/Outputs/frames/frame20001.png"
    mask1_path = curr_folder + "/Outputs/GTMasks/mask10001.png"
    mask1 = cv2.imread(mask1_path)

    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

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
        print("saving flow")
        flow_path = curr_folder + \
            str("/Outputs/flow/")+f"frame{1:03d}.png"
        cv2.imwrite(flow_path, flo)
        print("saved flow")


        pred,cX, cY = postprocess(curr_folder,flow_path, frame1,mask1)



if __name__=="__main__":
   # donot run main.py if imported as a module
   main()