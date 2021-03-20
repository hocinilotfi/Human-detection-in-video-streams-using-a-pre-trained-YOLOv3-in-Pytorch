#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:20:50 2021

@author: lotfi
"""

#%%
'''
    Import necessary packages

'''
from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import requests

import cv2
import ffmpeg

from tqdm import tqdm
import time
import csv

import sys, getopt

class DetectHumans:

    """Summary
    The Detection class

    Attributes:
        class_path (str): Description
        classes (TYPE): Description
        conf_thres (float): Description
        config_path (str): Description
        detectionFlags (list): Description
        f_height (TYPE): Description
        f_width (TYPE): Description
        img_size (int): Description
        input_video_path (TYPE): Description
        model (TYPE): Description
        nms_thres (float): Description
        ouput_video_path (TYPE): Description
        Tensor (TYPE): Description
        weights_path (str): Description
    """

    def __init__(self):
        
        self.config_path='config/yolov3.cfg'
        self.weights_path='config/yolov3.weights'
        self.class_path='config/coco.names'
        
        self.img_size=416
        self.conf_thres=0.5
        self.nms_thres=0.4
        self.detectionFlags=[]
        
    def load_model(self):
        """Summary
        Load model and weights
        """
        
        self.model = Darknet(self.config_path, img_size=self.img_size)

        #check if yolov3.weights file exists else download it
        if not os.path.exists(self.weights_path):
            print("downloading weights from web")
            filename=self.weights_path
            url="https://pjreddie.com/media/files/yolov3.weights"
            chunkSize = 1024
            r = requests.get(url, stream=True)
            with open(filename, 'wb') as f:
                pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
                for chunk in r.iter_content(chunk_size=chunkSize): 
                    if chunk: # filter out keep-alive new chunks
                        pbar.update (len(chunk))
                        f.write(chunk)



        self.model.load_weights(self.weights_path)
        self.model.cuda()
        self.model.eval()
        self.classes = utils.load_classes(self.class_path)
        self.Tensor = torch.cuda.FloatTensor

    def detect_image(self,img):
        """Summary
        Detect Humans
        Args:
            img (Image): the current frame to process
        
        Returns:
            all detected objects
        """
        # scale and pad image
        ratio = min(self.img_size/img.size[0], self.img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])

        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(self.Tensor))

        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = utils.non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)
        return detections[0]
    
    # load image and get detections
    def detect(self, input_video_path, output_video_path):
        """Summary
            Detecting humans in the frame and draw the corresponding bounding boxes
        Args:
            input_video_path (string): input video file
            output_video_path (TYPE):  output videofile
        """
        import cv2
        self.input_video_path = input_video_path
        self.ouput_video_path = output_video_path
        
        # initialization of the video stream reader
        cap = cv2.VideoCapture(self.input_video_path)

        # Get the number of frames per second
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the total number of frames
        tnf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
       
        self.f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.f_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("tmp_video.avi",fourcc, fps, (self.f_width,self.f_height))

        print("\n\nFrame Processing in progress ...\n")
        #initialization of the progress bar
        pbar = tqdm(total=tnf)
        while(cap.isOpened()):
            ret, imgcv = cap.read()

            if ret == False: # If there is no grabed frame
                break

            else: 

                img = Image.fromarray(imgcv)
                
                detections = self.detect_image(img)

                img = imgcv
                pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
                pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
                unpad_h = self.img_size - pad_y
                unpad_w = self.img_size - pad_x
                
                #flag indicating the presence of Humans in the current frame
                there_is_humans=0 

                if detections is not None: #if there a detected object

                    # browse detections and draw bounding boxes
                    
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        if cls_pred==0: #if the detected object is a person
                            there_is_humans = 1

                            # Re-calculate the bounding boxes coordinate since the frames are padded
                            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                            # Starting coordinates
                            start_point = (int(x1), int(y1)) 
                            
                            # Ending coordinate (represents the bottom right corner of rectangle) 
                            end_point = (int(x1+box_w), int(y1+box_h)) 
                            
                            # An awsome color in BGR hhh
                            color = (36,255,12)
                            
                            # Line thickness of 2 px 
                            thickness = 2
                            
                            # Draw a rectangle 'color' line borders and thickness of 2 px
                            cv2.rectangle(img, start_point, end_point, color, thickness) 

                            # Draw the caption rectangle + triange + text
                            triangle_cnt = np.array( [(int(x1+100), int(y1)), (int(x1+100), int(y1-28)), (int(x1+128), int(y1))] )
                            cv2.drawContours(img, [triangle_cnt], 0, color, -1)
                            cv2.rectangle(img, (int(x1), int(y1-28)), (int(x1+100), int(y1)), color, -1)
                            cv2.putText(img, 'Human', (int(x1+5), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,200,255), 2)
                                              
                self.detectionFlags.append(there_is_humans)
                out.write(img)
                
                # Progress bar update
                pbar.update(1)

                # Wait for interruptions
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        pbar.close()

        # Write the detection flags
        with open('detectionFlags.csv', 'w') as f:
            write = csv.writer(f) 
            write.writerow(self.detectionFlags) 

        cap.release()
        out.release()
        cv2.destroyAllWindows()
       
    
    def mix_video_and_audio(self):
        """Summary
        Mixing the audio and the video streams
        """
        print("\n\nMixing the Audio and Video:\n")
        vinput_without_sound = ffmpeg.input("tmp_video.avi")
        vinput_original= ffmpeg.input(self.input_video_path)
        audio = vinput_original.audio
        video = vinput_without_sound.video
        out = ffmpeg.output(audio, video, self.ouput_video_path)
        out.run()    
        
    def runDetection(self, inputfile, outputfile):
        """Summary
        Run the the whole aperations
        Args:
            inputfile 
            outputfile
        """
        self.load_model()
        self.detect(inputfile,outputfile)
        time.sleep(2)
        self.mix_video_and_audio()
        
def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print ('usage: detect_humans.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('usage: detect_humans.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    if (inputfile != "") and (outputfile != ""):
        DH = DetectHumans()
        DH.runDetection(inputfile, outputfile)
    else:
        print("Invalide arguments !")
        print ('usage: detect_humans.py -i <inputfile> -o <outputfile>')


if __name__ == "__main__":
    main(sys.argv[1:])