# -*- coding: utf-8 -*-
"""

Team:tripLM 16-3

U1610146 Mirzashomol Karshiev
U1610137 Mardon Zarifjonov
U1610143 Mirpulatjon Shukurov
U1610125 Lazizbek Qahorov


"""

import cv2
import tensorflow as tf
import numpy as np

#codec for saving the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')

#The video will be saved as result.avi format with resolution 800x600
out = cv2.VideoWriter('output.avi',fourcc, 30, (800,600))
out1 = cv2.VideoWriter('result2.avi',fourcc, 30, (800,600))
#Initialization kernels for edge detection
kernel = np.ones((3,3), np.uint8) 
kernel_edge = np.array([
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]])


class ImageManipulation:
    
    #<-----------------Object Creation------------------>
    def __init__(self):
        pass
       
    
    #<-----------------Machine Learning (Linear Regression)----------------->
    def __regressionLines__(self,x_train,y):
        
        #Checks wheter training set is empty or not
        if len(x_train)==0 or len(y)==0:
            print('empty list')
        else :
            
            #Creates the graph
            sess = tf.compat.v1.Session()
                
            m=len(y)
            
            #Weight parameters of model
            W=tf.Variable(tf.random.normal([len(x_train)]),name="weight")
            #Bias parameters of model
            b=tf.Variable(tf.random.normal([len(x_train)]),name='bias')
            #Hypothesis model
            y_pred=x_train*W+b
            #Loss function that is MSE (Mean Squarred Error)    
            cost=tf.reduce_mean(tf.square(y-y_pred))/(2*m)
            #Gradient descent optimizer for finding min value of loss 
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(0.00001)
            #Starts to take deriviate from cost function based on parameters
            train=optimizer.minimize(cost)
            #Initializes the Weight and Bias parameters
            sess.run(tf.compat.v1.global_variables_initializer())
            
            #Number of steps for training the model
            for step in range(301):
                
                #optmizes the parameters by taking derivative
                sess.run(train)
                #if step%20==0:
                    #print(step,sess.run(cost),sess.run(W),sess.run(b))
            
            #Final Weight parameters are obtained
            weight = sess.run(W) 
            #Final Bias parameters are obtained
            bias = sess.run(b) 
            #Prediction of X_train data and returnning the responces
            prediction=np.array(x_train)*weight+bias
            
            #returns numpy array of y responces to x_train points(line)
            return prediction
    
    
    #<-----------------Image Processing using OpenCV------------------>
    def __houghLinesDetection__(self,img):
        
        
        try:
            #Resizing the image in order to fit output video resoultion
            image = cv2.resize(img,(800,600),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            
            
            ######################## EDGE DETECTION############################
            #Convering the image into Gray Scale
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #Blurring image using GaussianBlur in order to average pixels (reduce the noise)
            denoised_image=cv2.GaussianBlur(gray,(5,5),0)
            #Converting the image into form of binary numbers to reduce the features
            binary_frame = cv2.threshold(denoised_image,128,255,cv2.THRESH_BINARY)[1]
            #dilation is needed to better detect edges
            dilation = cv2.dilate(binary_frame,kernel,iterations=1)
            #edge detection using 2d filter
            edge = cv2.filter2D(dilation,-1,kernel_edge)
          
            
            ###################### REGION OF INTEREST #########################
            #takes the height of numpy array image
            height=edge.shape[0]        
            #takes the width of numpy array image
            width=edge.shape[1] 
            
            #Create the dark image of size as original image
            dark_image1=np.zeros([height, width, 3], dtype=np.uint8)
            #Creating polyong using 3 points for region of interest
            polygon1=np.array([[(140,600),(750,600),(420,400)]])
            #Masking the dark image wiht polygon
            cv2.fillPoly(dark_image1,polygon1,(255,255,255))           
            #Bitwise the image and mask to get region of interest
            masked_image1=cv2.bitwise_and(image,dark_image1)            
            
            
            
            ############# EDGE DETECTION OF REGION OF INTEREST ################
            #Converting ROI (region of interest) into gray scale
            roi_gray=cv2.cvtColor(masked_image1,cv2.COLOR_BGR2GRAY)
            #Denoising the ROI
            roi_blur=cv2.GaussianBlur(roi_gray,(5,5),0)
            #Converting denoised ROI into binary format
            roi_binary= cv2.threshold(roi_blur,128,255,cv2.THRESH_BINARY)[1]
            #Detectin the edges from binary ROI
            roi_edge=cv2.filter2D(roi_binary,-1,kernel_edge)
            
            
            
            ############# LINE DETECTION OF REGION OF INTEREST ################
            #Detectin the lines from edges that are obtained from filter2D
            lines=cv2.HoughLinesP(roi_edge,1,np.pi/180,45,np.array([]),minLineLength=2,maxLineGap=150)
            
            
            
            ############# SEPERATING LINES INTO LEFT AND RIGHT ################
            #Checks whether frame has lines
            if lines is not None: 
                #Left line coordinates list
                left=[]
                #Right line coordintes list
                right=[]
                #Gettin the the center of image or half of width
                img_center = roi_edge.shape[1]//2
                #If lines is not empty
                if len(lines)!=0:
                    #Seperating Lines into left and right
                    for line in lines:
                        #Coordinates of line
                        for x2, y2, x1, y1 in line:
                            #If coordinates are righter from center of image
                            if x1>img_center and x2>img_center:
                                #Store the coordinates into right line
                                right.append(np.float32([x1,y1]))
                                right.append(np.float32([x2,y2]))
                            
                            #If coordinates are lefter from center of image
                            elif x1<img_center and x2<img_center:
                                #Store the coordinates into left line
                                left.append(np.float32([x1,y1]))
                                left.append(np.float32([x2,y2]))
                            else:
                                continue
                
                
                ########### SPLITTING LINES COORDINATES FOR MODEL  ############
                #Seperating the training data of left line
                X_left_train=[]
                y_left=[]
                if len(left)!=0:
                    #Seperating x_train values of left line
                    X_left_train = [first[0] for first in left]
                    #Seperating y values of left line
                    y_left = [first[1] for first in left]
                
                #Seperating the training data of right line
                X_right_train=[]
                y_right=[]
                if len(right)!=0:
                    #Seperating x_train values of right line
                    X_right_train=[first[0] for first in right]
                    #Seperating y values of right line
                    y_right=[first[1] for first in right]
           
           
                #Right line list
                line_points_right=[]
                #Left line list
                line_points_left=[]
                
                ########### FITTING THE MODEL WITH TRAINING DATA  #############
                #The lines will be drawed on the frame only if ROI contains both left and right lines
                if len(X_right_train)!=0 and len(X_left_train)!=0:
                    
                    #Fit the right line data into model
                    right_model=self.__regressionLines__(X_right_train, y_right)
                   
                    #Making coordinates for right line
                    bool=0
                    for i in range(len(X_right_train)):
                        bool+=1
                        
                        if bool==2:
                            line_points_right.append([X_right_train[i-1],right_model[i-1],X_right_train[i],right_model[i]])
                            bool=0
    
                    #Drawing line of right line on image
                    for i in range(1):
                        cv2.line(image,(line_points_right[i][0],line_points_right[i][1]),(line_points_right[i][2],line_points_right[i][3]),(0,0,255),5)
                    
                    
                    
                    #Fit the left line data into model 
                    left_model=self.__regressionLines__(X_left_train, y_left)
                    #Making cooridnates of left line
                    bool=0
                    for i in range(len(X_left_train)):
                        bool+=1
                        
                        if bool==2:
                            line_points_left.append([X_left_train[i-1],left_model[i-1],X_left_train[i],left_model[i]])
                            bool=0
                    
                    #Drawing line of left line on the miage
                    for i in range(1):
                        cv2.line(image,(line_points_left[i][0],line_points_left[i][1]),(line_points_left[i][2],line_points_left[i][3]),(0,0,255),5)
                
                    
                    #returing line detected image
                    return image
        except Exception as e:
            print(e)
    
    #<----------------- Predicting the turn based on regression lines ------------------>   
    def __predictTurn__(self,frame):
        ################# DRAWING RECTANGLE FOR PREDICTING TURNS###############
        #FPS of the video
        cv2.putText(frame,'FPS of video: 30',(50,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        #Obtained FPS
        cv2.putText(frame,'FPS: '+str(int(fps)),(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
        
        #Drawing Rectangle for turn prediction
        cv2.line(frame,(220,580),(600,580),(255,0,0),5)
        cv2.line(frame,(600,580),(600,500),(255,0,0),5)
        cv2.line(frame,(220,580),(220,500),(255,0,0),5)
        cv2.line(frame,(220,500),(600,500),(255,0,0),5)
        
        
        #Converting to gray scale        
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray_blur=cv2.GaussianBlur(gray,(5,5),0)
        edge = cv2.filter2D(gray_blur,-1,kernel_edge)
                
        
        
        ############## DETECTING THE LINES OF ROI##############################
        #takes the height of numpy array image
        height=edge.shape[0]        
        #takes the width of numpy array image
        width=edge.shape[1] 
           
        #Create the dark image of size as original image
        dark_image2=np.zeros([height, width, 3], dtype=np.uint8)
        #Creating polyong using 3 points for region of interest
        polygon2=np.array([[(140,600),(750,600),(420,320)]])
        #Masking the dark image wiht polygon
        cv2.fillPoly(dark_image2,polygon2,(255,255,255))           
        #Bitwise the image and mask to get region of interest
        masked_image2=cv2.bitwise_and(frame,dark_image2)            
                
        #Converting ROI (region of interest) into gray scale
        roi_gray=cv2.cvtColor(masked_image2,cv2.COLOR_BGR2GRAY)
        #Denoising the ROI
        roi_blur=cv2.GaussianBlur(roi_gray,(5,5),0)
        #Converting denoised ROI into binary format
        roi_binary= cv2.threshold(roi_blur,128,255,cv2.THRESH_BINARY)[1]
        #Detectin the edges from binary ROI
        roi_edge=cv2.filter2D(roi_binary,-1,kernel_edge)
                
        #detecting lines
        lines=cv2.HoughLinesP(roi_edge,1,np.pi/180,45,np.array([]),minLineLength=2,maxLineGap=150)
            
        
        
        ############ FINDS THE COORDINATES OF LINES FOR LANE CENTER##########  
        if lines is not None:
            
            #Frame center of ROI
            frame_center=360.0
            img_center = edge.shape[1]//2
                
            line_x1=0
            line_x2=0
            line_y1=0
            line_y2=0
            lane_center=0
            for line in lines:
                for x1,y1,x2,y2 in line:
                    if x1>img_center and x2>img_center:
                        #right line coordinates
                        line_x1=x2
                        line_y1=y2
                        
                    elif x1<img_center and x2<img_center:
                        #left line coordinates
                        line_x2=x2
                        line_y2=y2
                    else:
                        continue
             
            ################# BASED ON DISTANCE PREDICTS THE TURN##############
            if line_x1!=0 and line_x2!=0:
                #Calculating the center of lane
                lane_center=(line_x1+line_x2)/2
                        
                #Distance between center of frame and center of lane
                distance=frame_center-lane_center
                cv2.putText(frame,'DISTANCE: '+str(distance),(50,90),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0))
                print('dist',distance)       
            
                if (distance<-120 and distance>-150):
                    print("distance",frame_center-lane_center)
                    cv2.putText(frame,'turn right',(300,300),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0))
                    out1.write(frame)
                        
                if (distance>-90 and distance<-60):
                    print("distance",frame_center-lane_center)
                    cv2.putText(frame,'turn slightly right',(300,300),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0))
                    out1.write(frame)
                
                if (distance>60 and distance<90):
                    print("distance",frame_center-lane_center)
                    cv2.putText(frame,'turn sligtly left',(300,300),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0))
                    out1.write(frame)
                if (distance>120 and distance<150):
                    print("distance",frame_center-lane_center)
                    cv2.putText(frame,'turn left',(300,300),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0))
                    out1.write(frame)
                    
                    
                    
        
        dark_image1=np.zeros([height, width, 3], dtype=np.uint8)
        #Creating polyong using 3 points for region of interest
        polygon1=np.array([[(180,450),(500,450),(500,400),(180,400)]])
        #Masking the dark image wiht polygon
        cv2.fillPoly(dark_image1,polygon1,(255,255,255))           
        #Bitwise the image and mask to get region of interest
        masked_image1=cv2.bitwise_and(frame,dark_image1) 
        
        cv2.imshow('ob',masked_image1)
        imgray = cv2.cvtColor(masked_image1, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
        for c in contours:
            if cv2.contourArea(c) <= 50:
                continue    
            
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            
            if len(approx)==4:
                x,y,w,h = cv2.boundingRect(c)
                center = (x,y)
                print (center)
                
                
                    
                if line_y1<y:
                    if line_x2>x and line_x2<line_x1:
                        print('left:',line_x2,'x:',x)
                        cv2.rectangle(frame, (int(x), int(y)), (int(x) + w, int(y) + h), (0, 255,0), 2)
                        cv2.putText(frame,'the car is moving on the left side',(0,200),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0))
                    
                    elif line_x1<x and line_x1>line_x2:
                        print('right:',line_x1,'x:',x)
                        cv2.rectangle(frame, (int(x), int(y)), (int(x) + w, int(y) + h), (0, 255,0), 2)
                        cv2.putText(frame,'the car is moving on the right side',(0,200),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0))
                
                if line_y2<y:
                    if line_x2>x and line_x2<line_x1:
                        print('left:',line_x2,'x:',x)
                        cv2.rectangle(frame, (int(x), int(y)), (int(x) + w, int(y) + h), (0, 255,0), 2)
                        cv2.putText(frame,'the car is moving on the left side',(0,200),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0))
                    
                    elif line_x1<x and line_x1>line_x2:
                        print('right:',line_x1,'x:',x)
                        cv2.rectangle(frame, (int(x), int(y)), (int(x) + w, int(y) + h), (0, 255,0), 2)
                        cv2.putText(frame,'the car is moving on the right side',(0,200),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0))
                
        
        #Storing frames in video
        out1.write(frame)
        



if __name__ == '__main__':
    #Object of Class ImageManipulation
    ob = ImageManipulation()
    #Object of VideoCapture
    capture=cv2.VideoCapture('test.mp4')
     ############# For detecting lines from video###############
    try:
        
        while True:
            #Reads the frames of video
            ret, frame = capture.read()
            timer = cv2.getTickCount()
            fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
            #If the ret is False then continue
            if ret is False:
                continue
            else:
                b=ob.__houghLinesDetection__(frame)
                out.write(b)
    
    #To stop recording the video press Ctrl-C
    except KeyboardInterrupt:
        print("Ctrl-C is Pressed")
        pass
    
    
    
    
    
    ############# For detecting lines and predicting turn from video###############
   
    """
    
    capture2=cv2.VideoCapture("output.avi")
    

    try:
        
        while True:
            #Reads the frames of video
            ret, frame = capture2.read()
            timer = cv2.getTickCount()
            fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
            #If the ret is False then continue
            if ret is False:
                continue
            else:
                ob.__predictTurn__(frame)
    
    #To stop recording the video press Ctrl-C
    except KeyboardInterrupt:
        print("Ctrl-C is Pressed")
        pass
    
"""
    
    #Release the object
    capture.release()
    #Relase the object
    out.release()
    #Close all windows
    cv2.destroyAllWindows()
    
    
