import cv2
import numpy as np
import imutils
import os
import time
import cv2
from PIL import Image
import imutils
from PIL import ImageFilter
from pytesseract import pytesseract


imageAdress = "Dataset/9.jpg" 
pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"

def detectTextForCrop(imageAdress):
        img = Image.open(imageAdress)
        width, height = img.size
        img = cv2.imread(imageAdress)
        if width > 180 and height >60: #180 60
            img = imutils.resize(img, width=900)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (11, 11), 2)
            kernel = np.ones((25,25), np.uint8)
            img_dilation = cv2.dilate(blurred, kernel, iterations=1)
            gray = img_dilation
            rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
            gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
            gradX = np.absolute(gradX)
            (minVal, maxVal) = (np.min(gradX), np.max(gradX))
            gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
            gradX = gradX.astype("uint8")
            print('>> Finding vertical scharr gradient')
            gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
            thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 20))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
            print('>> Generating Horizontal projection to find text')
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            areas = [cv2.contourArea(c) for c in cnts]
            max_index = np.argmax(areas)
            cnt=cnts[max_index]
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img,(x-20,y-20),(x+w+20,y+h+20),(0,255,0),2)
            cropped_img = img[y-10:y+h+10, x-10:x+w+10]
            cropped_img = imutils.resize(cropped_img, height=100)
            print('>> Cropping Text Roi')
            return(cropped_img)
        
        else:
            img = cv2.imread(imageAdress)
            return img


def detectLettersFromImage(mainImg):
    image = mainImg.copy()
    dst = cv2.fastNlMeansDenoisingColored(image.copy(), None, 10, 10, 7, 15)
    gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    edge = cv2.Canny(gray,40,20) #40,20
    cv2.imshow('edge', edge)
    kernel = np.ones((5,5),np.float32)/25
    blurred = cv2.filter2D(edge,-1,kernel)
    words_in_image =pytesseract.image_to_string(blurred)
    cv2.imshow('lastImage', blurred)
    print(words_in_image)
    return mainImg


mainImg = detectTextForCrop(imageAdress)
mainImg = detectLettersFromImage(mainImg)

cv2.imshow('main', mainImg)
cv2.waitKey(0)