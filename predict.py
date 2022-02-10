from __future__ import division
import cv2
import glob
import natsort
import time
import imutils
import sys
import joblib
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy.signal import convolve2d

#import model
filename = "RF25.joblib"
loaded_model = joblib.load(filename)

PATH_TO_DATA = sys.argv[1]
PATH_TO_OUTPUT = sys.argv[2]

def lpq(img):
    winSize=3
    STFTalpha=1/winSize  
    img=np.float64(img) # Convert np.image to double
    radius=(winSize-1)/2 
    x=np.arange(-radius,radius+1)[np.newaxis] # Form spatial coordinates in window
    w0=np.ones_like(x)
    w1=np.exp(-2*np.pi*x*STFTalpha*1j)
    w2=np.conj(w1)

    ## Run filters to compute the frequency response in the four points
    filterResp1=convolve2d(convolve2d(img,w0.T,'valid'),w1,'valid')
    filterResp2=convolve2d(convolve2d(img,w1.T,'valid'),w0,'valid')
    filterResp3=convolve2d(convolve2d(img,w1.T,'valid'),w1,'valid')
    filterResp4=convolve2d(convolve2d(img,w1.T,'valid'),w2,'valid')

    # Initilize frequency domain matrix for four frequency coordinates
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    #Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    LPQdesc=np.histogram(LPQdesc.flatten(),range(100))[0]
    LPQdesc=LPQdesc/LPQdesc.sum()
    return LPQdesc

def getBoundingBox(image):
    coords = cv2.findNonZero(image) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    rect = image[y:y+h, x:x+w]
    return rect

def unifyBackground(image):
    line1 = image[0,:]
    line2 = image[-1,:]
    avg1 = np.average(line1)
    avg2 = np.average(line2)
    avg = (avg1 + avg2) / 2
    if avg > 0.5:
        image = 1-image
    return image

def denoise(image):
    # Find contours and remove small noise
    img=np.copy(image)
    img=image.astype(np.uint8)
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rect_areas = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        rect_areas.append(w * h)
    avg_area = np.mean(rect_areas)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cnt_area = w * h
        if cnt_area < 0.01 * avg_area:
            img[y:y + h, x:x + w] = 0          
    return img

def preprocessing(image):
    gray = rgb2gray(image)
    # gray=cv2.resize(gray,(400,200))
    thresh = threshold_otsu(gray)
    gray[gray > thresh] = 1
    gray[gray < thresh] = 0
    gray = unifyBackground(gray)
    denoised= denoise(gray)
    gray = getBoundingBox(denoised)
    return gray


first = True
f = open(PATH_TO_OUTPUT+'/results.txt', 'w')
f1 = open(PATH_TO_OUTPUT+'/times.txt', 'w')
for image in natsort.natsorted(glob.glob(PATH_TO_DATA +"/*")):
    if not first:
        f.write('\n')
        f1.write('\n')
    img = cv2.imread(str(image))
    t_before=time.time()
    preprocessedImg = preprocessing(img)
    lp=lpq(preprocessedImg)
    predicted = loaded_model.predict([lp])
    t_duration = time.time()-t_before
    t_duration = round(t_duration, 3)
    if t_duration == 0:
        f1.write("0.001")
    else:
        f1.write(str(t_duration))
    f.write(str(int(predicted[0])))
    first = False

f.close()
f1.close()