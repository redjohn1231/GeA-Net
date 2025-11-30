# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import random
from numpy.random import uniform
import threading

def gasuss_noise(image, mu=0.0, sigma=0.1):

    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mu, sigma, image.shape)
    gauss_noise = image + noise
    if gauss_noise.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    gauss_noise = np.clip(gauss_noise, low_clip, 1.0)
    gauss_noise = np.uint8(gauss_noise * 255)
    return gauss_noise


def preprocessing(image,num):

    for i in num:
        if i==0: 
            image= image
        
        if i==1:
            r = np.random.rand()
            if r<=1:
                jpeg_quality = random.randint(30,100)  
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                result, encimg = cv2.imencode('.jpg', image, encode_param)
                decimg = cv2.imdecode(encimg, 1)
                image = decimg
            else:
                image = image
            
        if i==2:
            r = np.random.rand()
            if r<=1:
                # sigma = uniform(0,3,size=1)
                sigma =1
                sigma = float(sigma)
                image = cv2.GaussianBlur(image,(3,3),sigma)
            else:
                image = image
                
        if i==3: 
            image = gasuss_noise(image, mu=0.0, sigma=0.1)
        
        if i==4:
            m=random.randint(178,250)
            for i in range(m):
                x=random.randint(0,299)
                y=random.randint(0,299)
                image = cv2.rectangle(image, (x, y), (x+5, y+5), (0, 0, 0), -1)
                
    return image

def magnitude_phase_split(img):
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)
    phase_spectrum = np.angle(dft_shift)
    return magnitude_spectrum,phase_spectrum,dft_shift 


def magnitude_phase_combine(img_m,img_p):
    
    img_mandp = img_m*np.e**(1j*img_p)
    img_mandp = np.abs(np.fft.ifft2(img_mandp))
    return img_mandp


def normalization(img,newmin,newmax):
    image = ((img - np.min(img))*(newmax-newmin))/(np.max(img) - np.min(img))+newmin
    return image

def pro(temp,height,width,num):
    
    image=cv2.imread(temp[0])
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(height,width,),0,0,cv2.INTER_LINEAR)
    image = preprocessing(image,num)
    img = image/255
    
    #CSCD
    (r, g, b) = cv2.split(img)
    CDI_1 = r-g
    CDI_1 = np.expand_dims(CDI_1, axis=2)
    CDI_2 = b-g
    CDI_2 = np.expand_dims(CDI_2, axis=2)
    CDI_3 = r-b
    CDI_3 = np.expand_dims(CDI_3, axis=2)
    CDI = np.concatenate([CDI_1,CDI_2,CDI_3], axis=2)
    CDI=CDI.astype(np.float32)
    
    #LFS
    img_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_g = img_g/255
    img_g = cv2.GaussianBlur(img_g, (5, 5), 0)
    magnitude_spectrum,phase_spectrum,S=magnitude_phase_split(img_g)
    
    freq = np.log(1 +np.abs(S))
    freq=freq.astype(np.float32)
   
    return CDI,freq,temp[1]
    
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    def __iter__(self):
        return self
    def __next__(self): # python3
        with self.lock:
            return self.it.__next__()
      # def next(self): # python2
      #     with self.lock:
      #       return self.it.next()
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generate_data(path_fake,path_real,height=299,width=299,batch_size=20,num=[0]):
    D=[]
    label=[]

    for p in range(len(path_fake)):
        class_path_fake=path_fake[p]
        for fake_name in os.listdir(class_path_fake):
            fake_path=class_path_fake+fake_name
            D.append(fake_path)
            label.append(0)

    for p in range(len(path_real)):
        class_path_real=path_real[p]
        for real_name in os.listdir(class_path_real):
            real_path=class_path_real+real_name
            D.append(real_path)
            label.append(1)
        
    temp=np.array([D, label]) 
    temp=temp.transpose()
    total_num=temp.shape[0]
  
    while True:
        np.random.shuffle(temp)
        for index in range(0, total_num, (batch_size)):
            x_1=[]
            x_2=[]
            y_1=[]
            
            end = min(index + batch_size, total_num)
            for i in range(index, end):
                f = pro(temp[i], height, width, num)
                x_1.append(f[0])
                x_2.append(f[1])

                if f[2]=='0':
                    y_1.append([1,0])
                if f[2]=='1':
                    y_1.append([0,1])

            y = np.array(y_1)
            yield ([np.array(x_1),np.array(x_2)],np.array(y))
           

