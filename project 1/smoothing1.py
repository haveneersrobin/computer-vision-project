# -*- coding: utf-8 -*-
'''
Gaussian smoothing with Python.
'''

import cv2
import numpy as np
import math
import os

def gaussian_filter(sigma, filter_length=None):
    '''
    Given a sigma, return a 1-D Gaussian filter.
    @param     sigma:         float, defining the width of the filter
    @param     filter_length: optional, the length of the filter, has to be odd
    @return    A 1-D numpy array of odd length, 
               containing the symmetric, discrete approximation of a Gaussian with sigma
               Summation of the array-values must be equal to one.
    '''
    if filter_length==None:
        #determine the length of the filter
        filter_length= math.ceil(sigma*5) 
        #make the length odd
        filter_length= 2*(int(filter_length)/2) +1   
    
    #make sure sigma is a float
    sigma=float(sigma)
    
    #create the filter
    result = np.zeros( filter_length )
    
    #do your best!
    counter = 0
    for x in range(int(-math.floor(filter_length/2)), int(math.floor(filter_length/2))+1):
        result[counter] = (1/(math.sqrt(2*math.pi)*sigma))*math.e**((-x**2)/(2*sigma**2))
        counter = counter + 1
        
    arraysum = np.sum(result)
    for x in range(0, filter_length):
        result[x] = result[x]/arraysum
    
    #return the filter
    return result


def test_gaussian_filter():
    '''
    Test the Gaussian filter on a known input.
    '''
    sigma = math.sqrt(1.0/2/math.log(2))
    f = gaussian_filter(sigma, filter_length=3)
    print f
    correct_f = np.array([0.25, 0.5, 0.25])
    error = np.abs( f - correct_f)
    if np.sum(error)<0.001:
        print "Congratlations, the filter works!"
        cv2.waitKey(1000)

    else:
        print "Still some work to do.."
        
      

    
def gaussian_smooth1(img, sigma): 
    '''
    Do gaussian smoothing with sigma.
    Returns the smoothed image.
    '''
    result = np.zeros_like(img)
    #get the filtercv2.waitKey(0)
    filter = gaussian_filter(sigma)
    
    #smooth every color-channel
    for c in range(3):
        #smooth the 2D image img[:,:,c]
        #tip: make use of numpy.convolve
        shape = img[:,:,c].shape
        flattened_channel = img[:,:,c].flatten()
        convolved_channel = np.convolve(flattened_channel, filter,'same')
        convolved_channel.shape = shape
        result[:,:,c] = convolved_channel
    return result



#this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
    #test the gaussian filter
    #test_gaussian_filter()

    dir = os.path.dirname(__file__)

    #read an image
    img = cv2.imread(os.path.join(dir,'image.jpg'))
    
    #print the dimension of the image
    print img.shape
    
    #show the image, and wait for a key to be pressed
    cv2.imshow('img',img)
    cv2.waitKey(0)
    
    #smooth the image
    smoothed_img = gaussian_smooth1(img, 6)
    
    #show the smoothed image, and wait for a key to be pressed
    cv2.imshow('smoothed_img',smoothed_img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows() 
    