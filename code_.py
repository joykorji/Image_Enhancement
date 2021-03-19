#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:21:28 2021

"""
import numpy as np
import matplotlib.pyplot as plt

#get the data from pgm file
def readpgm(name):
    with open(name) as f:
        lines = f.readlines()
    # This ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    # here, it makes sure it is ASCII greyscale format (P2)
    assert lines[0].strip() == 'P2' 
    # Converts data to a list of integers
    data = []
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])
    #the first two data points are the shape, the third is the max value
    #and the rest are the pixel values
    return (np.array(data[3:]),(data[1],data[0]),data[2])

#get from the user which image will be analyzed
def get_image():
   
    print("Which image would you like to analyze? Enter 1 for the building image, 2 for the MRI and 3 for the peppers")
    image = input()
    pgm = ()
    if image == '1':
        pgm = readpgm('Building.pgm')
    elif image == '2':
        pgm = readpgm('MRI.pgm')
    elif image == '3':
        pgm = readpgm('peppers.pgm')
    else:
        print("Invalid input, enter either 1, 2 or 3")
        get_image()
    return pgm,image

#allows the user to apply any combination of approaches to enhance the image
def enhance(image,name):
    _vertical = False
    _horizontal = False
    plt.figure()
    plt.title("Original Image")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    print("Now we will be enhancing this image. do you want to apply histogram stretching? yes/no")
    apply_stretching = input()
    if apply_stretching == 'yes':
        stretching(image)
        plt.figure()
        plt.title("Image after histogram stretching")
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        
    print("Do you want to apply Power law transformation ? yes/no ")
    apply_power_law = input()
    if apply_power_law == 'yes':
        if name == '1':
            power_transformation(1,3,image)
        elif name == '2':
            power_transformation(1,0.6,image)
        elif name == '3':
            power_transformation(1,1.2,image)
        plt.figure()
        plt.title("Image after power law transformation")
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        
    print("Do you want to apply convolution ? yes/no ")
    apply_convolution = input()
    if apply_convolution == 'yes':
        print("Enter the size of your mask (enter an odd integer, for example 3 for a 3 by 3 mask). The bigger the mask, the blurrier the image will be")
        size = int(input())
        print("Enter 1 for a mask of 1s, 2 for a gaussian mask")
        if input() == '1':
            image = convolution(image, size)
        else:
            image = gaussian_convolution(image, size)
        plt.figure()
        plt.title("Image after convolution")
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    
    print("Median filter is good when you have salt and pepper noise, do you want to apply it ? yes/no ")
    apply_median_filter = input()
    if apply_median_filter == 'yes':
        image = median_filter(image)
        plt.figure()
        plt.title("Image after median filter")
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    
    print("Now let's detect some edges, do you want to apply horizontal detection ? yes/no ")
    apply_prewitt_edge_horizontal = input()
    if apply_prewitt_edge_horizontal == 'yes':
        _horizontal = True
        print("enter 1 for prewitt filter and 2 for sobel filter")
        apply_prewitt_or_sobel = input()
        if apply_prewitt_or_sobel == '1':
            horizontal_image = image.copy()
            horizontal_image = prewitt_edge_horizontal(horizontal_image)                       
        elif apply_prewitt_or_sobel == '2':
            horizontal_image = image.copy()
            horizontal_image = sobel_edge_horizontal(horizontal_image)      
        plt.figure()
        plt.title("Image after horizontal edge detection")
        plt.imshow(horizontal_image, cmap='gray', vmin=0, vmax=255)
            
            
    print("Do you want to apply vertical detection ? yes/no ")
    apply_prewitt_edge_vertical = input()
    if apply_prewitt_edge_vertical == 'yes':
        _vertical = True
        print("enter 1 for prewitt filter and 2 for sobel filter")
        apply_prewitt_or_sobel = input()
        if apply_prewitt_or_sobel == '1':
            vertical_image = image.copy()
            vertical_image = prewitt_edge_vertical(vertical_image)
        elif apply_prewitt_or_sobel == '2': 
            vertical_image = image.copy()
            vertical_image = sobel_edge_vertical(vertical_image)
        plt.figure()
        plt.title("Image after vertical edge detection")
        plt.imshow(vertical_image, cmap='gray', vmin=0, vmax=255)
       
    if(_horizontal and _vertical):
        print("congratulation, you made it this far, our last question is do you want to combine horizontal and vertical edge filters ? yes/no ")
        combine_filters = input()
        if combine_filters == 'yes': 
            combined_image = vertical_image + horizontal_image
            stretching(combined_image)
            plt.figure()
            plt.title("Image after combined edge detection")
            plt.imshow(combined_image, cmap='gray', vmin=0, vmax=255)
            

'''
Part 1:
    Histogram stretching, power or log transformation,
    linear (convolution) and non-linear filtering (median filter)
'''

#Histogram stretching, resets range of values to 0 to 255
def stretching(image):
    #y = ax+b
    mini = image.min()
    maxi = image.max()
    a = 255/(maxi-mini).astype(int)
    b = 255 - (a*maxi).astype(int)
    size_x, size_y = image.shape
    for i in range(0, size_x):
        for j in range(0, size_y):
            image[i][j] = a*image[i][j]+b
    return image


def power_transformation(c, gamma, image):
    size_x, size_y = image.shape
    for i in range(0, size_x):
        for j in range(0, size_y):
            image[i][j] = pow(image[i][j], gamma) * c
    image = stretching(image)
    return image


#convolution with a mask of one's
def convolution(image, size):
    size_x, size_y = image.shape
    new_image = image.copy()
    interval = int((size-1)/2)
    for i in range(interval, size_x-interval):
        for j in range(interval, size_y-interval):
            total = 0
            for k in range (i-interval, i+interval+1):
                for l in range (j-interval, j+interval+1):
                    total += image[k][l]
            new_image[i][j] = total/(size*size)
    new_image = stretching(new_image)
    return new_image


#get a value for the gaussian mask
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

#produces a gaussian mask
def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / kernel_2D.max()
    #normalize the mask
    kernel_2D = kernel_2D / np.sum(kernel_2D) 
    return kernel_2D

#Convolution with gaussian mask
def gaussian_convolution(image, size):
    shape_x, shape_y = image.shape #image shape
    mask = gaussian_kernel(size)
    interval = int((size-1)/2)
    new_image = image.copy()
    for i in range (interval, shape_x - interval):
        for j in range(interval, shape_y - interval):
            total = 0
            #apply mask
            for k in range (-interval, interval+1):
                for l in range (-interval, interval+1):
                    total += image[i+k][j+l] * mask[k+interval][l+interval]
            new_image[i][j] = total
    stretching(new_image)
    return(new_image)
        


def median_filter(image):
    size_x, size_y = image.shape
    new_image = image.copy()
    for i in range(1, size_x-1):
        for j in range(1, size_y-1):
            values = []
            for k in range (i-1, i+2):
                for l in range (j-1, j+2):
                   values.append(image[k][l])
            new_image[i][j] = np.median(values)
    new_image = stretching(new_image)
    return new_image


'''
Part 2:
    Edge detectors
'''
#horizontal edge detection and vertical smoothing uses the prewitt filter
def prewitt_edge_horizontal(image):
    size_x, size_y = image.shape
    new_image = image.copy()
    for i in range(1, size_x-1):
        for j in range(1, size_y-1):
            total = 0
            for k in range (-1, 2):
                for l in range (j-1, j+2):
                    total += k*image[i+k][l]
            new_image[i][j] = total
    new_image = stretching(new_image)
    return new_image

#vertical edge detection and horizontal smoothing using the prewitt filter
def prewitt_edge_vertical(image):
    size_x, size_y = image.shape
    new_image = image.copy()
    for i in range(1, size_x-1):
        for j in range(1, size_y-1):
            total = 0
            for k in range (i-1, i+2):
                for l in range (-1, 2):
                    total += l*image[k][j+l]
            new_image[i][j] = total
    new_image = stretching(new_image)
    return new_image


#horizontal edge detection and vertical smoothing uses the sobel filter
def sobel_edge_horizontal(image):
    size_x, size_y = image.shape
    new_image = image.copy()
    for i in range(1, size_x-1):
        for j in range(1, size_y-1):
            total = 0
            for k in range (-1, 2):
                for l in range (-1, 2):
                    if l == 0:
                        total += k*2*image[i+k][j+l]
                    else:
                        total += k*image[i+k][j+l]
            new_image[i][j] = total
    new_image = stretching(new_image)
    return new_image

#vertical edge detection and horizontal smoothing using the sobel filter
def sobel_edge_vertical(image):
    size_x, size_y = image.shape
    new_image = image.copy()
    for i in range(1, size_x-1):
        for j in range(1, size_y-1):
            total = 0
            for k in range (-1, 2):
                for l in range (-1, 2):
                    if k == 0:
                        total += 2*l*image[i+k][j+l]
                    else:
                        total += l*image[i+k][j+l]
            new_image[i][j] = total
    new_image = stretching(new_image)
    return new_image

def main():
    pgm, name = get_image()
    enhance(np.reshape(pgm[0],pgm[1]), name)
    
main()   
    
'''pgm2 = readpgm('MRI.pgm')
data2 = np.reshape(pgm2[0],pgm2[1])
plt.figure()
plt.imshow(data2, cmap='gray', vmin=0, vmax=255)

image2 = data2.copy()
print(np.unique(image2))
image2 = stretching(image2)
print(np.unique(image2))
plt.figure()
plt.imshow(image2, cmap='gray', vmin=0, vmax=255)



pgm3 = readpgm('peppers.pgm')
data3 = np.reshape(pgm3[0],pgm3[1])
plt.figure()
plt.imshow(data3, cmap='gray')
image3 = data3.copy()
image3 = convolution(image3)
plt.figure()
plt.imshow(image3, cmap='gray', vmin=0, vmax=255)

image3 = data3.copy()
image3 = convolution(image3, 5)
plt.figure()
plt.imshow(image3, cmap='gray', vmin=0, vmax=255)
'''

