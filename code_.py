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


#reshape image as a 2D array
data1 = np.reshape(pgm1[0],pgm1[1])
plt.figure()
plt.imshow(data1, cmap='gray')

data2 = np.reshape(pgm2[0],pgm2[1])
plt.figure()
plt.imshow(data2, cmap='gray')

data3 = np.reshape(pgm3[0],pgm3[1])
plt.figure()
plt.imshow(data3, cmap='gray')

'''
Part 1:
    Implement Histogram stretching, power or log transformation,
    linear (convolution) and non-linear filtering (median filter)
'''

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

image1 = data1.copy()
plt.figure()
plt.imshow(image1, cmap='gray')
image1 = stretching(image1)
plt.figure()
plt.imshow(image1, cmap='gray')

image2 = data2.copy()
plt.figure()
plt.imshow(image2, cmap='gray')
image2 = stretching(image2)
plt.figure()
plt.imshow(image2, cmap='gray')

image3 = data3.copy()
plt.figure()
plt.imshow(image3, cmap='gray')
image3 = stretching(image3)
plt.figure()
plt.imshow(image3, cmap='gray')
    
def power_transformation(c, gamma, image):
    size_x, size_y = image.shape
    for i in range(0, size_x):
        for j in range(0, size_y):
            image[i][j] = pow(image[i][j], gamma) * c
    return image

image1 = data1.copy()
image1 = power_transformation(1, 3, image1)
plt.figure()
plt.imshow(image1, cmap='gray')
image1 = stretching(image1)
plt.figure()
plt.imshow(image1, cmap='gray')


def convolution(image):
    #mask = np.array([[1,1,1], [1,1,1], [1,1,1]])
    size_x, size_y = image.shape
    new_image = image.copy()
    for i in range(1, size_x-1):
        for j in range(1, size_y-1):
            total = 0
            for k in range (i-1, i+2):
                for l in range (j-1, j+2):
                    total += image[k][l]
            new_image[i][j] = total/9
    return new_image

image3 = data3.copy()
image3 = convolution(image3)
plt.imshow(image3, cmap='gray')

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
    return new_image

image2 = data2.copy()
image2 = convolution(image2)
plt.imshow(image2, cmap='gray')

'''
Part 2:
    Provide a selection of edge detectors
    For the display, generate the magnitude of the gradient
'''
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
    return new_image

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
    return new_image

image1 = data1.copy()
image1 = power_transformation(1, 3, image1)
image1 = stretching(image1)
plt.figure()
plt.imshow(image1, cmap='gray')

image1_horizontal = image1.copy()
image1_horizontal = prewitt_edge_horizontal(image1_horizontal)
plt.figure()
plt.imshow(image1_horizontal, cmap='gray')
image1_horizontal = stretching(image1_horizontal)

image1_vertical = image1.copy()
image1_vertical = prewitt_edge_vertical(image1_vertical)
plt.figure()
plt.imshow(image1_vertical, cmap='gray')
image1_vertical = stretching(image1_vertical)

image1_edges = image1_horizontal + image1_vertical
image1_edges = stretching(image1_edges)
plt.figure()
plt.imshow(image1_edges, cmap='gray')

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
    return new_image

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
    return new_image

image1 = data1.copy()
image1 = power_transformation(1, 3, image1)
image1 = stretching(image1)
plt.figure()
plt.imshow(image1, cmap='gray')

image1_horizontal = image1.copy()
image1_horizontal = sobel_edge_horizontal(image1_horizontal)
plt.figure()
plt.imshow(image1_horizontal, cmap='gray')
image1_horizontal = stretching(image1_horizontal)

image1_vertical = image1.copy()
image1_vertical = sobel_edge_vertical(image1_vertical)
plt.figure()
plt.imshow(image1_vertical, cmap='gray')
image1_vertical = stretching(image1_vertical)

image1_edges = image1_horizontal + image1_vertical
image1_edges = stretching(image1_edges)
plt.figure()
plt.imshow(image1_edges, cmap='gray')
