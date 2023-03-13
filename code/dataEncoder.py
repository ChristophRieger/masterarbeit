# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 10:51:59 2022

@author: chris
"""

# creates a list with 2 values for each pixel of the image
# for white pixels 1, 0 is appended to the list
# for black pixels 0, 1 is appended to the list
def encodeImage(image):
  
  # in nessler paper in ex1 they say they encode 2D -> 1D randomly, i will do it non random for now
  
  encodedImage = []
  
  for row in range(image.shape[0]):
    for column in range(image.shape[1]):
      if(image[row, column] == 255):
        encodedImage.append(1)
        encodedImage.append(0)
      else:
        encodedImage.append(0)
        encodedImage.append(1)
  return encodedImage