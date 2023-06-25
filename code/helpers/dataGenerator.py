# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 09:25:33 2022

@author: chris
"""

import numpy as np
import cv2
import random

#  imageSize(height, width)
def generateRandomlyOrientedLineImage(imageSize = (29, 29), noiseLevel = 0.1, lineThickness = 7):
  white = 255
  black = 0
  
  # Create an empty image with all pixels set to white (255)
  image = np.full(imageSize, white, dtype=np.uint8)
  
  angle = np.random.uniform(0, 360)
  lineLength = min(imageSize)
  
  # Calculate the start and end points of the line
  # Start in the center
  startRow = imageSize[0] // 2
  startColumn = imageSize[1] // 2
  #draw 1st half of the line
  endColumn = int(startColumn + lineLength/2 * np.cos(np.deg2rad(angle)))
  endRow = int(startRow + lineLength/2 * np.sin(np.deg2rad(angle)))
  #arg4 = color
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  #draw 2nd half of the line
  endColumn = int(startColumn - lineLength/2 * np.cos(np.deg2rad(angle)))
  endRow = int(startRow - lineLength/2 * np.sin(np.deg2rad(angle)))
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Flip pixelvalues randomly
  for row in range(imageSize[0]):
    for column in range(imageSize[1]):
      if random.random() < noiseLevel:
        if image[row,column] == white:
          image[row,column] = black
        else:
          image[row,column] = white
  
  # create circular mask
  ## Initialize mask with all zeros
  circularMask = np.full(imageSize, black, dtype=np.uint8)
  maskRadius = max(imageSize)/2
  ## Set the mask inside the radius to 1
  for row in range(imageSize[1]):
    for column in range(imageSize[0]):
      if (column - startRow)**2 + (row - startColumn)**2 <= maskRadius**2:
        circularMask[column, row] = 1
  image[circularMask == 0] = white
  
  return image, angle

#  imageSize(height, width)
def generateImage(angle, imageSize = (29, 29), noiseLevel = 0.1, lineThickness = 7):
  white = 255
  black = 0
  
  # Create an empty image with all pixels set to white (255)
  image = np.full(imageSize, white, dtype=np.uint8)
  
  lineLength = min(imageSize)
  
  # Calculate the start and end points of the line
  # Start in the center
  startRow = imageSize[0] // 2
  startColumn = imageSize[1] // 2
  #draw 1st half of the line
  endColumn = int(startColumn + lineLength/2 * np.cos(np.deg2rad(angle)))
  endRow = int(startRow + lineLength/2 * np.sin(np.deg2rad(angle)))
  #arg4 = color
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  #draw 2nd half of the line
  endColumn = int(startColumn - lineLength/2 * np.cos(np.deg2rad(angle)))
  endRow = int(startRow - lineLength/2 * np.sin(np.deg2rad(angle)))
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Flip pixelvalues randomly
  for row in range(imageSize[0]):
    for column in range(imageSize[1]):
      if random.random() < noiseLevel:
        if image[row,column] == white:
          image[row,column] = black
        else:
          image[row,column] = white
  
  # create circular mask
  ## Initialize mask with all zeros
  circularMask = np.full(imageSize, black, dtype=np.uint8)
  maskRadius = max(imageSize)/2
  ## Set the mask inside the radius to 1
  for row in range(imageSize[1]):
    for column in range(imageSize[0]):
      if (column - startRow)**2 + (row - startColumn)**2 <= maskRadius**2:
        circularMask[column, row] = 1
  image[circularMask == 0] = white
  
  return image, angle

#  imageSize(height, width)
def generateRandomVerticalLineImage(imageSize = (29, 29), noiseLevel = 0.1, lineThickness = 7, priorFlipChance = 0.1):
  white = 255
  black = 0
  
  # Create an empty image with all pixels set to white (255)
  image = np.full(imageSize, white, dtype=np.uint8)
  
  # random position of the line
  xPosition = int(np.random.uniform(0, imageSize[1]))
  
  # Calculate the start and end points of the line
  startRow = 0
  endRow = imageSize[0]
  startColumn = xPosition
  endColumn = xPosition
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Flip pixelvalues randomly
  for row in range(imageSize[0]):
    for column in range(imageSize[1]):
      if random.random() < noiseLevel:
        if image[row,column] == white:
          image[row,column] = black
        else:
          image[row,column] = white
  
  # Vertical Lines are represented by prior = 0
  prior = 0
  if random.random() < priorFlipChance:
    prior = 1
  
  return image, xPosition, prior, 0

def generateVerticalLineImage(xPosition, imageSize = (29, 29), noiseLevel = 0.1, lineThickness = 7):
  white = 255
  black = 0
  
  # Create an empty image with all pixels set to white (255)
  image = np.full(imageSize, white, dtype=np.uint8)
  
  # Calculate the start and end points of the line
  startRow = 0
  endRow = imageSize[0]
  startColumn = xPosition
  endColumn = xPosition
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Flip pixelvalues randomly
  for row in range(imageSize[0]):
    for column in range(imageSize[1]):
      if random.random() < noiseLevel:
        if image[row,column] == white:
          image[row,column] = black
        else:
          image[row,column] = white
  
  # Vertical Lines are represented by prior = 0
  prior = 0
  
  return image, xPosition, prior, 0
  
#  imageSize(height, width)
def generateRandomHorizontalLineImage(imageSize = (29, 29), noiseLevel = 0.1, lineThickness = 7, priorFlipChance = 0.1):
  white = 255
  black = 0
  
  # Create an empty image with all pixels set to white (255)
  image = np.full(imageSize, white, dtype=np.uint8)
  
  # random position of the line
  yPosition = int(np.random.uniform(0, imageSize[0]))
  
  # Calculate the start and end points of the line
  startColumn = 0
  endColumn = imageSize[1]
  startRow = yPosition
  endRow = yPosition
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Flip pixelvalues randomly
  for row in range(imageSize[0]):
    for column in range(imageSize[1]):
      if random.random() < noiseLevel:
        if image[row,column] == white:
          image[row,column] = black
        else:
          image[row,column] = white
          
  # horizontal Lines are represented by prior = 1
  prior = 1
  if random.random() < priorFlipChance:
    prior = 0
  
  return image, yPosition, prior, 1

#  imageSize(height, width)
def generateHorizontalLineImage(yPosition, imageSize = (29, 29), noiseLevel = 0.1, lineThickness = 7):
  white = 255
  black = 0
  
  # Create an empty image with all pixels set to white (255)
  image = np.full(imageSize, white, dtype=np.uint8)
  
  # Calculate the start and end points of the line
  startColumn = 0
  endColumn = imageSize[1]
  startRow = yPosition
  endRow = yPosition
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Flip pixelvalues randomly
  for row in range(imageSize[0]):
    for column in range(imageSize[1]):
      if random.random() < noiseLevel:
        if image[row,column] == white:
          image[row,column] = black
        else:
          image[row,column] = white
          
  # horizontal Lines are represented by prior = 1
  prior = 1

  return image, yPosition, prior, 1

#  imageSize(height, width)
# generates a cross and randomly chooses which orientation its supposed to represent
def generateRandomCrossLineImage(imageSize = (29, 29), noiseLevel = 0.1, lineThickness = 7, orientationFlipChance = 0.5):
  white = 255
  black = 0
  
  # Create an empty image with all pixels set to white (255)
  image = np.full(imageSize, white, dtype=np.uint8)
  
  # random position of the vertical line
  xPosition = int(np.random.uniform(0, imageSize[1]))
  # random position of the horizontal line
  yPosition = int(np.random.uniform(0, imageSize[0]))
  position = [xPosition, yPosition]
  
  # Calculate the start and end points of the vertical line
  startRow = 0
  endRow = imageSize[0]
  startColumn = xPosition
  endColumn = xPosition
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Calculate the start and end points of the horizontal line
  startColumn = 0
  endColumn = imageSize[1]
  startRow = yPosition
  endRow = yPosition
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Flip pixelvalues randomly
  for row in range(imageSize[0]):
    for column in range(imageSize[1]):
      if random.random() < noiseLevel:
        if image[row,column] == white:
          image[row,column] = black
        else:
          image[row,column] = white
  
  # Vertical Lines are represented by prior = 0
  prior = 0
  if random.random() < orientationFlipChance:
    prior = 1
  
  return image, position, prior

#  imageSize(height, width)
# generates a cross and user sets the prior class
# orientation = 0 = vertical
# orientation = 1 = horizontal
def generateCrossLineImage(orientation, imageSize = (29, 29), noiseLevel = 0.1, lineThickness = 7):
  white = 255
  black = 0
  
  # Create an empty image with all pixels set to white (255)
  image = np.full(imageSize, white, dtype=np.uint8)
  
  # random position of the vertical line
  xPosition = int(np.random.uniform(0, imageSize[1]))
  # random position of the horizontal line
  yPosition = int(np.random.uniform(0, imageSize[0]))
  
  # Calculate the start and end points of the vertical line
  startRow = 0
  endRow = imageSize[0]
  startColumn = xPosition
  endColumn = xPosition
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Calculate the start and end points of the horizontal line
  startColumn = 0
  endColumn = imageSize[1]
  startRow = yPosition
  endRow = yPosition
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Flip pixelvalues randomly
  for row in range(imageSize[0]):
    for column in range(imageSize[1]):
      if random.random() < noiseLevel:
        if image[row,column] == white:
          image[row,column] = black
        else:
          image[row,column] = white
  
  return image, xPosition, yPosition, orientation

#  imageSize(height, width)
# generates a cross and user sets the prior class
# orientation = 0 = vertical
# orientation = 1 = horizontal
def generateCrossLineImageAtPosition(orientation, xPosition, yPosition, imageSize = (29, 29), noiseLevel = 0.1, lineThickness = 7):
  white = 255
  black = 0
  
  # Create an empty image with all pixels set to white (255)
  image = np.full(imageSize, white, dtype=np.uint8)
  
  # Calculate the start and end points of the vertical line
  startRow = 0
  endRow = imageSize[0]
  startColumn = xPosition
  endColumn = xPosition
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Calculate the start and end points of the horizontal line
  startColumn = 0
  endColumn = imageSize[1]
  startRow = yPosition
  endRow = yPosition
  image = cv2.line(image, (startColumn, startRow), (endColumn, endRow), black, thickness=lineThickness)
  
  # Flip pixelvalues randomly
  for row in range(imageSize[0]):
    for column in range(imageSize[1]):
      if random.random() < noiseLevel:
        if image[row,column] == white:
          image[row,column] = black
        else:
          image[row,column] = white
  
  return image, xPosition, yPosition, orientation

#  imageSize(height, width)
def generateRandom1DLineImage(imageSize = (1, 9), noiseLevel = 0.1, priorFlipChance = 0.1):
  activePixels = 3 # not variable for now, as not needed. To be variable adapt 'position' and setting of active pixels
  possibleClasses = 4
  classes = [0, 1, 2, 3]
  positions = [1, 3, 5, 7] # center coords of the 4 classes
  
  white = 255
  black = 0
  
  # Create an empty image with all pixels set to white (255)
  image = np.full(imageSize, white, dtype=np.uint8)
  
  chosenClass = random.choice(classes)
  position = positions[chosenClass]
  
  # set the active pixels to black, according to position
  image[0, position] = 0
  image[0, position + 1] = 0
  image[0, position - 1] = 0

  # Flip pixelvalues randomly
  for row in range(imageSize[0]):
    for column in range(imageSize[1]):
      if random.random() < noiseLevel:
        if image[row,column] == white:
          image[row,column] = black
        else:
          image[row,column] = white
          
  # prior and its flip chance
  prior = chosenClass
  if random.random() < priorFlipChance:
    classes.remove(chosenClass)
    # each 'wrong' prior is equally likely
    prior = random.choice(classes)
  
  return image, prior
