import imutils
import cv2 as cv


# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)
image = input(str('Enter the path of images: '))
image = cv.imread(image)
image = imutils.resize(image, width=600) # Resize the image
(h, w, d) = image.shape
print("Width={}, Height={}, Depth={}".format(w,h,d))


gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # convert the image to grayscale
gray = cv.GaussianBlur(gray, (15,15),0) #Smoothing an image
edge = cv.Canny(gray, 30,105)  # applying edge detection we can find the outlines of objects in images
threshold = cv.threshold(gray, 135, 255, cv.THRESH_BINARY)[1]

# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
mask = threshold.copy()
mask = cv.dilate(mask, None)
mask_bit = cv.bitwise_and(image, image, mask=mask)

# display the image to our screen
cv.imshow('Mask', mask)
cv.imshow('Original', image)
cv.imshow('Gray', gray)
cv.imshow('output', mask_bit)
#cv.imshow('Threshold', threshold)
#cv.imshow('Edge', edge)


# find contours (i.e., outlines) of the foreground objects in the thresholded image
cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
	cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

output = image.copy()
 
# loop over the contours
for c in cnts:
	# draw each contour on the output image with a 2px thick red
	# outline, then display the output contours one at a time
	cv.drawContours(output, [c], -1, (0, 0, 215), 2)
	cv.imshow("Contours", output)
	cv.waitKey(0)


# draw the total number of contours found in red
text = "ships detected: {}".format(len(cnts))
cv.putText(output, text, (10, 25),  cv.FONT_HERSHEY_SIMPLEX, 0.8,
	(0,255,0), 1)
cv.imshow("Contours", output)
cv.waitKey(0)
