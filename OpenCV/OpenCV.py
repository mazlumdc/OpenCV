import cv2
import numpy as np




#######################################################################################################################
#INTRODUCTION TO OpenCV
#######################################################################################################################

# Read image
image = cv2.imread("./images/lena.jpeg")

# Image properties
print("Type of image: ",type(image))
print("Data type of image: ",image.dtype) # why uint8 ?-> https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html#imshow


print(image.shape)
print(image.shape[1])
print(image.shape[2])
print(image.shape[3])


# Reach to Pixel Vectors
b,g,r = cv2.split(image)

b = image[:,:,0]
g = image[:,:,1]
r = image[:,:,2]


'''
The gray color tone has a special case because a gray color image is stored in two dimensions. function to upload images 
'cv2.imread()' By default, it loads 3D, but there is no need to do anything while uploading gray images. Additionally, 
it is possible to make a loaded color image gray, even tints can be made in different color spaces, not just gray.
'''

# Conversion to grayscale
print("Original image shape: ",image.shape)
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
print("Grayscale image shape: ",gray_image.shape)
gray_image = cv2.imread("./images/rgb_image.jpeg",cv2.IMREAD_GRAYSCALE)


# Image Visualization
cv2.imshow("blue channel",b)
cv2.imshow("image",image)
cv2.imshow("gray image",gray_image)
cv2.waitKey(0)




#######################################################################################################################
# MANIPULATING PIXEL VALUES (Accessing and changing pixels of an image)
#######################################################################################################################

# Let's create a picture ourselves
img = np.zeros(shape=(500,500,3),dtype=np.uint8)

cv2.imshow("img",img)
cv2.waitKey(0)

# Change the hue of the pixel
img[50,50] = (255,0,0) #just change blue channel
img[450,250] = (255,255,255)

cv2.imshow("img",img)
cv2.waitKey(0)

img_grayscale = np.full(shape=(500,500),fill_value=255,dtype=np.uint8)
img_grayscale[50,50] = 0


cv2.imshow("Grayscale",img_grayscale)
cv2.waitKey(0)




#######################################################################################################################
# READING VIDEO WITH THE OpenCV
#######################################################################################################################

cap = cv2.VideoCapture("./images/spot.mp4")

while True:
    ret,frame = cap.read()

    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # if you want to make it gray

    cv2.imshow("video",frame)
    key = cv2.waitKey(30)
    if key == ord('q'):
        break

cap.release() #'Deletes the cap variable so that it does not take up more space in memory'
cv2.destroyAllWindows() #'destroys the open window'




#######################################################################################################################
# Resizing the image
#######################################################################################################################

"""
Types of interpolation in OpenCV:

- INTER_NEAREST - a nearest-neighbor interpolation
- INTER_LINEAR - a bilinear interpolation (used by default)
- INTER_AREA - Resampling using pixel area relation. It may be a preferred method for image decimation. 
                But when the image is zoomed, it is similar to the INTER_NEAREST method.              
- INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
- INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

"""

img = cv2.imread("./images/lena.jpeg")
print("Original image shape: ",img.shape)

new_img = cv2.resize(img,dsize=(200,200),interpolation=cv2.INTER_LINEAR) # interpolation
print("Shape after resizing: ",img.shape)

'''
Attention! , the first value of img.shape gives the height of the image 
but the first value written in resize() is the width in the new resize.
'''

cv2.imshow("Img",img)
cv2.imshow("New img",new_img)
cv2.waitKey(0)




#######################################################################################################################
# ADDING DIFFERENT SHAPES AND TEXTS TO THE PICTURE
#######################################################################################################################

blank = np.zeros((500,500,3),dtype=np.uint8)
cv2.imshow('Blank',blank)

# point image a certain color
blank[0:100,0:100] = 0,255,0 # Paint an area 100x100

# draw rectangle
cv2.rectangle(blank,(50,50),(100,100),(0,0,255),1) # trt with thickness=-1

# draw circle
cv2.circle(blank,(255,250),50,(0,0,255),1)

# draw line
cv2.line(blank,(0,0),(300,300),(0,0,255),2)

# put text
cv2.putText(blank,'ESTU',(70,70),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),1)


cv2.imshow("Green",blank)
cv2.waitKey(0)




#######################################################################################################################
# THRESHOLD
#######################################################################################################################

'It is the process of dyeing values higher or lower than a specified threshold value into different colors.'


'Check here for more information.'
# https://docs.opencv.org/3.4/db/d8e/tutorial_threshold.html


img = cv2.imread("./images/coins.jpg")
cv2.imshow("img",img)

#It is useful to make the picture gray before applying.
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)


# simple thresholding
threshold,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
threshold,thresh_inv = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)

cv2.imshow("thresh invert",thresh_inv)
cv2.imshow("thresh",thresh)
cv2.waitKey(0)




#######################################################################################################################
# Why we use hsv color space?
#######################################################################################################################

img = cv2.imread("./images/red_car.jpg")
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower1 = np.array([161, 155, 84])
upper1 = np.array([179, 255, 255])

lower2 = np.array([0,50,50])
upper2 = np.array([10,255,255])

red_mask1 = cv2.inRange(hsv_img, lower1, upper1)
red_mask2 = cv2.inRange(hsv_img,lower2,upper2)

red_mask = red_mask1 + red_mask2
red = cv2.bitwise_and(img, img, mask=red_mask)

cv2.imshow("red_mask",red_mask)
cv2.imshow("red",red)
cv2.imshow("bgr",img)
cv2.imshow("hsv",hsv_img)
cv2.waitKey(0)
