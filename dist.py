import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2




def process_image(image):
    image = np.array(image)
    #ref_point = [(0, 300), (1072, 720)]
    #plt.imshow(image,  cmap="hot")
    #plt.show()
    #image = image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
    #cv2.imshow('Lane lines', image)
    #cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    #---------CANNY EDGE DETECTION-----------------------------------------------
    # The algorithm will first detect strong edge (strong gradient) pixels above 
    # the high_threshold, and reject pixels below the low_threshold. 
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    print (edges.shape)


    #Defining ROI aka Masked edges
    mask = np.zeros_like(edges)
    ignore_mask_color = 200   

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    top_left = (0, 400)
    top_right = (1000, 400)
    bottom_left = (0,imshape[0]) #(50, 539)
    bottom_right = (imshape[1], imshape[0])

    # vertices = np.array([[bottom_left,top_left, top_right, bottom_right]], dtype=np.int32)
    vertices = np.array([[bottom_left,top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    #-------------HOUGH TRANSFORMATION----------------------------------------------
    # Defining the Hough transform parameters
    rho = 2 #1
    theta = np.pi/180
    threshold = 20 #1
    min_line_length = 20 #5
    max_line_gap = 10 #1
    line_image = np.copy(image)*0 #creating a blank to draw lines on

    # Run Hough on edge detected image
    # More Rho pixels -> More unstraight lines will be detected
    #More threshold -> less lines detected
    # more min_line_length -> more short noises discarded
    # max_line_gap -> for connecting the lines
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    return lines
    # Iterate over the output "lines" and draw lines on the blank

    # # Display the image
    # plt.imshow(edges, cmap='Greys_r')
    # plt.show()
    # if cv2.waitKey(1) & 0xff == ord('q'):
    #     break

