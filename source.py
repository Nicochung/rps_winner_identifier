import os
import sys
import cv2
import imutils
import math
import numpy as np
import scipy.ndimage as ndi
# Fixed size from the image
height = 480
width = 640
handBox = []
## Helper function to read in the image and answer
def _massiveRead(filePath,filetype):
    # Hardcode filepath, change according to needs
    itemList = os.listdir(r'/Users/nicochung/desktop/cvProject/test_data'+str(filePath))
    # Find all file ends with 'filetype' and return the list of the name of file
    for i in range(len(itemList)):
        pos = itemList[i].find(filetype)
        if pos is -1:
            itemList[i] = ''
    itemList.sort()
    while itemList[0] is '':
        itemList.pop(0)
    return itemList

## Helper function to color the label component
def _imshow_components(labels,name):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    # set bg label to black
    labeled_img[label_hue==0] = 0
    cv2.imwrite(name, labeled_img)
    #cv.imshow("Labelled img", labeled_img)

## Helper function to threshold the label and color them
def _threshColor(img,name):
    num_labels, labels_im = cv2.connectedComponents(img)
    # Create temp image for storing the thresholded label
    mask = np.zeros((height,width), dtype="uint8")
    for (i, label) in enumerate(np.unique(labels_im)):
            if label == 0:
                continue
            labelMask = np.zeros((height,width), dtype="uint8")
            labelMask[labels_im == label] = label
            numPixels = cv2.countNonZero(labelMask)
            if numPixels > 3750:
                # Pass threshold and add to the mask to coloring
                mask = cv2.add(mask, labelMask)
    # Call the helper function for coloring the label
    _imshow_components(mask,name)

## Helper function to check local peaks
def _checkLocalPeak(localmax,localmin,ft1,ft):
    shortPeaks = 0
    height =localmin+(localmax-localmin)*0.4
    # Pass the threshhold and create peaks
    for i in range(len(ft1)):
        if ft1[i] < height:
            ft1[i] = -1
    # Check for finger like peaks
    angleTestProcess = False
    for i in range(len(ft1)):
        if ft1[i] > 0 and angleTestProcess is False:
            startAngle = ft[i]
            starti = i
            angleTestProcess = True
        elif ft1[i] < 0 and angleTestProcess is True:
            if ft[i-1] - startAngle > 0.15:
                shortPeaks += 1
            angleTestProcess = False
    return shortPeaks

## Helper function to convert angle so that the image will be rotated to pointing up
def _angleConvert(dy,dx,angle):
    # both dx+ dy+
    if dx > 0 and dy > 0:
        angle = 90-angle
    # both dx- dy-
    elif dx < 0 and dy < 0:
        angle = abs(angle)+90
    # both dx+ dy-
    elif dx > 0 and dy < 0:
        angle = abs(angle)+90
    # both dx- dy+
    elif dx < 0 and dy > 0:
        angle = -angle+90
    elif dx is 0 and dy >0:
        angle = 0
    elif dx is 0 and dy < 0:
        angle = 180
    elif dx > 0 and dy is 0:
        angle = 90
    elif dx < 0 and dy is 0:
        angle = 270
    return angle

## Helper function to clip and rotate the mask from the binary image
def _clippedAndRotate(thresh):
    # Two blur to smooth the image
    thresh = cv2.blur(thresh,(5,5),0)
    thresh = cv2.blur(thresh,(5,5),0)
    # For drawing the bounding box and clipping
    segmented, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y = [], []
    for contour_line in contours:
        for contour in contour_line:
            x.append(contour[0][0])
            y.append(contour[0][1])
    x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
    clipped = segmented[y1:y2, x1:x2]

    # Rotate the image around centre
    M = cv2.moments(thresh)
    # calculate centre of mass the bw img
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    dx = cX-320
    dy = cY-240
    angle0 = math.degrees(math.atan2(dy, dx))
    angle0 = _angleConvert(dy,dx,angle0)

    rotated = imutils.rotate_bound(clipped, angle0)
    # Convert the distance from centre to edge to polar coordinate

    # Find the centre in the rotated img
    segmented, contours, hierarchy = cv2.findContours(rotated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y = [], []
    for contour_line in contours:
        for contour in contour_line:
            x.append(contour[0][0])
            y.append(contour[0][1])
    M = cv2.moments(rotated)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(rotated,(cX,cY), 2, (0,0,0), 3)

    # Convert to polar coordinate
    feature = []
    rmax = 0
    rmin = 10000
    for i in range(len(x)):
        dx = x[i]-cX
        dy = y[i]-cY
        r = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy,dx)
        feature.append([angle,r])
        rmax = max(r,rmax)
        rmin = min(r,rmin)
    feature.sort()
    ft = np.transpose(feature)
    # Smooth the curve
    ft1=ndi.convolve1d(ft[1], [1/3.0, 1/3.0, 1/3.0], output=np.float64, mode='nearest')

    # Too round as a shape,most likely be a rock, too thin and long, most likely be a scissors
    if max(ft1)-min(ft1) <= 70 and max(ft1) <= 80:
        global handBox
        # i is in order xmin ymin xmax ymax geseture
        handBox.append([x1,y1,x2,y2,"Rock"])
        return
    if max(ft1)-min(ft1) >=100 and min(ft1) <= 20:
        handBox.append([x1,y1,x2,y2,"Scissors"])
        return

    # thresholing the fingers and try to remove bounces
    holdAngleSize = math.pi*7/180
    height=rmin+(rmax-rmin)*0.5
    startAngle = 0
    starti = 0
    shortPeaks = 0
    localPeaks = 0
    numPeaks = 0
    ## See how long it hold
    ## Count number of peaks
    angleTestProcess = False
    for i in range(len(ft1)):
        # Pass the threshhold and create peaks
        if ft1[i] < height:
            ft1[i] = -1
        if ft1[i] > 0 and angleTestProcess is False:
            startAngle = ft[0][i]
            starti = i
            angleTestProcess = True
        elif ft1[i] < 0 and angleTestProcess is True:
            # Check how wide is the angle, if larger than threshold we keep it, otherwise make them all -1
            if holdAngleSize > ft[0][i-1] - startAngle:
                for j in range(starti,i):
                    ft1[j] = -1
            angleTestProcess = False
    ## Check for finger like peaks and consective peaks
    localmax = 0
    localmin = 1000
    angleTestProcess = False
    for i in range(len(ft1)):
        if ft1[i] > 0:
            if angleTestProcess is False:
                startAngle = ft[0][i]
                starti = i
                angleTestProcess = True
            else:
                localmax = max(localmax,ft1[i])
                localmin = min(localmin,ft1[i])
        elif ft1[i] < 0 and angleTestProcess is True:
            # Check how wide is the angle, if it is small, it is a shortpeak, otherwise check for consective peaks
            if ft[0][i-1] - startAngle < 0.3:
                shortPeaks += 1
            elif ft[0][i-1] - startAngle < 0.75:
                localcopy = ft1[starti:i]
                localcopy2 = ft[0][starti:i]
                localPeaks += _checkLocalPeak(localmax,localmin,localcopy,localcopy2)
            localmax = 0
            localmin = 1000
            angleTestProcess = False

    # Gesture decision
    numPeaks = shortPeaks+localPeaks
    gesture = ''
    if numPeaks <= 0:
        gesture = "Rock"
    elif numPeaks >= 3:
        gesture = "Paper"
    else:
        gesture = "Scissors"
    # i is in order xmin ymin xmax ymax geseture
    handBox.append([x1,y1,x2,y2,str(gesture)])

## Helper function to detect skin color like pixels
def _checkSkinSim(img):
    # Convert BGR to YCrCb
    (B, G, R) = cv2.split(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (y, cr, cb) = cv2.split(ycrcb)
    for i in range(height):
        for j in range(width):
            if cb[i,j] < 127 and cb[i,j] >= 77 and cr[i,j] < 173 and cr[i,j] > 133:
                B[i,j] = 255
            else:
                B[i,j] = 0
    # Return only one channel as binary image
    return B

## Helper function to threshold the label and sent them next clipping and rotation stage
def _threshBox(img):
    num_labels, labels_im = cv2.connectedComponents(img)
    for (i, label) in enumerate(np.unique(labels_im)):
            if label == 0:
                continue
            labelMask = np.zeros((height,width), dtype="uint8")
            labelMask[labels_im == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            if numPixels > 3750:
                im_floodfill = labelMask.copy()
                # Mask used to flood filling.
                # Notice the size needs to be 2 pixels than the image.
                h, w = labelMask.shape[:2]
                mask = np.zeros((h+2, w+2), np.uint8)
                # Floodfill from point (0, 0)
                cv2.floodFill(im_floodfill, mask, (0,0), 255);
                # Invert floodfilled image
                im_floodfill_inv = cv2.bitwise_not(im_floodfill)
                # Combine the two images to get the foreground.
                im_out = labelMask | im_floodfill_inv
                _clippedAndRotate(im_out)

## Helper function to determine the winning gesture base on boolean flags
def _winner(rock,paper,scissors):
    winner = ''
    if rock is True and paper is True and scissors is False:
        winner = "Paper"
    elif rock is False and paper is True and scissors is True:
        winner = "Scissors"
    elif rock is True and paper is False and scissors is True:
        winner = "Rock"
    else:
        winner = "Draw"
    return winner

## Helper function to do a laplacian filtering/Sharpening
def _laplacian(src):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
    sharp = np.float32(src)
    imgResult = sharp - imgLaplacian
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    return imgResult

## Helper function to do Morphological Transformations
def _openDilate(bw):
        kernel = np.ones((3,3), dtype=np.uint8)
        opening = cv2.morphologyEx(bw,cv2.MORPH_OPEN,kernel, iterations = 2)
        final = cv2.dilate(opening,kernel,iterations=2)
        return final

## main function
if __name__ == '__main__':
    # filepath for level1 or level2
    filepath = '/level2'
    # nameList and ansList are hardcoded path, check _massiveRead(), change the path if needed
    nameList = _massiveRead(filepath,'.ppm')
    ansList = _massiveRead(filepath,'.tr')
    # get the current directory and go the directory with the image files and answer files
    cwd = os.getcwd()
    os.chdir(cwd+filepath)
    # For each image and answer file
    for i in range(len(nameList)):
        # Clear the list for storing coordinate of box and gesture
        handBox.clear()
        imname = nameList[i]
        ansname = ansList[i]
        name = imname[:5]
        name1 = ansname[:5]
        # Show the current image name
        print(imname)
        # Make sure the image and answer file are matched
        if name != name1:
            print("File name not match, break")
            break

        src = cv2.imread(imname, cv2.IMREAD_COLOR)
        imgResult = _laplacian(src)
        bw = _checkSkinSim(imgResult)
        morphBW = _openDilate(bw)

        # Create name for output image
        name = imname[:5]+'_Result.jpg'
        # Only uncomment _threshColor when want to see all the labelled mask together
        #_threshColor(morphBW,name)

        # Call the threshold function for labelling
        _threshBox(morphBW)

        # Winner decision
        paper = False
        rock = False
        scissors = False
        # Put box and text for different gesture with unique color
        for i in handBox :
            if i[4] is "Paper":
                color = (0,255,0)
                paper = True
            elif i[4] is "Rock":
                color = (255,0,0)
                rock = True
            else:
                color = (0,0,255)
                scissors = True
            # i is in order xmin ymin xmax ymax geseture
            cv2.rectangle(src,(i[0],i[1]),(i[2],i[3]),color,3)
            cv2.putText(src, i[4], (i[0],(i[1]+i[3])//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            #print(i[0]," ",i[1]," ",i[2]," ",i[3]," ",i[4])
        winner = _winner(rock,paper,scissors)
        if winner is "Draw":
            for i in handBox:
                if i[4] is "Paper":
                    color = (0,255,0)
                elif i[4] is "Rock":
                    color = (255,0,0)
                else:
                    color = (0,0,255)
                cv2.putText(src, "Draw", (i[0],(i[1]+i[3])//2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        else:
            for i in handBox:
                if i[4] is "Paper":
                    color = (0,255,0)
                elif i[4] is "Rock":
                    color = (255,0,0)
                else:
                    color = (0,0,255)
                if i[4] is winner:
                    cv2.putText(src, "Win", (i[0],(i[1]+i[3])//2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                else:
                    cv2.putText(src, "Lose", (i[0],(i[1]+i[3])//2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Put answer box and text with pink color, comment the following when answer is not require
        f = open(ansname, "r")
        color = (204,153,255)
        while True:
            line = f.readline()
            if not line:
                break
            item = line.strip()
            x = item.split()
            # In order of Xmin Ymin, Xlen Ylen Result
            cv2.rectangle(src,(int(x[0]),int(x[1])),(int(x[0])+int(x[2]),int(x[1])+int(x[3])),color,3)
            cv2.putText(src, x[4], (int(x[0])+5,(int(x[1])+int(x[3])//4)), cv2.FONT_HERSHEY_SIMPLEX, 1,color, 2, cv2.LINE_AA)
        f.close
        # Write answer to the src as well
        cv2.imwrite(name, src)

    print("Program end")
