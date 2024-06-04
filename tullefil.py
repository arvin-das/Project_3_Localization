import numpy as np
import cv2
from glob import glob
import os
import pickle

open_path = "imgs/New_Paper_Chessboard/"        # folder of calibration images
save_path = "imgs/Undistorted/"             # folder to save undistorted images in
data_path = "data/Calibration_data"         # filename under which calibration data will be saved
patternSize = (9,6)                         # number of inner corners of chessboard pattern, (columns,rows)
squareSize = 26                             # length of single chessboard field in [mm] 
chosenImg = "0"                             # name of image whose coordinate system is to be used for localization
z_shift = 5                                 # distance in [mm] by which object centers are shifted upwards from surface (in z direction)

# Generate points in object coordinate system
# Origin in bottom left corner
obj_points = np.zeros((patternSize[0]*patternSize[1],3), np.float32)  # float32 needed for cv2.calibrateCamera
obj_points[:,:2] = np.mgrid[0:patternSize[0], patternSize[1]-1:-1:-1].T.reshape(-1,2)
# second index (for y-coordinate) in reverse order to account for OpenCV corner detection going from top left to bottom right
obj_points *= squareSize # Scale by square size

# Arrays for all object and image points of corners
objectPoints = []   # 3D points in real world 
imagePoints = []    # 2D points in image
detectedCounter = 0 # counter for number of images where corners were correctly detected

# Get filenames of all .jpg images in folder
filenames = sorted(glob(os.path.join(open_path, "*.jpg")), key=len)  

for imageName in filenames:     # for every image in folder
    fileNr = ""     
    for char in imageName:       # extract numeric chars from full file name
        if char.isnumeric():
            fileNr += char 

    bgr_img = cv2.imread(imageName)                         # load image
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)    # convert to grayscale, single channel image

    # Find pixel coordinates of inner chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_img, patternSize, None) 

    if ret: # if corners were found
        print("Corners on image {:} were found.".format(fileNr))

        if fileNr == chosenImg:         # if image is the one chosen to be used for localization
            chosenIdx = detectedCounter # store index

        detectedCounter +=1

        # Refine corner location with better subpixel accuracy
        corners = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), 
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        # (11,11) is size of search window
        # (-1,-1) for zero zone where gradient is ignored to avoid singularities, not used in this case
        # criteria for termination: either if corners move by less than 0.1 pixel per iteration or after 30 iterations

        objectPoints.append(obj_points)
        imagePoints.append(corners)

        # Draw chessboard onto image
        # cv2.drawChessboardCorners(bgr_img, patternSize, corners, ret) # draw chessboard corners on image
        # cv2.imshow(fileNr, bgr_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("Could not find corners on image {:}.".format(fileNr))

# Calibrate camera, calculate intrinsic and extrinsic parameters
imgSize = (gray_img.shape[1],gray_img.shape[0]) # (width,height)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imgSize, None, None)
print("Reprojection error across all images:\t{:.5f}".format(ret))
# None:     No initial guesses for camera matrix and distortion coefficients

# ret:      float, reprojection error
# mtx:      3x3 numpy array, intrinsic matrix of camera
# dist:     distortion coefficients [k1, k2, p1, p2, k3]
# rvecs:    vector of 3x1 rotation vectors, rotation axis with |r|=theta
# tvecs:    vector of 3x1 translation vectors, from object to camera coordinate system

# Undistort images
for imageName in filenames:     # for each image in folder
    fileNr = ""     
    for char in imageName:      # extract numeric chars from full file name
        if char.isnumeric():
            fileNr += char 

    img = cv2.imread(imageName)
    undst = cv2.undistort(img, mtx, dist)
    cv2.imwrite(save_path + fileNr + '_undistorted.bmp',undst)

    # cv2.imshow('Original',img)
    # cv2.waitKey(0)
    # cv2.imshow('Undistorted',undst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Create complete transformation matrices
cTo_matrices = []           # list of transformation matrices, 3D world COS to 3D camera COS
P_matrices = []             # list of transformation matrices, 3D world COS to 2D image COS
H_matrices = []             # list of homography matrices, 2D world COS to 2D image COS

for imgNr in range(detectedCounter):    # for each image where corners have successfully been detected

    R = cv2.Rodrigues(rvecs[imgNr])[0]  # 3x3 rotation matrix from rotation vector
    T = tvecs[imgNr]                    # 3x1 translation vector
    # Homogeneous transformation from object to camera coordinate system
    cTo = np.vstack([np.hstack([R,T]),np.array([0,0,0,1])]) 
    # Complete transformation from 3D space to pixel space (not including distortion)
    P = np.dot(mtx,np.hstack([R,T]))
    # Calculate homography matrix
    H = np.delete(P,2,axis=1)
    
    # Store matrices in lists
    cTo_matrices.append(cTo)
    P_matrices.append(P) 
    H_matrices.append(H)

# Modify transformation of chosen image to account for object height
T_shift = np.eye(4) 
T_shift[2,3] = z_shift                          # homogeneous transf. which shifts coordinate system in z direction
cTo = np.dot(cTo_matrices[chosenIdx],T_shift)   # homogeneous transf. from shifted world coordinates to 3D camera coordinates
P = np.dot(mtx,cTo[:3,:])                       # transformation from 3D space to pixel space 
H = np.delete(P,2,axis=1)                       # homography matrix

# Store matrices in first position in lists
cTo_matrices.insert(0, cTo)
P_matrices.insert(0, P)
H_matrices.insert(0, H)

# Convert lists of arrays to numpy array
cTo_matrices = np.asarray(cTo_matrices)
P_matrices = np.asarray(P_matrices)
H_matrices = np.asarray(H_matrices)

# Export calibration data with pickle
with open(data_path, 'wb') as file: # wb for writing, binary
    # Serialize and write the variable to the file
    pickle.dump([mtx, dist, H_matrices, cTo_matrices], file) 
    # Exported data:
    # intrinsic matrix, distortion parameters,
    # all homography matrices, all homogeneous transformation matrices





##### Kode fra morten


z_shift = 7.36 # [mm] average distance from plane to bean/chickpea

# Perform homogenous transform to shift system in z-direction
T_shift = np.eye(3)
T_shift[1,2] = z_shift # Originalt [2,3] pga 3x4 matrise

# Shift from world coordinates to camera coordinates
camToObject = np.dot(extrinsic_matrix, T_shift)

# Transform from 3D space to pixels
pixel_matrix = np.dot(mtx, camToObject[:3,:])
print("Pixel matrix\n", pixel_matrix)

# Final homography matrix to utilize
homo_matrix = np.delete(P, 2, axis=1)
print("Homogenous matrix:\n", homo_matrix)

#Slik ser det ut for oss hvis jeg klarte å tolke deres kode riktig, de jobber med 3x4 matrise, så prøvde å justere for det ved å endre fra [2,3] til [1,2] (se kommentar)
