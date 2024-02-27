import cv2
from stitching import Stitcher

stitcher = Stitcher(detector="sift", confidence_threshold=0.2)
# panorama = stitcher.stitch(["./Images/1/1.png", "./Images/1/2.png"])
# panorama = stitcher.stitch(["./Images/2/01.jpg", "./Images/2/02.jpg"])
(panorama) = stitcher.stitch(["./Images/3/01.jpg", "./Images/3/02.jpg", "./Images/3/03.jpg"])
# panorama = stitcher.stitch(["./Images/4/00000000-1.jpg", "./Images/4/00000000-2.jpg"])
cv2.imshow("Image", panorama)
cv2.waitKey(0) 


# Replace with the actual paths to your image files
# image_paths = ["./Images/1/1.png", "./Images/1/2.png", "./Images/1/3.png", "./Images/1/4.png"]
# image_paths = ["./Images/2/01.jpg", "./Images/2/02.jpg"]
# image_paths = ["./Images/3/01.jpg", "./Images/3/02.jpg", "./Images/3/03.jpg"]
image_paths = ["./Images/4/00000000-1.jpg", "./Images/4/00000000-2.jpg"]
