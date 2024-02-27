import cv2
import numpy as np
from stitching import Stitcher

# Replace with the actual paths to your image files
# image_paths = ["./Images/1/1.png", "./Images/1/2.png", "./Images/1/3.png", "./Images/1/4.png"]
# image_paths = ["./Images/2/01.jpg", "./Images/2/02.jpg"]
# image_paths = ["./Images/3/01.jpg", "./Images/3/02.jpg",   "./Images/3/03.jpg"]
# image_paths = ["./Images/4/00000000-1.jpg", "./Images/4/00000000-2.jpg"]
image_paths = ["./Images/5/1.jpg", "./Images/5/2.jpg", "./Images/5/3.jpg", "./Images/5/4.jpg"]

# Load images
images = [cv2.imread(path) for path in image_paths]

# Resize images (optional)
for i in range(len(images)):
    images[i] = cv2.resize(images[i], (0, 0), fx=0.5, fy=0.5)

# Initialize an empty array to store the final stitched image
final_stitched_image = images[0]

# Feature Detection and Matching (using ORB) for each consecutive pair of images
orb = cv2.ORB_create()
bf = cv2.BFMatcher()

for i in range(len(images) - 1):
    # Detect ORB keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(final_stitched_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(images[i + 1], None)

    # Use a feature matching algorithm (e.g., Brute Force Matcher)
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the current image onto the stitched result using the homography matrix
    result = cv2.warpPerspective(final_stitched_image, H, (final_stitched_image.shape[1] + images[i + 1].shape[1], final_stitched_image.shape[0]))

    # Overlay the next image onto the warped result
    result[0:images[i + 1].shape[0], 0:images[i + 1].shape[1]] = images[i + 1]

    # Draw matches for the current pair
    stitched_image = cv2.drawMatches(final_stitched_image, keypoints1, images[i + 1], keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Update the final stitched image
    final_stitched_image = result

    # Display the matches for the current pair
    cv2.imshow(f"Matches {i}-{i+1}", stitched_image)
    cv2.waitKey(0)

# Try using the Stitcher for additional stitching
stitcherResult = Stitcher(detector="sift", confidence_threshold=0.2)
panorama = stitcherResult.stitch(image_paths)

# Check if the stitching was successful
if panorama is not None:
    cv2.imshow("Final Stitched Image", panorama)
    cv2.waitKey(0)
else:
    print("Manual stitching: No match exceeds the given confidence threshold.")

cv2.destroyAllWindows()