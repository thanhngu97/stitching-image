import cv2
import numpy as np

def stitch_images(image_top_path, image_bottom_path, image_left_path, image_right_path):
    # Step 1: Reading images
    image_top = cv2.imread(image_top_path)
    image_bottom = cv2.imread(image_bottom_path)
    image_left = cv2.imread(image_left_path)
    image_right = cv2.imread(image_right_path)

    # Step 2: Convert all of thing to gray image
    image_top_gray = cv2.cvtColor(image_top, cv2.COLOR_BGR2GRAY)
    image_bottom_gray = cv2.cvtColor(image_bottom, cv2.COLOR_BGR2GRAY)
    image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)


    # Step 3: Find key points and descriptors
    orb = cv2.ORB_create()
    kp_top, des_top = orb.detectAndCompute(image_top_gray, None)
    kp_bottom, des_bottom = orb.detectAndCompute(image_bottom_gray, None)
    kp_left, des_left = orb.detectAndCompute(image_left_gray, None)
    kp_right, des_right = orb.detectAndCompute(image_right_gray, None)


    # Step 4: Matching descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_top_bottom = bf.match(des_top, des_bottom)
    matches_left_right = bf.match(des_left, des_right)


    # Step 5: Sort points matching
    matches_top_bottom = sorted(matches_top_bottom, key=lambda x: x.distance)
    matches_left_right = sorted(matches_left_right, key=lambda x: x.distance)

    # Hold a few points matching
    matches_top_bottom = matches_top_bottom[:10]
    matches_left_right = matches_left_right[:10]

    # Step 6: Make Homography matrix
    src_pts_top_bottom = np.float32([kp_top[m.queryIdx].pt for m in matches_top_bottom]).reshape(-1, 1, 2)
    dst_pts_top_bottom = np.float32([kp_bottom[m.trainIdx].pt for m in matches_top_bottom]).reshape(-1, 1, 2)
    H_top_bottom, _ = cv2.findHomography(src_pts_top_bottom, dst_pts_top_bottom, cv2.RANSAC, 5.0)

    src_pts_left_right = np.float32([kp_left[m.queryIdx].pt for m in matches_left_right]).reshape(-1, 1, 2)
    dst_pts_left_right = np.float32([kp_right[m.trainIdx].pt for m in matches_left_right]).reshape(-1, 1, 2)
    H_left_right, _ = cv2.findHomography(src_pts_left_right, dst_pts_left_right, cv2.RANSAC, 5.0)


   # Step 7: Warping and Stitching
    result_top_bottom = cv2.warpPerspective(image_top, H_top_bottom, (image_top.shape[1], image_top.shape[0] + image_bottom.shape[0]))
    result_top_bottom[0:image_bottom.shape[0], 0:image_bottom.shape[1]] = image_bottom
    print("result_top_bottom:", result_top_bottom.shape)

    result_left_right = cv2.resize(result_left_right, (result_top_bottom.shape[1], result_top_bottom.shape[0]))
    print("result_left_right:", result_left_right.shape)

    final_result = cv2.addWeighted(result_top_bottom, 0.5, result_left_right, 0.5, 0)


    # Step 8: Show result
    cv2.imshow('Stitched Image', final_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # image_paths = ["./Images/5/1.jpg", "./Images/5/2.jpg", "./Images/5/3.jpg", "./Images/5/4.jpg"]
    stitch_images("./Images/5/1.jpg", "./Images/5/2.jpg", "./Images/5/3.jpg", "./Images/5/4.jpg")
