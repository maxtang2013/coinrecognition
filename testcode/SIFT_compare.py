import cv2
import numpy as np

img1 = cv2.imread("/Users/maxtang/Projects/study/vip/model_100/data/onedollar/43.png")
img2 = cv2.imread("/Users/maxtang/Projects/study/vip/model_100/data/onedollar/44.png")

sift = cv2.xfeatures2d.SIFT_create()
kp1,des1 = sift.detectAndCompute(img1, None)
kp2,des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=5)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)


kpt_img1 = cv2.drawKeypoints(img1, kp1,img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('out_kpts1.png', kpt_img1)

kpt_img2 = cv2.drawKeypoints(img2, kp2,img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('out_kpts2.png', kpt_img2)


# print (des1)
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.9*n.distance:
#         good.append([m])
# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

cv2.imwrite('out.png', img3)