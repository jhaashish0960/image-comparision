import cv2
import numpy as np
img1 = cv2.imread("C:\\Users\\dell\\Documents\\proj\\einstein 20190210_211406.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("C:\\Users\\dell\\Documents\\proj\\akansha 20180731_150952.jpg" , cv2.IMREAD_GRAYSCALE)
#detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute( img1, None)
kp2, des2 = orb.detectAndCompute(img2 , None)
#for d in des1:
 #   print(d)
 #brute force matching
bf =cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches , key = lambda x:x.distance)
matching_result = cv2.drawMatches(img1 ,kp1, img2,kp2, matches[:30], None )
#for m in matches:
 #   print(m.distance)

cv2.imshow("img1" , img1)
cv2.imshow("img2" , img2)
cv2.imshow("matching result" , matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()