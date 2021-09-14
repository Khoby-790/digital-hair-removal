
from removehair import removeHair
import cv2

path = r'inputImages/ISIC_0000095.jpg'


src = cv2.imread(path)
cv2.imshow("original Image", src)


dst = removeHair(src)
cv2.imwrite('newImage.jpg', dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
