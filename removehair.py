import cv2


def removeHair(img):
    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))
    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(img, thresh2, 1, cv2.INPAINT_TELEA)
    return dst
