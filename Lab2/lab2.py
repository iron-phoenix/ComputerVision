import cv2
import numpy as np

image_filename = "text.bmp"

def binary_img(image):
	gaussian_kernel_size = 21
	blurred_image = cv2.GaussianBlur(image, (gaussian_kernel_size, gaussian_kernel_size), 0)
	laplacian = cv2.Laplacian(blurred_image, cv2.CV_8U, None, ksize = 5, scale = 1)
	_, result_image = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY)
	return result_image

def show_image(window_title, image):
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def morph_trans(image):
	kernel = np.ones((3, 3), np.uint8)
	morph = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	return morph

source_image = cv2.imread(image_filename)
img_gray = cv2.cvtColor(source_image, cv2.COLOR_RGB2GRAY)

bin_img = binary_img(img_gray)

show_image("Binary image", bin_img)

morph = morph_trans(bin_img)

show_image("Morphological Transformations", morph)

contours, _ = cv2.findContours(morph, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(source_image, (x, y), (x + w, y + h),(0, 255, 0), 2)
show_image("Result", source_image)