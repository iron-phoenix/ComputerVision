import cv2

image_filename = "text.bmp"
gaussian_kernel_size = 21

def show_image(window_title, image):
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

source_image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
blurred_image = cv2.GaussianBlur(source_image, (gaussian_kernel_size, gaussian_kernel_size), 0)
laplacian = cv2.Laplacian(blurred_image, cv2.CV_32F)
_, result_image = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY)
show_image("result", result_image)
