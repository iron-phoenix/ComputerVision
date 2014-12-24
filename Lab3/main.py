import cv2
import numpy

ZERO_SIZE = 30

if __name__ == "__main__":
	image = cv2.imread("mandril.bmp", cv2.CV_LOAD_IMAGE_GRAYSCALE)
	
	fft_shift = numpy.fft.fftshift(numpy.fft.fft2(image))
	rows, cols = fft_shift.shape
	center_row, center_col = rows / 2, cols / 2
	fft_shift[center_row - ZERO_SIZE:center_row + ZERO_SIZE, center_col - ZERO_SIZE:center_col + ZERO_SIZE] = 0
	result_image = numpy.abs(numpy.fft.ifft2(numpy.fft.ifftshift(fft_shift)))
	
	cv2.imwrite('filtered.bmp', result_image)
	cv2.imwrite('laplassian.bmp', cv2.Laplacian(image, cv2.CV_32F))