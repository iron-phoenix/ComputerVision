import cv2
import numpy

EPS = 1.7

def transform_image(image, angle, scale):
    rows, cols = image.shape
    image_center = (rows / 2, cols / 2)
    rotatation_matrix = cv2.getRotationMatrix2D(image_center, angle, scale)
    result_image = cv2.warpAffine(image, rotatation_matrix, image.shape)
    return result_image, rotatation_matrix

if __name__ == "__main__":
	image = cv2.imread("mandril.bmp", cv2.CV_LOAD_IMAGE_GRAYSCALE)
	transformed_image, rotation_matrix = transform_image(image, 45, 0.5)
	cv2.imwrite("transformed_image.bmp", transformed_image)

	sift = cv2.SIFT(500)
	kp_original, des_original = sift.detectAndCompute(image, None)
	kp_transformed, des_transformed = sift.detectAndCompute(transformed_image, None)
	original_keypoints = cv2.drawKeypoints(image, kp_original, color = (0,255,0), flags = 0)
	transformed_keypoints = cv2.drawKeypoints(transformed_image, kp_transformed, color = (0,255,0), flags = 0)
	cv2.imwrite("original_keypoints.bmp", original_keypoints)
	cv2.imwrite("transformed_keypoints.bmp", transformed_keypoints)
	
	transformed_points = []
	for point in numpy.array(map(lambda a: numpy.array(a.pt, numpy.float32), kp_original)):
		transformed_points.append(numpy.dot(rotation_matrix, (point[0], point[1], 1)))
	transformed_old_points = numpy.int32(numpy.array(transformed_points))
	for point in transformed_old_points:
		cv2.circle(transformed_image, (point[0], point[1]), 3, (255, 255, 255), thickness = -1)
	cv2.imwrite("transformed_keypoints_dots.bmp", transformed_image)
	
	matches = (cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})).knnMatch(des_original, des_transformed, k = 2)
	
	good = []
	for m, n in matches:
		if m.distance < 0.75 * n.distance:
			good.append(m)
	count = 0
	for g in good:
		distance = numpy.linalg.norm(transformed_old_points[g.queryIdx] - kp_transformed[g.trainIdx].pt)
		if distance < EPS:
			count += 1
	
	matched = count * 100. / len(good)
	print('{0}% of keypoints were matched. Epsilon = {1}'.format(matched, EPS))