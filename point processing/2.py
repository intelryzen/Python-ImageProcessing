import numpy as np
import math
import cv2

'''
2. Histogram processing을 통한 대비 개선 (필수)

: Histogram Equalization 구현

: 첨부된 럏0316 영상에 대해 수행 
'''

dir_path = "Point processing/images"
img_name = "HW1-1 Fig0316(4)(bottom_left).jpg"
test_image_path = f"{dir_path}/{img_name}"

def show_img(image, title="image"): 
    cv2.imshow(title, image)  # 이미지 출력 (BGR)
    # cv2.imwrite(title + img_name, image)
    cv2.waitKey(0)            # 키보드 입력 대기 (아무키 입력시 꺼짐)
    cv2.destroyAllWindows()   # 나타는 Window 제거

# 256 단계 중 가장 큰 값
max_level = 255

# 이미지 불러오기
img = cv2.imread(test_image_path, cv2.IMREAD_COLOR)

'''
컬러 이미지를 Gray 이미지로 변환 (둘 중 하나 쓰면 됨.)
gray = ( R + G + B ) / 3
'''
# gray_img = np.mean(img, axis=2).astype(np.uint8)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
clear histogram to 0
'''
histogram = [0 for i in range(0, max_level + 1)]
sum_hist = [0 for i in range(0, max_level + 1)]

'''
calculate histogram
'''
for i in range(gray_img.shape[0]):   
    for j in range(gray_img.shape[1]): 
        histogram[gray_img[i, j]] += 1
        
'''
calculate normalized sum of histogram
'''
sum = 0
shape = gray_img.shape
total_pixels = shape[0] * shape[1]
scale_factor = max_level / total_pixels
for i in range(0, len(histogram)):
    sum += histogram[i]
    sum_hist[i] = math.floor(sum * scale_factor + 0.5)

'''
transform image using new sum_histogram as a LUT
'''
for i in range(gray_img.shape[0]):   
    for j in range(gray_img.shape[1]): 
        gray_img[i, j] = sum_hist[gray_img[i, j]]

show_img(gray_img, title="Histogram Equalization-")
