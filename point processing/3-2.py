import math
import cv2

'''
3. 컬러영상에서의 point processing (옵션 - 가산점 있음)

: HSI 변환 후, Intensity에 대해 위의 point processing을 수행한 후, 결과 영상을 RGB로 변환해서 출력 

: HSI 변환은 라이브러리로 처리 가능 

: 첨부된 컬러 영상들 
'''

dir_path = "Point processing/images"
img_name = "baboon.png"
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
RGB 이미지를 HSI 이미지로 변환 후 Intensity 만 가져옴
'''
hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
intensity = hsi[:,:,2]

'''
clear histogram to 0
'''
histogram = [0 for i in range(0, max_level + 1)]
sum_hist = [0 for i in range(0, max_level + 1)]

'''
calculate histogram
'''
for i in range(intensity.shape[0]):   
    for j in range(intensity.shape[1]): 
        histogram[intensity[i, j]] += 1
        
'''
calculate normalized sum of histogram
'''
sum = 0
shape = intensity.shape
total_pixels = shape[0] * shape[1]
scale_factor = max_level / total_pixels
for i in range(0, len(histogram)):
    sum += histogram[i]
    sum_hist[i] = math.floor(sum * scale_factor + 0.5)

'''
transform image using new sum_histogram as a LUT
'''
for i in range(intensity.shape[0]):   
    for j in range(intensity.shape[1]): 
        intensity[i, j] = sum_hist[intensity[i, j]]
        
'''
HSI to RGB
'''
rgb = cv2.cvtColor(hsi, cv2.COLOR_HSV2BGR)

show_img(rgb, title="Histogram equalization-")
