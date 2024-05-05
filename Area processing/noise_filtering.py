import numpy as np
import cv2

dir_path = "Area processing/images"
img_name = "HW1-2  Fig0525(c)(aerial_view_turb_c_0pt001).tif"
test_image_path = f"{dir_path}/{img_name}"

def show_img(image, title="image"): 
    cv2.imshow(title, image)  # 이미지 출력 (BGR)
    # cv2.imwrite(title + img_name, image)
    cv2.waitKey(0)            # 키보드 입력 대기 (아무키 입력시 꺼짐)
    cv2.destroyAllWindows()   # 나타는 Window 제거
    
# 제로패딩
def apply_zero_padding(image, kernel):
    kernel_height = kernel.shape[0]
    padding = kernel_height // 2
    return np.pad(image, padding, mode='constant', constant_values=0)

# 마스크 연산
def apply_kernel(image, kernel):
    height, width = image.shape
    kernel_height = kernel.shape[0]
    padded_image = apply_zero_padding(image, kernel)
    output = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            input = padded_image[i:i+kernel_height, j:j+kernel_height]
            output[i, j] = np.sum(input * kernel)
    return output

# 가우시안 필터링 3x3
def apply_gaussian(image):
    kernel = (1/16)*np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ])
    output = apply_kernel(image, kernel)
    show_img(output)

# 가우시안 필터링 5X5
def apply_gaussian2(image):
    kernel = (1/273)*np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ])
    output = apply_kernel(image, kernel)
    show_img(output)

# 미디안 필터링
def apply_median(image, size):
    height, width = image.shape
    padded_image = apply_zero_padding(image, np.zeros((size, size)))
    output = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            input = padded_image[i:i+size, j:j+size]
            output[i, j] = np.median(input)
    show_img(output)

if __name__ == "__main__":
    # 이미지 불러오기
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

    apply_gaussian(img)
    apply_gaussian2(img)
    apply_median(img, 3)
    apply_median(img, 5)
