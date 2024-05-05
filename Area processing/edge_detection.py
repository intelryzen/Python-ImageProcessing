import numpy as np
import cv2
from noise_filtering import apply_kernel, apply_zero_padding

dir_path = "Area processing/images"
img_name = "peppers.png"
test_image_path = f"{dir_path}/{img_name}"

def show_img(image, title="image"): 
    cv2.imshow(title, image)  # 이미지 출력 (BGR)
    # cv2.imwrite(title + img_name, image)
    cv2.waitKey(0)            # 키보드 입력 대기 (아무키 입력시 꺼짐)
    cv2.destroyAllWindows()   # 나타는 Window 제거
    
def apply_sobel(image):
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    height, width = image.shape
    kernel_height = kernel_x.shape[0]
    padded_image = apply_zero_padding(image, kernel_x)
    output = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            input = padded_image[i:i+kernel_height, j:j+kernel_height]
            Gx = np.sum(kernel_x * input)
            Gy = np.sum(kernel_y * input)
            output[i, j] = np.sqrt(Gx**2 + Gy**2)

    # output_x = apply_kernel(image, kernel_x)
    # output_y = apply_kernel(image, kernel_y)

    # 각 축에 대한 쓰레시홀딩
    # thres_x = np.where(output_x >= 127, 255, 0).astype(np.uint8)
    # thres_y = np.where(output_y >= 127, 255, 0).astype(np.uint8)

    # 쓰레시홀딩
    thres = np.where(output >= 50, 255, 0).astype(np.uint8)
    show_img(thres)

def apply_prewitt(image):
    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    kernel_y = np.array([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])

    height, width = image.shape
    kernel_height = kernel_x.shape[0]
    padded_image = apply_zero_padding(image, kernel_x)
    output = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            input = padded_image[i:i+kernel_height, j:j+kernel_height]
            Gx = np.sum(kernel_x * input)
            Gy = np.sum(kernel_y * input)
            output[i, j] = np.sqrt(Gx**2 + Gy**2)

    # output_x = apply_kernel(image, kernel_x)
    # output_y = apply_kernel(image, kernel_y)

    # 각 축에 대한 쓰레시홀딩
    # thres_x = np.where(output_x >= 127, 255, 0).astype(np.uint8)
    # thres_y = np.where(output_y >= 127, 255, 0).astype(np.uint8)

    # 쓰레시홀딩
    thres = np.where(output >= 50, 255, 0).astype(np.uint8)
    show_img(thres)

if __name__ == "__main__":
    # 이미지 불러오기
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    # apply_sobel(img)
    apply_prewitt(img)
