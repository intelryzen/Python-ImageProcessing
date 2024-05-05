# High-boost = A*Original + Highpass

import numpy as np
import cv2
from noise_filtering import apply_kernel

dir_path = "Area processing/images"
img_name = "HW1-2 Fig0309(a)(washed_out_aerial_image).tif"
test_image_path = f"{dir_path}/{img_name}"

def show_img(image, title="image"): 
    cv2.imshow(title, image)  # 이미지 출력 (BGR)
    # cv2.imwrite(title + img_name, image)
    cv2.waitKey(0)            # 키보드 입력 대기 (아무키 입력시 꺼짐)
    cv2.destroyAllWindows()   # 나타는 Window 제거
    
def apply_highboost(image, A=1.0):
    kernel = np.array([
        [0, -1, 0],
        [-1, A+4, -1],
        [0, -1, 0]
    ])
    output = apply_kernel(image, kernel)
    show_img(output)

if __name__ == "__main__":
    # 이미지 불러오기
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    apply_highboost(img)
