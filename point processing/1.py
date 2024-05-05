import numpy as np
import cv2

'''
1. function 을 이용한 변환 (필수)

: negative, log, power-law transformation (감마값의 범위에 따라 다양한 결과 분석, >1, <1 의 각 범위에서 두가지 값 이상 설정)

: 첨부된 Fig0308, Fig0309, Fig0525 파일에서 결과 분석 

'''

dir_path = "Point processing/images"
img_name = "HW1-2  Fig0525(c)(aerial_view_turb_c_0pt001).tif"
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
Negative 이미지
negative = max_level - 현재 컬러 레벨
'''
def trans_negative(gray_img):
    negative_img = max_level - gray_img
    show_img(negative_img)
    
trans_negative(gray_img)

'''
Log 이미지
s = c log ( 1 + f(x,y) )
특히 어두운 것을 밝게 
'''
def trans_log(gray_img):
    print(np.max(gray_img))
    log_img =np.uint8(np.log(1+np.double(gray_img)) * (max_level/np.log(max_level + 1)))
    print(np.max(log_img))
    show_img(log_img)
    
trans_log(gray_img)

'''
power-law transformation 이미지
s = c r^γ
γ 가 1보다 크면 밝은 것을 어둡게
γ 가 1이면 그대로
γ 가 1보다 작으면 어두운 것을 밝게
'''
def trans_pow(gray_img):
    for i in [5, 2, 1, 0.9, 0.7]:
        print(f"gamma = {i}, 전: ", np.max(gray_img))
        gamma = i
        c = 1
        power_law_img =c*(np.double(gray_img)**gamma)
        power_law_img = (max_level/(c*(max_level**gamma))*power_law_img).astype(np.uint8)
        print("ㄴ 후: ", np.max(power_law_img))
        show_img(power_law_img, title=f"Power({i})-")

trans_pow(gray_img)     

