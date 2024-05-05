import numpy as np
import cv2

'''
3. 컬러영상에서의 point processing (옵션 - 가산점 있음)

: HSI 변환 후, Intensity에 대해 위의 point processing을 수행한 후, 결과 영상을 RGB로 변환해서 출력 

: HSI 변환은 라이브러리로 처리 가능 

: 첨부된 컬러 영상들 
'''

dir_path = "Point processing/images"
img_name = "peppers.png"
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
RGB 이미지를 HSI 이미지로 변환
'''
hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

'''
Negative 이미지
negative = max_level - 현재 컬러 레벨
'''
def trans_negative(hsi):
    temp = hsi.copy()
    intensity = temp[:,:,2]
    temp[:,:,2] = max_level - intensity
    rgb = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
    show_img(rgb, title="Negative-")

trans_negative(hsi)

'''
Log 이미지
s = c log ( 1 + f(x,y) )
특히 어두운 것을 밝게 
'''
def trans_log(hsi):
    temp = hsi.copy()
    intensity = temp[:,:,2]
    print(np.max(intensity))
    temp[:,:,2] = np.uint8(np.log(1+np.double(intensity)) * (max_level/np.log(max_level + 1)))
    print(np.max(temp[:,:,2]))
    rgb = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
    show_img(rgb, title="Log-")

trans_log(hsi)

'''
power-law transformation 이미지
s = c r^γ
γ 가 1보다 크면 밝은 것을 어둡게
γ 가 1이면 그대로
γ 가 1보다 작으면 어두운 것을 밝게
'''
def trans_pow(hsi):
    for i in [7, 2, 1, 0.8, 0.5]:
        temp = hsi.copy()
        intensity = temp[:,:,2]
        print(f"gamma = {i}, 전: ", np.max(intensity))
        gamma = i
        c = 1
        intensity =c*(np.double(intensity)**gamma)
        temp[:,:,2] = (max_level/(c*(max_level**gamma))*intensity).astype(np.uint8)
        print("ㄴ 후: ", np.max(temp[:,:,2]))
        rgb = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
        show_img(rgb, title=f"Power({i})-")

trans_pow(hsi)
