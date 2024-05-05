import cv2

dir_path = "Area processing/images"
img_name = "peppers.png"
test_image_path = f"{dir_path}/{img_name}"

def show_img(image, title="image"): 
    cv2.imshow(title, image)  # 이미지 출력 (BGR)
    # cv2.imwrite(title + img_name, image)
    cv2.waitKey(0)            # 키보드 입력 대기 (아무키 입력시 꺼짐)
    cv2.destroyAllWindows()   # 나타는 Window 제거
    
if __name__ == "__main__":
    # 이미지 불러오기
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, threshold1=50, threshold2=200)
    show_img(edges)
