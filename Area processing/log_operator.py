import cv2

dir_path = "Area processing/images/Edge Detection"
img_name = "Fig0327(a)(tungsten_original).jpg"
test_image_path = f"{dir_path}/{img_name}"

def show_img(image, title="image"): 
    # cv2.imshow(title, image)  # 이미지 출력 (BGR)
    cv2.imwrite(title + img_name, image)
    # cv2.waitKey(0)            # 키보드 입력 대기 (아무키 입력시 꺼짐)
    # cv2.destroyAllWindows()   # 나타는 Window 제거

def log(image, sigma):
    gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
    output = cv2.Laplacian(gaussian, cv2.CV_64F)
    show_img(output, title=f"sigma {sigma}")

def log_color(image, sigma):
    hsi = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    intensity = hsi[:,:,2]

    gaussian = cv2.GaussianBlur(intensity, (0, 0), sigmaX=sigma, sigmaY=sigma)
    output = cv2.Laplacian(gaussian, cv2.CV_64F)
    show_img(output, title=f"sigma {sigma}")

if __name__ == "__main__":
    # 이미지 불러오기
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

    # Color 이미지
    img2 = cv2.imread(test_image_path, cv2.IMREAD_COLOR)

    log(img, sigma=0.3)
    # log_color(img2, sigma=0.3)
