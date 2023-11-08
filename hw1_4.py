import cv2
import numpy as np

# 디버그 모드 플래그
DEBUG_MODE = True  # 필요한 경우 True로 설정, 필요하지 않은 경우 False로 설정

# 이미지를 불러옴.
image_path = 'img_2.png'  # 이미지 파일 경로를 지정하세요.
image = cv2.imread(image_path)

# 이미지를 그레이스케일로 변환하세요.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Debug: Binary Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Canny 엣지 검출을 수행하세요.
edges = cv2.Canny(gray, 0, 254)

# 엣지 이미지를 디버깅용으로 화면에 출력하세요.
cv2.imshow('Debug: Binary Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 허프 변환을 사용하여 원 검출
circles = cv2.HoughCircles(
    gray,               # 그레이스케일 이미지
    cv2.HOUGH_GRADIENT,       # Hough 변환 방법
    dp=1,                     # 이미지 해상도에 대한 역비율
    minDist=20,               # 원 사이의 최소 거리
    param1=50,                # Canny 엣지 검출기의 고장도 임계값
    param2=30,                # 작으면 더 많은 원을 검출, 크면 정확한 원 검출
    minRadius=0,              # 원의 최소 반지름
    maxRadius=0               # 원의 최대 반지름
)

if circles is not None:
    # 검출된 원 그리기
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])   # 원의 중심 좌표
        radius = i[2]           # 원의 반지름
        cv2.circle(image, center, radius, (0, 255, 0), 2)  # 원 그리기
        cv2.circle(image, center, 2, (0, 0, 255), 3)       # 중심점 그리기

    # 결과 이미지 저장
    cv2.imwrite('output_image.jpg', image)
else:
    print("원을 찾지 못했습니다.")
