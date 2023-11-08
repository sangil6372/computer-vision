import cv2
import numpy as np

# 디버그 모드 플래그
DEBUG_MODE = True  # 필요한 경우 True로 설정, 필요하지 않은 경우 False로 설정


# 이미지를 불러오세요.
image_path = 'img.png'  # 이미지 파일 경로를 지정하세요.
image = cv2.imread(image_path)

# 이미지를 그레이스케일로 변환하세요.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 이미지를 이진화하세요.
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Canny 엣지 검출을 수행하세요.
edges = cv2.Canny(binary, 100, 200)

# 엣지 이미지를 디버깅용으로 화면에 출력하세요.
cv2.imshow('Debug: Binary Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 확률적 허프 변환을 사용하여 직선을 찾으세요.
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

# 가로 및 세로 선을 선택하세요.
horizontal_lines = []
vertical_lines = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1)  # 직선의 기울기 각도 계산
    if -np.pi / 4 < angle < np.pi / 4:
        vertical_lines.append(line)
    else:
        horizontal_lines.append(line)

# 가로 및 세로 선이 만나는 지점을 계산하세요.
corners = []

for hl in horizontal_lines:
    for vl in vertical_lines:
        x1_h, y1_h, x2_h, y2_h = hl[0]
        x1_v, y1_v, x2_v, y2_v = vl[0]
        A = np.array([[y2_h - y1_h, x1_h - x2_h], [y2_v - y1_v, x1_v - x2_v]])
        b = np.array([x1_h * (y2_h - y1_h) - y1_h * (x2_h - x1_h), x1_v * (y2_v - y1_v) - y1_v * (x2_v - x1_v)])
        intersection = np.linalg.solve(A, b)
        corners.append(intersection.astype(int))

# 결과를 출력하세요.
for corner in corners:
    cv2.circle(image, tuple(corner), 5, (0, 0, 255), -1)

cv2.imshow('Checkerboard Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
