import cv2
import numpy as np

# 디버그 모드 플래그
DEBUG_MODE = True  # 필요한 경우 True로 설정, 필요하지 않은 경우 False로 설정

# 이미지를 불러오세요.
image_path = 'img_4.png'  # 이미지 파일 경로를 지정하세요.
image = cv2.imread(image_path)
# 이미지를 그레이스케일로 변환하세요.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 적응형 이진화를 수행하세요.
binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=51, C=2
)
cv2.imshow('Debug: Binary Image', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Canny 엣지 검출을 수행하세요.  낮은 임계값, 높은 임계값 설정 유의
edges = cv2.Canny(binary, 10, 200)

# 엣지 디버깅
if DEBUG_MODE:
    # 엣지 이미지를 디버깅용으로 화면에 출력하세요.
    cv2.imshow('Debug: Binary Image', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 확률적 허프 변환을 사용하여 직선 검출
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=10, maxLineGap=10)

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


# 결과를 이미지에 그리고 화면에 표시합니다.
for corner in corners:
    cv2.circle(image, tuple(corner), 5, (0, 255, 0), -1)


# 꼭짓점 중에서 체커보드의 교차점을 선택합니다.
corners.sort(key=lambda x: (x[0], x[1]))  # x와 y 좌표를 기준으로 정렬

# 교차점을 루트로 계산하여 출력합니다.
grid_size_x = int(np.sqrt(len(corners)))
grid_size_y = int(np.sqrt(len(corners)))
print(f"{grid_size_x-1}x{grid_size_y-1}")


# 디버그 모드일 때 결과 이미지를 표시합니다.
if DEBUG_MODE:
    cv2.imshow('Estimated Checkerboard Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
