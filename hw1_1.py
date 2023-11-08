import cv2
import numpy as np

# 디버그 모드 플래그
DEBUG_MODE = True  # 필요한 경우 True로 설정, 필요하지 않은 경우 False로 설정

# 이미지를 불러옴
image_path = 'board1.png'  # 이미지 파일 경로를 지정
image = cv2.imread(image_path)
# 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 적응형 이진화를 수행
binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=17, C=2
)

# Canny 엣지 검출을 수행.  낮은 임계값, 높은 임계값 설정 유의
edges = cv2.Canny(binary, 15, 250)
# 확률적 허프 변환을 사용하여 직선 검출
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=102, minLineLength=10, maxLineGap=10)

# 가로 및 세로 선
horizontal_lines = []
vertical_lines = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1)  # 직선의 기울기 각도 계산
    if -np.pi / 4 < angle < np.pi / 4:
        vertical_lines.append(line)
    else:
        horizontal_lines.append(line)

# 가로 및 세로 선이 만나는 지점을 계산
corners = []
for hl in horizontal_lines:
    for vl in vertical_lines:
        x1_h, y1_h, x2_h, y2_h = hl[0]
        x1_v, y1_v, x2_v, y2_v = vl[0]
        A = np.array([[y2_h - y1_h, x1_h - x2_h], [y2_v - y1_v, x1_v - x2_v]])
        b = np.array([x1_h * (y2_h - y1_h) - y1_h * (x2_h - x1_h), x1_v * (y2_v - y1_v) - y1_v * (x2_v - x1_v)])
        intersection = np.linalg.solve(A, b)
        corners.append(intersection.astype(int))


# 교차점을 루트로 계산하여 출력
grid_size = int(np.sqrt(len(corners)))
print(f"{grid_size-1} x {grid_size-1}")

cv2.imshow('Checker', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


