import cv2
import numpy as np
import sys

# 이미지를 불러옴
# 명령형 인자로부터 이미지 경로를 받습니다.
if len(sys.argv) > 1:
    image_path = sys.argv[1]  # 첫 번째 인자가 이미지 경로입니다.
else:
    print("Usage: python script.py <image_path>")
    sys.exit(1)

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

# 이미지의 모서리 좌표
top_left = np.array([0, 0])
top_right = np.array([image.shape[1], 0])
bottom_left = np.array([0, image.shape[0]])
bottom_right = np.array([image.shape[1], image.shape[0]])

corners = np.array(corners)

# 각 모서리에 가장 가까운 교차점 찾기
def find_closest_corner(corner, all_corners):
    distances = np.linalg.norm(all_corners - corner, axis=1)
    return all_corners[np.argmin(distances)]


closest_top_left = find_closest_corner(top_left, corners)
closest_top_right = find_closest_corner(top_right, corners)
closest_bottom_left = find_closest_corner(bottom_left, corners)
closest_bottom_right = find_closest_corner(bottom_right, corners)

# 결과 꼭짓점
extreme_corners = [
    closest_top_left,
    closest_top_right,
    closest_bottom_right,
    closest_bottom_left
]

width = 300
height = 300
dstQuad = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
pers = cv2.getPerspectiveTransform(np.array(extreme_corners, dtype=np.float32), dstQuad)  # 투영 변환 행렬 계산
dst = cv2.warpPerspective(image, pers, (width, height))  # 투영 변환 행렬 적용

# Canny 엣지 검출을 수행하세요.
edges = cv2.Canny(dst, 0, 254)

# 허프 변환을 사용하여 엣지에서 원 검출
circles = cv2.HoughCircles(
    edges,
    cv2.HOUGH_GRADIENT,  # Hough 변환 방법
    dp=1,  # 이미지 해상도에 대한 역비율
    minDist=10,  # 원 사이의 최소 거리
    param1=50,  # Canny 엣지 검출기의 고장도 임계값
    param2=21,  # 작으면 더 많은 원을 검출, 크면 정확한 원 검출
    minRadius=5,  # 원의 최소 반지름
    maxRadius=20  # 원의 최대 반지름
)

# 밝은 원과 어두운 원의 개수를 저장할 변수 초기화
w = 0
b = 0
# 임계값 설정 1
threshold = 150

# 검출된 원이 있는 경우
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # 원 중심의 그레이스케일 값 확인
        pixel_value = gray[y, x]

        # 밝기에 따라 밝은 원과 어두운 원 분류
        if pixel_value > threshold:
            w += 1
        else:
            b += 1

        cv2.circle(dst, (x, y), r, (0, 255, 0), 4)  # 원 그리기
        cv2.circle(dst, (x, y), 2, (0, 128, 255), -1)  # 중심점 그리기

print('w:', w, 'b:', b)

# 결과 이미지 표시
cv2.imshow("Detected Circles", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
