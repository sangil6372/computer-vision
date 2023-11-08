import cv2
import sys
import numpy as np

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
    closest_bottom_left,
    closest_bottom_right
]


def order_points(pts):
    # 네 점을 정렬하는 함수 (상단 왼쪽, 상단 오른쪽, 하단 오른쪽, 하단 왼쪽 순서)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


ordered_corners = order_points(np.array(extreme_corners))

w = 300
h = 300
dstQuad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
pers = cv2.getPerspectiveTransform(ordered_corners, dstQuad)  # 투영 변환 행렬 계산
dst = cv2.warpPerspective(image, pers, (w, h))  # 투영 변환 행렬 적용
cv2.imshow('transformed img', dst)

# 결과 출력
cv2.imshow("original img", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
