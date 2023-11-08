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

# 엣지 이미지에서 컨투어를 찾으세요.
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

if DEBUG_MODE:
    # 컨투어를 그릴 빈 이미지를 생성합니다.
    contour_image = np.zeros_like(image)

    # 찾은 컨투어를 그립니다.
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # 화면에 출력합니다.
    cv2.imshow('Debug: Contours', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 찾은 컨투어 중에서 가장 큰 컨투어를 선택하세요.
largest_contour = max(contours, key=cv2.contourArea)

# 컨투어 근처에서 허프 변환을 사용하여 직선을 찾으세요.
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=10)


# 가로 및 세로 선을 선택하세요.
horizontal_lines = []
vertical_lines = []

for line in lines:
    rho, theta = line[0]
    if np.pi / 4 < theta < 3 * np.pi / 4:
        vertical_lines.append((rho, theta))
    else:
        horizontal_lines.append((rho, theta))

# 가로 및 세로 선이 만나는 지점을 계산하세요.
corners = []

for hl in horizontal_lines:
    for vl in vertical_lines:
        rho_h, theta_h = hl
        rho_v, theta_v = vl
        A = np.array([[np.cos(theta_h), np.sin(theta_h)], [np.cos(theta_v), np.sin(theta_v)]])
        b = np.array([rho_h, rho_v])
        intersection = np.linalg.solve(A, b)
        corners.append(intersection.astype(int))

# 결과를 출력하세요.
for corner in corners:
    cv2.circle(image, tuple(corner), 5, (0, 0, 255), -1)

cv2.imshow('Checkerboard Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
