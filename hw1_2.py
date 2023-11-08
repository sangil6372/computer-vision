import cv2  # OpenCV 라이브러리 추가
import numpy as np  # 배열 연산을 위해 numpy 라이브러리를 추가

#교재의 코드8-7 영상의 투시 변환 예제[ch08/perpective] 코드를 참고하였습니다.

# 마우스 콜백 함수
def on_mouse(event, x, y, _, __):
    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 클릭여부 확인
        if len(srcQuad) < 4: # 4점 선택여부 확인
            srcQuad.append([x, y])  # 점을 리스트의 형태로 추가
            cv2.circle(src, (x, y), 5, (0, 255, 0), -1) #선택된 지점에 원을 그림
            cv2.imshow('src', src)  # 변경된 이미지 화면에 재표시

            # 4개의 점이 모두 선택되면 변환을 실행합니다.
            if len(srcQuad) == 4:
                # dstQuad를 올바른 순서로 설정합니다.
                dstQuad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
                srcQuad_np = np.array(srcQuad, dtype=np.float32)  # srcQuad를 NumPy 배열로 변환합니다.
                pers = cv2.getPerspectiveTransform(srcQuad_np, dstQuad) # 투영 변환 행렬 계산
                dst = cv2.warpPerspective(src, pers, (w, h)) # 투영 변환 행렬 적용
                cv2.imshow('dst', dst)


# 이미지와 변수 초기화
src = cv2.imread('img.png')
srcQuad = []  # srcQuad를 리스트로 초기화
w, h = 300, 300  # 결과 이미지의 너비와 높이를 정사각형으로 설정

# 이미지가 제대로 로드되었는지 확인
if src is None:
    print('Image load failed!')  #실패 시 출력하고 종료
    exit()

cv2.namedWindow('src')    # src 이름의 윈도우 설정
cv2.setMouseCallback('src', on_mouse) # src 윈도우에 마우스 콜백함수 설정

cv2.imshow('src', src)
cv2.waitKey(0)
cv2.destroyAllWindows()
