import cv2
import numpy as np

def detect_rectangles_from_video(video_path):
    cap = cv2.VideoCapture(0)  # 打开默认摄像头
    #cap = cv2.VideoCapture("http://admin:admin@192.168.10.5:8081/")
    cap = cv2.VideoCapture(video_path)  # 打开视频文件
    if not cap.isOpened():
        print("错误：无法打开视频文件。")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 调整每一帧的大小（例如，设置为1920x1200）
        frame_resized = cv2.resize(frame, (1280, 960))  # 设置图像大小为1920x1200

        # 获取图像的实际宽度和高度
        height, width = frame_resized.shape[:2]

        # 创建窗口
        cv2.imshow('Rectangle Detection', frame_resized)

        # 设置显示窗口的大小为图像的实际尺寸
        cv2.resizeWindow('Rectangle Detection', width, height)

        # 转为灰度图
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        # 高斯模糊，减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)

        # 找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # 轮廓多边形逼近
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # 判断是否为矩形（四个顶点且是凸的）
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > 1000:  # 确保矩形面积足够大
                    # 画出矩形轮廓，绿色
                    cv2.drawContours(frame_resized, [approx], -1, (0, 255, 0), 4)
                    # 计算矩形的质心
                    M = cv2.moments(approx)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(frame_resized, (cx, cy), 2, (0, 0, 255), -1)

                    # 根据矩形的宽度或高度自适应调整圆环半径
                    # 这里用矩形的宽度来计算半径
                    w, h = cv2.boundingRect(approx)[2:4]
                    radius = min(w, h) // 4  # 使用矩形宽度或高度的1/4作为半径
                    for r in [radius, int(radius * 1.5),radius * 2]:
                        cv2.circle(frame_resized, (cx, cy), r, (0, 0, 255), 1)
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.putText(frame_resized, 'Rectangle', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Rectangle Detection', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_rectangles_from_video('input.mp4')
