import cv2
import json
import base64
from websockets.sync.client import connect
from threading import Thread

def send_frames(ws_url, camera_index=0):
    # 初始化摄像头
    cap = cv2.VideoCapture(camera_index)
    
    with connect(ws_url) as websocket:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # 压缩并编码为 base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # 构造消息
            message = {
                "frame": f"data:image/jpeg;base64,{frame_b64}",
                "timestamp": "2023-11-01T12:00:00"  # 可替换为实际时间戳
            }
            
            # 发送到服务端
            websocket.send(json.dumps(message))
            
            # 可选：接收服务端响应
            try:
                response = websocket.recv(timeout=0.1)
                print("Server response:", response)
            except:
                pass
            
            # 显示本地预览（可选）
            cv2.imshow('Client Preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 启动 WebSocket 客户端
    server_url = "wss://www.buptbnrc.xyz:9998//ws"  # 替换为实际地址
    Thread(target=send_frames, args=(server_url,)).start()
