# ws_server.py
import asyncio
import websockets
import os
from datetime import datetime

UPLOAD_FOLDER = './ws_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

async def handle_video(websocket, path):
    try:
        # 接收文件名
        filename = await websocket.recv()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{filename}")
        
        # 接收文件数据
        with open(save_path, 'wb') as f:
            while True:
                data = await websocket.recv()
                if data == b'END':
                    break
                f.write(data)
        
        await websocket.send(f"File saved to {save_path}")
        
    except Exception as e:
        await websocket.send(f"Error: {str(e)}")

start_server = websockets.serve(
    handle_video,
    '::',  # 监听所有IPv6地址
    8765,
    ssl=None  # 实际使用时应该配置SSL
)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()