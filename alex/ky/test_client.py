# ws_client.py
import asyncio
import websockets
import os

async def upload_video(file_path, server_uri):
    try:
        async with websockets.connect(server_uri) as websocket:
            # 发送文件名
            await websocket.send(os.path.basename(file_path))
            
            # 发送文件数据
            with open(file_path, 'rb') as f:
                while True:
                    data = f.read(1024 * 1024)  # 1MB chunks
                    if not data:
                        break
                    await websocket.send(data)
            
            # 发送结束标记
            await websocket.send(b'END')
            
            # 接收响应
            response = await websocket.recv()
            print(response)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    # 配置
    VIDEO_FILE = 'test.mp4'
    SERVER_IPV6 = '[::1]'  # 替换为服务器IPv6地址
    SERVER_PORT = 8765
    
    server_uri = f"ws://{SERVER_IPV6}:{SERVER_PORT}"
    
    asyncio.get_event_loop().run_until_complete(upload_video(VIDEO_FILE, server_uri))