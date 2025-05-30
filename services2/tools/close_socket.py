import socket

def send_msg(message):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 8088))   # 连接本地socket端口

    # 发送字符串数据到服务器端
    # message = "I'm fine, thanks."
    client_socket.send(message.encode('utf-8'))

    data = client_socket.recv(1024).decode('utf-8')
    print('res:', data)
    client_socket.close()
    return data

def closeSocket():
    send_msg('Terminate')


if __name__ == '__main__':
    closeSocket()