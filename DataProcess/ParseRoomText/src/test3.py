import sys

def int_to_binary_string(n: int) -> str:
    # 使用 bin() 函数将整数转换为二进制字符串，并去掉前缀 '0b'
    binary_string = bin(n)[2:]
    return binary_string

def is_101_in(x: int) -> bool:
    s = int_to_binary_string(x)
    return True if '101' in s else False


if __name__ == '__main__':
    s = sys.stdin.readline().strip()
    arr = s.split()
    a, b, cnt = int(arr[0]), int(arr[1]), 0
    for i in range(a, b + 1):
        if not is_101_in(i):
            cnt += 1
    print(cnt)
