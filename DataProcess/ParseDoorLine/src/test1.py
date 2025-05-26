import sys

def calcHamming(a: str, b: str):
    hw_a = len([ai for ai in a if ai != ' ' and ai != '0'])
    hw_b = len([bi for bi in b if bi != ' ' and bi != '0'])
    print(hw_a, hw_b)
    if len(a) != len(b):
        print(-1)
    hd = len([i for i in range(len(a)) if a[i] != b[i]])
    print(hd)


if __name__ == '__main__':
    a = sys.stdin.readline().strip()
    b = sys.stdin.readline().strip()
    calcHamming(a, b)
    

