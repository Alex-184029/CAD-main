def quick_sort(arr):
    if (len(arr)) < 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

def get_iou(rect1, rect2):
    x11, y11, x12, y12 = rect1
    x21, y21, x22, y22 = rect2

    x1 = max(x11, x21)
    y1 = max(y11, y21)
    x2 = min(x12, x22)
    y2 = min(y12, y22)

    if x1 < x2 and y1 < y2:
        return (x1, y1, x2, y2)
    else:
        return None


arr = [1, 3, 5, 4, 2, 6, 7]
print(*arr)
sorted_arr = quick_sort(arr)
print(*sorted_arr)