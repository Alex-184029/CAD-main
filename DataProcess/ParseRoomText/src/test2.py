import sys
from collections import defaultdict

def remove_duplicates_and_sort(nums):
    # 统计频率和第一次出现的索引
    frequency_map = defaultdict(int)    # 指定值类型，方便提供默认值
    first_index_map = {}
    for idx, num in enumerate(nums):
        frequency_map[num] += 1
        if num not in first_index_map:
            first_index_map[num] = idx

    # 去重
    unique_nums = list(set(nums))
    # 排序
    unique_nums.sort(key=lambda x: (-frequency_map[x], first_index_map[x]))
    # 打印
    n = len(unique_nums)
    for i in range(n - 1):
        print(unique_nums[i], end=',')
    print(unique_nums[n - 1])


if __name__ == '__main__':
    s = sys.stdin.readline().strip()
    nums = [int(i) for i in s.split(',')]
    remove_duplicates_and_sort(nums)
