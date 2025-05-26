import numpy as np

def point_to_line_distance(x0, y0, x1, y1, x2, y2):
    """
    计算点 (x0, y0) 到直线 (x1, y1) 和 (x2, y2) 的距离。

    :param x0: 点的x坐标
    :param y0: 点的y坐标
    :param x1: 直线上的第一个点的x坐标
    :param y1: 直线上的第一个点的y坐标
    :param x2: 直线上的第二个点的x坐标
    :param y2: 直线上的第二个点的y坐标
    :return: 点到直线的距离
    """
    # 定义向量 AB 和 AC
    AB = np.array([x2 - x1, y2 - y1])
    AC = np.array([x0 - x1, y0 - y1])
    
    # 计算向量 AB 和 AC 的叉积
    cross_product = np.cross(AB, AC)
    
    # 计算向量 AB 的模长
    magnitude_AB = np.linalg.norm(AB)
    
    # 计算点到直线的距离
    distance = np.abs(cross_product) / magnitude_AB
    
    return distance

def test():
    my_dict = {'apple': 19, 'banana': 31, 'grape': 10, 'orange': 25}
    sorted_dict = dict(sorted(my_dict.items(), key=lambda x: x[1]))
    print(sorted_dict)
   

if __name__ == '__main__':
    test()

