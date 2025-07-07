import json

def get_configs():
    category_txt = './LegendLabels/data/class.txt'
    catelog_json = '../ParseLabel/data/classify/classify_catelog.json'
    category_map = dict()
    with open(category_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        category_map[i] = line.strip()
    with open(catelog_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    catelog_map = dict()
    for key, value in data.items():
        for v in value:
            catelog_map[v] = key
    return category_map, catelog_map

def test():
    map1, map2 = get_configs()
    print(len(map1), len(map2))
    list1 = list(map1.values())
    list2 = list(map2.keys())
    gap1 = [item for item in list1 if item not in list2]
    gap2 = [item for item in list2 if item not in list1]
    print('gap1:', gap1)
    print('gap2:', gap2)


if __name__ == '__main__':
    test()
