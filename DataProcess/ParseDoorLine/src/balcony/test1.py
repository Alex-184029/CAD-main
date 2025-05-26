a = list()
a.append({'i': 1, 'X': 1824088.72, 'Y': -555887.25})
a.append({'i': 2, 'X': 1822118.72, 'Y': -555766.25})
a.append({'i': 3, 'X': 1824351.72, 'Y': -553771.25})
a.append({'i': 4, 'X': 1824900.72, 'Y': -559469.25})
a.append({'i': 5, 'X': 1821637.72, 'Y': -559626.25})
a.append({'i': 6, 'X': 1818600.72, 'Y': -559645.25})
a.append({'i': 7, 'X': 1818419.72, 'Y': -557400.25})
a.append({'i': 8, 'X': 1819945.72, 'Y': -555375.25})
a.append({'i': 9, 'X': 1824939.72, 'Y': -561796.25})

sorted_list = sorted(a, key=lambda x: x['Y'])
for item in sorted_list:
    print(item)
