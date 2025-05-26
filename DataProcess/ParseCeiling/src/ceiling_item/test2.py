fruits = dict()

fruits['banana'] = {'price': 20}
fruits['grape'] = {'price': 10}
fruits['apple'] = {'price': 20}

fruits = dict(sorted(fruits.items(), key=lambda item: item[1]["price"]))

print('fruits:', fruits)
