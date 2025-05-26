import random

def getWeight():
    rand_num = random.random()
    if rand_num < 0.93:
        return 1
    elif rand_num < 0.96:
        return 0.9
    else:
        return 0.8