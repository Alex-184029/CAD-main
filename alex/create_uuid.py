import uuid

def test_uuid():
    uuid_v4 = uuid.uuid4()   # 基于随机数，冲突概率极低
    uuid_v1 = uuid.uuid1()   # 基于时间戳和mac地址
    print('uuid_v4:', uuid_v4, type(uuid_v4))
    print('uuid_v1:', uuid_v1, type(uuid_v1))


if __name__ == '__main__':
    test_uuid()

