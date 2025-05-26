import pickle

def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except pickle.UnpicklingError:
        print(f"文件 {file_path} 无法被解pickle。")
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误: {e}")

# 示例文件路径
file_path = r'E:\School\Grad1\CAD\DeepLearn\TextClassify\Chinese-Text-Classification-Pytorch\THUCNews\data\vocab.pkl'

# 读取文件内容
data = read_pickle_file(file_path)

# 打印文件内容
if data is not None:
    print("文件内容:", data)