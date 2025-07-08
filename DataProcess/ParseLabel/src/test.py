import re

def match_fruits(text):
    """
    使用正则表达式匹配包含"apple"或"orange"，但不包含"banana"和"grape"的字符串
    
    参数:
    text (str): 需要匹配的文本
    
    返回:
    bool: 匹配成功返回True，否则返回False
    """
    # 正则表达式：正向肯定断言匹配"apple"或"orange"，负向前瞻断言排除"banana"和"grape"
    pattern = r'(?=.*(apple|orange))(?!.*banana)(?!.*grape)'
    
    # 编译正则表达式以提高性能
    regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    
    return bool(regex.search(text))

def main():
    """示例用法"""
    test_cases = [
        "I like apple eihei",           # 匹配
        "orange juice",           # 匹配
        "apple and grape",        # 不匹配（包含grape）
        "banana and orange",      # 不匹配（包含banana）
        "grape and banana",       # 不匹配（包含两者）
        "An Apple a day",         # 匹配（忽略大小写）
        "apple\norange",          # 匹配（跨行有效）
        "kiwi and strawberry",    # 不匹配（无apple或orange）
    ]
    
    for text in test_cases:
        result = match_fruits(text)
        print(f"{text!r}: {result}")

if __name__ == "__main__":
    main() 