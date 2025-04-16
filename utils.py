# utils.py
import re
import string

def clean_text(text):
    """
    清理文本数据：
    1. 转为小写
    2. 移除方括号及其内容 (例如 [abc])
    3. 移除所有非字母数字空格的字符 (保留单词和空格)
    4. 移除网址 (http/https/www)
    5. 移除HTML标签
    6. 移除标点符号 (虽然第3步已处理大部分，这里再确认一遍)
    7. 移除换行符
    8. 移除包含数字的单词
    """
    text = str(text).lower() # 转为小写
    text = re.sub(r'\[.*?\]', '', text) # 移除方括号内容
    text = re.sub(r'\W', ' ', text) # 移除非字母数字下划线的字符，替换为空格
    # 如果只想保留中文、英文、数字和空格，可以使用更精确的正则：
    # text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # 移除URL
    text = re.sub(r'<.*?>+', '', text) # 移除HTML标签
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # 再次确认移除标点
    text = re.sub(r'\n', ' ', text) # 替换换行符为空格
    text = re.sub(r'\w*\d\w*', '', text) # 移除含数字的单词 (可能对中文影响不大)
    text = re.sub(r'\s+', ' ', text).strip() # 移除多余的空格并去除首尾空格
    return text