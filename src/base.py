# base.py
class Base:
    def __init__(self, data):
        self.data = data  # 关键数据
        self.result = None  # 运算结果的存储位置

    def show_data(self):
        print(f"Data: {self.data}")
