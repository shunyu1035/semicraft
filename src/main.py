# main.py
from base import Base
from operations import Operations

if __name__ == "__main__":
    # 初始化基类
    data_array = [1, 2, 3, 4, 5]

    base_instance = Base(data=data_array)
    base_instance.show_data()

    # 用子类对基类的self数据进行运算
    operations_instance = Operations(base_instance.data)
    operations_instance.calculate_sum()      # 计算总和
    operations_instance.calculate_product()  # 计算乘积

    print(operations_instance.test)
