# operations.py
from base import Base

class Operations(Base):
    def calculate_sum(self):
        self.result = sum(self.data)
        print(f"Sum: {self.result}")

    def calculate_product(self):
        self.result = 1
        self.test = 1
        for num in self.data:
            self.result *= num
        print(f"Product: {self.result}")
