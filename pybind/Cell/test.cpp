#include <random>
#include <iostream>

// Rnd 类定义
class Rnd {
public:
    // 默认构造函数：使用随机设备种子
    Rnd() : mt_gen{std::random_device()()}, rnd_dist_double{0, 1.0} {}

    // 接受种子作为参数的构造函数
    explicit Rnd(unsigned int seed) : mt_gen{seed}, rnd_dist_double{0, 1.0} {}

    // 生成 [0, 1] 之间的 double 随机数
    double operator()() { return rnd_dist_double(mt_gen); }

    // 生成 [0, N] 之间的 int 随机数
    int getInt(int N) {
        std::uniform_int_distribution<int> dist(0, N);  // 指定范围
        return dist(mt_gen);
    }

protected:
    std::mt19937 mt_gen;  // Mersenne Twister 伪随机数生成器
    std::uniform_real_distribution<double> rnd_dist_double;  // double 随机数分布
};

// 目标类
class MyClass {
public:
    void generateRandom() {
        // 在成员函数中创建 Rnd 对象
        Rnd rnd;  // 默认构造：使用随机种子
        std::cout << "Random double (0-1): " << rnd() << std::endl;
        std::cout << "Random int (0-10): " << rnd.getInt(10) << std::endl;
    }

    void generateRandomWithSeed(unsigned int seed) {
        // 使用指定种子初始化
        Rnd rnd(seed);
        std::cout << "Random double with seed (0-1): " << rnd() << std::endl;
        std::cout << "Random int with seed (0-10): " << rnd.getInt(10) << std::endl;
    }
};

int main() {
    MyClass obj;
    obj.generateRandom();  // 使用默认种子
    obj.generateRandomWithSeed(42);  // 使用自定义种子

    return 0;
}
