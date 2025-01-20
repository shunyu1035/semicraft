cimport sputter_yield  # 引用 Cython 声明
import absorb

def test():
    print(absorb.rn_coeffcients)


def example_usage_cimport():

    # 调用 sputter_yield_angle
    yield_hist = sputter_yield.sputter_yield(0.5, 0.2, 10, 2)
    print("Sputter yield histogram:", yield_hist)

    return yield_hist
