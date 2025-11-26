import torch

from csrc import test_bit_convert


def check_num(x: float):
    v = torch.tensor(x, device="cuda", dtype=torch.float32)
    a, b = test_bit_convert(v)
    a = a.item()
    b = b.item()

    print(x, a - x, b - x)

    assert abs(a - x) < 0.02
    assert abs(b - x) < 0.02


check_num(0)

for i in range(-10, 10):
    check_num(2 ** (-i / 4))
    check_num(-(2 ** (-i / 2)))
