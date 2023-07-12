import unittest
import numpy as np
import torch
import torch.nn as nn
from cleverfunc import clever_distance, recover_audio


# 测试clever_distance和recover_audio函数
class TestAudio(unittest.TestCase):
    # 测试clever_distance函数
    def setUp(self):
        # 定义一个简单的神经网络
        self.model = nn.Sequential(
            nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2), nn.Sigmoid()
        )
        # 随机生成一个音频
        self.audio = np.random.rand(100, 10)
        self.epsilon = 0.1

    # 测试recover_audio函数
    def test_clever_distance(self):
        clever = clever_distance(self.model, self.audio, num_samples=20, k=5)
        self.assertIsInstance(clever, float)

    def test_recover_audio(self):
        recovered_audio = recover_audio(
            self.model, self.audio, self.epsilon, num_samples=20
        )
        self.assertIsInstance(recovered_audio, np.ndarray)  # recovered_audio是numpy数组
        self.assertEqual(
            recovered_audio.shape, self.audio.shape
        )  # recovered_audio的形状和原音频一致


if __name__ == "__main__":
    unittest.main()
