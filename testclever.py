import unittest
import numpy as np
import torch
import torch.nn as nn
from cleverfunc import clever_distance, recover_audio


# 定义一个测试类 TestAudio，用于测试 clever_distance 和 recover_audio 函数
class TestAudio(unittest.TestCase):
    def setUp(self):
        # 假设我们有一个预训练模型的结构定义
        class PretrainedModel(nn.Module):
            def __init__(self):
                super(PretrainedModel, self).__init__()
                # 添加你的模型层
                pass

            def forward(self, x):
                # 定义你的前向传播
                pass

        # 实例化模型
        self.model = PretrainedModel()
        # 加载预训练模型的参数
        self.model.load_state_dict(torch.load("pretrained_model.pth"))
        # 随机生成一个音频
        self.audio = np.random.rand(100, 10)
        self.epsilon = 0.1

    # 测试 clever_distance 函数
    def test_clever_distance(self):
        # 计算 Clever 指数
        clever = clever_distance(self.model, self.audio, num_samples=20, k=5)
        # 判断 Clever 指数的类型是否为 float
        self.assertIsInstance(clever, float)

    # 测试 recover_audio 函数
    def test_recover_audio(self):
        # 恢复被扰动的音频信号
        recovered_audio = recover_audio(
            self.model, self.audio, self.epsilon, num_samples=20
        )
        # 判断恢复后的音频信号的类型是否为 numpy 数组
        self.assertIsInstance(recovered_audio, np.ndarray)
        # 判断恢复后的音频信号的形状是否与原音频一致
        self.assertEqual(
            recovered_audio.shape, self.audio.shape
        )


# 如果这个文件是主程序，就执行测试
if __name__ == "__main__":
    unittest.main()