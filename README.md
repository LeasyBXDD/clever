# CLEVER

## 一、介绍
> CLEVER距离（Cross Lipschitz Extreme Value for nEtwork Robustness）以及音频对抗样本。

### 1.1 CLEVER距离

CLEVER距离是一种估计神经网络鲁棒性的方法。CLEVER度量了模型输出关于输入的敏感程度，即Lipschitz连续性。

### 1.2 音频对抗样本

对抗样本是一种攻击方法，通过对输入添加微小的扰动，使得神经网络误分类。音频对抗样本是针对语音识别和音频处理模型的对抗攻击。

### 1.3 本模块的目的
恢复模块的目的是消除音频对抗样本中的扰动，使模型能够正确识别音频。

## 二、使用方法

需要安装`cleverhans`库来使用这个示例。
可以使用`!pip install cleverhans`命令进行安装。

在`cleverfunc`中，首先定义了一个计算CLEVER距离的函数`clever_distance`。
它计算了模型输出关于输入的雅可比矩阵（Jacobian），然后使用随机输入对计算Lipschitz连续性。

接着，我们定义了`recover_audio`函数，它使用CLEVER距离和投影梯度下降（PGD）攻击来恢复音频。

要使用这个模块，需要提供一个预训练好的模型（如语音识别模型）和受到对抗攻击的音频。然后调用`recover_audio`函数来恢复音频。

## 三、功能测试

`testone.py`是一个 Python 的单元测试文件，用于测试 clever_distance 和 recover_audio 两个函数的正确性。
这两个函数都是在 `cleverfunc.py` 文件中定义的，用于计算音频信号的 Clever 指数和恢复被扰动的音频信号。

在这个测试文件中，我们首先定义了一个 TestAudio 类，它继承自 unittest.TestCase 类。
在这个类中，我们定义了两个测试函数 test_clever_distance 和 test_recover_audio，分别用于测试 clever_distance 和 recover_audio 函数的正确性。

在 setUp 函数中，我们定义了一个简单的神经网络模型 self.model 和一个随机生成的音频信号 self.audio，并设置了一个扰动参数 self.epsilon。

在 test_clever_distance 函数中，我们调用了 clever_distance 函数，并检查它的输出类型是否为浮点数。

在 test_recover_audio 函数中，我们调用了 recover_audio 函数，并检查它的输出类型是否为 numpy 数组，并且检查它的形状是否与原始音频信号的形状一致。

最后，在 if __name__ == "__main__": 语句中，我们调用了 unittest.main() 函数来执行这些测试函数。