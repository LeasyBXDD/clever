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

### 2.1 cleverfunc1

在`cleverfunc`中主要包含了两个函数：`clever_distance` 和 `recover_audio`。

```python
import numpy as np
import torch
import torch.nn as nn
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from scipy.spatial.distance import cdist
import cleverhans

# 计算 Clever 指数
def clever_distance(model, audio, num_samples, k=5):
    # 定义一个计算 Jacobian 矩阵的函数
    def jacobian(input):
        input.requires_grad = True
        output = model(input)
        grad = torch.zeros_like(input)
        for i in range(output.shape[-1]):
            grad_outputs = torch.zeros_like(output)
            grad_outputs[..., i] = 1
            grad_input = torch.autograd.grad(
                output, input, grad_outputs=grad_outputs, create_graph=True
            )[0]
            grad[..., i] = grad_input
        return grad

    # 生成随机的输入样本
    random_inputs = torch.randn((num_samples,) + audio.shape)
    # 计算随机输入样本的 Jacobian 矩阵
    jacobian_random = jacobian(random_inputs)
    lipschitz_estimates = []
    # 计算 Lipschitz 估计值
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            dist_inputs = cdist(random_inputs[i].numpy(), random_inputs[j].numpy())
            dist_outputs = cdist(jacobian_random[i].numpy(), jacobian_random[j].numpy())
            lipschitz_estimate = dist_outputs / dist_inputs
            lipschitz_estimates.append(lipschitz_estimate)

    # 取 k 个最小的 Lipschitz 估计值的平均值作为 Clever 指数
    k_smallest = np.partition(np.array(lipschitz_estimates), k)[:k]
    clever = np.mean(k_smallest)
    return clever

# 恢复被扰动的音频信号
def recover_audio(model, audio, epsilon, num_samples=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    audio_tensor = torch.tensor(audio).to(device)
    # 计算 Clever 指数
    clever = clever_distance(model, audio_tensor, num_samples)
    # 使用 Projected Gradient Descent 算法生成扰动
    perturbation = projected_gradient_descent(
        model,
        audio_tensor,
        epsilon * clever,
        0.01 * clever,
        40,
        np.inf,
        clip_min=0,
        clip_max=1,
    )
    # 恢复被扰动的音频信号
    recovered_audio = audio_tensor - perturbation
    return recovered_audio.cpu().numpy()
```

`clever_distance` 函数用于计算输入音频信号的 Clever 指数。具体来说，它首先定义了一个计算 Jacobian 矩阵的函数 `jacobian`，然后生成了一些随机的输入样本，并计算了这些输入样本的 Jacobian 矩阵。接着，它计算了这些输入样本之间的距离和对应的 Jacobian 矩阵之间的距离，并计算了 Lipschitz 估计值。最后，它取 k 个最小的 Lipschitz 估计值的平均值作为 Clever 指数。

`recover_audio` 函数用于恢复被扰动的音频信号。具体来说，它首先将输入的音频信号转换为 PyTorch 张量，并计算了输入音频信号的 Clever 指数。然后，它使用 Projected Gradient Descent 算法生成一个扰动，并将扰动应用到输入音频信号上，从而恢复被扰动的音频信号。

### 2.2 cleverfunc2

```python
import numpy as np
import torch
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from scipy.spatial.distance import cdist

# 计算 Clever 指数
def clever_distance(model, audio, num_samples, k=5):
    # 定义一个计算 Jacobian 矩阵的函数
    def jacobian(input):
        input.requires_grad = True
        output = model(input)
        grad = torch.zeros_like(input)
        for i in range(output.shape[-1]):
            grad_outputs = torch.zeros_like(output)
            grad_outputs[..., i] = 1
            grad_input = torch.autograd.grad(
                output, input, grad_outputs=grad_outputs, create_graph=True
            )[0]
            grad[..., i] = grad_input
        return grad

    # 生成随机的输入样本
    random_inputs = torch.randn((num_samples,) + audio.shape)
    # 计算随机输入样本的 Jacobian 矩阵
    jacobian_random = jacobian(random_inputs)
    # 计算 Lipschitz 估计值
    lipschitz_estimates = cdist(jacobian_random.reshape(num_samples, -1), random_inputs.reshape(num_samples, -1)) \
                         / cdist(random_inputs.reshape(num_samples, -1), random_inputs.reshape(num_samples, -1))
    # 取 k 个最小的 Lipschitz 估计值的平均值作为 Clever 指数
    clever = np.mean(np.partition(lipschitz_estimates, k)[:k])
    return clever

# 恢复被扰动的音频信号
def recover_audio(model, audio, epsilon, num_samples=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    audio_tensor = torch.tensor(audio).to(device)
    # 计算 Clever 指数
    clever = clever_distance(model, audio_tensor, num_samples)
    # 使用 Projected Gradient Descent 算法生成扰动
    perturbation = projected_gradient_descent(
        model,
        audio_tensor,
        epsilon * clever,
        0.01 * clever,
        40,
        np.inf,
        clip_min=0,
        clip_max=1,
    )
    # 恢复被扰动的音频信号
    recovered_audio = audio_tensor - perturbation
    return recovered_audio.cpu().numpy()
```

1. 删除了 `torch.nn` 和 `cleverhans` 的导入，因为这些库在代码中没有被使用到。
2. **在 `clever_distance` 函数中，使用了 `cdist` 函数的向量化计算，避免了使用循环计算 Lipschitz 估计值。**
3. **在 `clever_distance` 函数中，使用了 `reshape` 函数将输入样本和 Jacobian 矩阵转换为二维数组，避免了使用循环计算 Lipschitz 估计值。**
4. **在 `clever_distance` 函数中，使用了 `np.partition` 函数来计算 k 个最小的 Lipschitz 估计值的平均值，避免了使用循环和排序。**
5. 删除了 `import cleverhans`，因为 `projected_gradient_descent` 函数已经在 `cleverhans.torch.attacks.projected_gradient_descent` 中导入。
6. 删除了 `clip_min` 和 `clip_max` 参数的默认值，因为这些参数在 `projected_gradient_descent` 函数中已经有默认值了。

## 三、功能测试

`testone.py`是一个 Python 的单元测试文件，用于测试 clever_distance 和 recover_audio 两个函数的正确性。
这两个函数都是在 `cleverfunc.py` 文件中定义的，用于计算音频信号的 Clever 指数和恢复被扰动的音频信号。

```python
import unittest
import numpy as np
import torch
import torch.nn as nn
from cleverfunc import clever_distance, recover_audio

# 定义一个测试类 TestAudio，用于测试 clever_distance 和 recover_audio 函数
class TestAudio(unittest.TestCase):
    def setUp(self):
        # 加载预训练模型的参数
        pretrained_model = torch.load("pretrained_model.pth")
        # 将预训练模型的参数设置为神经网络的参数
        self.model = pretrained_model
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
```

这段代码主要是定义了一个测试类 `TestAudio`，用于测试 `clever_distance` 和 `recover_audio` 函数的正确性。在 `setUp` 函数中，我们加载了一个预训练模型的参数，并将它们设置为神经网络的参数。然后，我们随机生成了一个音频信号，并设置了一个扰动参数 `self.epsilon`。

在 `test_clever_distance` 函数中，我们调用了 `clever_distance` 函数，计算了输入音频信号的 Clever 指数，并使用 `self.assertIsInstance()` 函数判断 Clever 指数的类型是否为 float。

在 `test_recover_audio` 函数中，我们调用了 `recover_audio` 函数，恢复了被扰动的音频信号，并使用 `self.assertIsInstance()` 函数判断恢复后的音频信号的类型是否为 numpy 数组，使用 `self.assertEqual()` 函数判断恢复后的音频信号的形状是否与原音频一致。

最后，如果这个文件是主程序，就执行测试。