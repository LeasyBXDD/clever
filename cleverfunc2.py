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