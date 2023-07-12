import numpy as np
import torch
import torch.nn as nn
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)
from scipy.spatial.distance import cdist
import cleverhans


def clever_distance(model, audio, num_samples, k=5):
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

    random_inputs = torch.randn((num_samples,) + audio.shape)
    jacobian_random = jacobian(random_inputs)
    lipschitz_estimates = []
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            dist_inputs = cdist(random_inputs[i].numpy(), random_inputs[j].numpy())
            dist_outputs = cdist(jacobian_random[i].numpy(), jacobian_random[j].numpy())
            lipschitz_estimate = dist_outputs / dist_inputs
            lipschitz_estimates.append(lipschitz_estimate)

    k_smallest = np.partition(np.array(lipschitz_estimates), k)[:k]
    clever = np.mean(k_smallest)
    return clever


def recover_audio(model, audio, epsilon, num_samples=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    audio_tensor = torch.tensor(audio).to(device)
    clever = clever_distance(model, audio_tensor, num_samples)
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
    recovered_audio = audio_tensor - perturbation
    return recovered_audio.cpu().numpy()
