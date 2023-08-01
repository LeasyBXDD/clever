import torch
from torch.autograd.variable import Variable
from torchvision.utils import save_image
import librosa
import numpy as np
import soundfile as sf
from my_gan_model import MyGANModel, Generator, Discriminator  # 导入自定义的我的GAN模型
import torch.optim as optim

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 提前设定的权重
weight_clever = 0.7
weight_second_prob = 0.3


# 使用第二大可能性恢复标签
def recover_label_with_second_probability(model, denoised_mel_spectrogram):
    probabilities = model(denoised_mel_spectrogram)  # 获取模型对去噪音频的预测概率
    sorted_probabilities, sorted_labels = torch.sort(probabilities, descending=True)  # 按概率值降序排列
    second_most_probable_label = sorted_labels[1]  # 获取第二大可能性的标签
    return second_most_probable_label


# C&W攻击方法
def CW_attack(model, input, target, num_steps=1000, learning_rate=0.01):
    """执行 Carlini & Wagner (C&W) 攻击方法。"""
    # 初始化扰动为零
    perturbation = torch.zeros_like(input).cuda()
    perturbation.requires_grad = True

    # 定义优化器
    optimizer = optim.Adam([perturbation], lr=learning_rate)

    for step in range(num_steps):
        # 计算扰动后的输入
        perturbed_input = input + perturbation

        # 计算模型的输出
        output = model(perturbed_input)

        # 计算损失
        loss = torch.nn.CrossEntropyLoss()(output, target)

        # 反向传播
        loss.backward()

        # 更新扰动
        optimizer.step()

        # 投影扰动到 L2 球（可选）
        perturbation.data = project_to_l2_ball(perturbation.data)

    # 计算扰动后的输入
    perturbed_input = input + perturbation

    return perturbed_input


def project_to_l2_ball(x, eps=1.0):
    """将输入张量投影到 L2 球上。"""
    norm = torch.norm(x)
    if norm > eps:
        x = x / norm * eps
    return x


# 使用C&W攻击来计算每个类别的距离
def compute_distance_with_cw_attack(model, input, num_classes):
    distances = []
    for target_class in range(num_classes):
        target = torch.tensor([target_class]).cuda()
        perturbed_input = CW_attack(model, input, target)
        distance = torch.norm(input - perturbed_input)
        distances.append(distance)
    return distances


# 使用C&W计算距离函数来恢复标签
def recover_label_with_cw_attack(model, denoised_mel_spectrogram, num_classes):
    distances = compute_distance_with_cw_attack(model, denoised_mel_spectrogram, num_classes)  # 计算C&W攻击的距离
    recovered_label = torch.argmin(distances)  # 选择距离最小的标签作为恢复的标签
    return recovered_label

# 扰动的置信度（计算模型对扰动输入的置信度）
def compute_disturbed_confidence(model, input, num_samples=100, epsilon=0.01):
    """计算模型对扰动输入的置信度。"""
    # 生成随机噪声
    noise = torch.randn(num_samples, *input.shape).cuda() * epsilon

    # 将噪声添加到输入中
    disturbed_inputs = input + noise

    # 计算模型的预测结果
    predictions = model(disturbed_inputs)

    # 计算预测结果的置信度
    confidence = predictions.max(dim=1)[0].mean().item()

    return confidence


# 采用恢复系统处理输出
def recovery_system(outputs, weights, threshold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluation_values = torch.tensor(outputs).to(device) * torch.tensor(weights).to(device)  # 计算评估值
    max_evaluation_value = torch.max(evaluation_values)  # 获取最大的评估值

    # 如果最大的评估值小于阈值threshold，则拒绝样本，进行手动审核或丢弃
    if max_evaluation_value < threshold:
        print("Sample rejected, send it for manual review or discard.")
        return None
    else:
        # 找出评估值等于最大评估值的输出标签
        possible_labels = [output for output, evaluation_value in zip(outputs, evaluation_values) if
                           evaluation_value == max_evaluation_value]
        if not possible_labels:
            print("No possible labels found.")
            return None
        # 从可能的标签中随机选择一个作为真实标签
        true_label = np.random.choice(possible_labels)
        print(f"Recovered label: {true_label}")
        return true_label


def prepare_input(audio_file):
    # 加载音频文件
    y, sr = librosa.load(audio_file, sr=None)

    # 可选：在此处应用一些音频处理步骤，例如：
    # - 对音频应用短时傅里叶变换（STFT）将其转换为频域
    # - 对音频应用梅尔频率倒谱系数（MFCC）提取其特征

    # 将音频数据转换为 Torch 张量，并为批处理大小添加一个额外的维度
    input_tensor = torch.tensor(y).unsqueeze(0)

    return input_tensor


# 用GAN去噪
def denoise_with_GAN(GAN_model, audio_file, C=0.5):
    y, sr = librosa.load(audio_file)  # 加载音频文件
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr)  # 计算音频的梅尔频谱
    mel_spectrogram_tensor = torch.from_numpy(mel_spectrogram).unsqueeze(0).float()  # 将梅尔频谱转换为张量

    mel_spectrogram_tensor = mel_spectrogram_tensor.to(device)  # 将张量放到正确的设备上

    mel_spectrogram_variable = Variable(mel_spectrogram_tensor, requires_grad=True)  # 创建Variable
    GAN_model.G.eval()  # 设置生成器为评估模式
    denoised_mel_spectrogram = GAN_model.G(mel_spectrogram_variable)  # 使用生成器去噪

    num_classes = GAN_model.num_classes

    GAN_model.D.eval()  # 设置鉴别器为评估模式

    # 使用C&W方法恢复标签
    recovered_label_cw_attack = recover_label_with_cw_attack(GAN_model.D, denoised_mel_spectrogram, num_classes)
    print("Recovered label with CLEVER method:", recovered_label_cw_attack)

    # 使用第二大概率恢复标签
    recovered_label_second_prob = recover_label_with_second_probability(GAN_model.D, denoised_mel_spectrogram)
    print("Recovered label with second probability method:", recovered_label_second_prob)

    # 计算输出和权重
    outputs = [recovered_label_cw_attack, recovered_label_second_prob]
    weights = [weight_clever, weight_second_prob]

    # 使用恢复系统得到最终的恢复标签
    recovered_label = recovery_system(outputs, weights, C)
    print("Final recovered label:", recovered_label)

    # 保存去噪后的梅尔频谱
    denoised_mel_spectrogram = denoised_mel_spectrogram.detach().cpu().numpy()
    denoised_audio = librosa.feature.inverse.mel_to_audio(denoised_mel_spectrogram)
    sf.write('denoised.wav', denoised_audio, sr)

    denoised_mel_spectrogram = GAN_model.G(mel_spectrogram_variable)  # 使用生成器去噪

    # 保存生成的图像
    save_image(denoised_mel_spectrogram.detach().cpu(), 'denoised.png')


# 主函数
def main():
    G = Generator(nz=100, ngf=64, nc=3).to(device)
    G.load_state_dict(torch.load("generator.pth"))

    D = Discriminator(nc=3, ndf=64).to(device)
    D.load_state_dict(torch.load("discriminator.pth"))

    GAN_model = MyGANModel(nz=100, ngf=64, ndf=64, nc=3).to(device)
    GAN_model.G = G
    GAN_model.D = D

    audio_file = 'noisy.wav'
    denoise_with_GAN(GAN_model, audio_file)


if __name__ == "__main__":
    main()
