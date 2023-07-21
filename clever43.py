import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torchvision.utils import save_image
import librosa
import numpy as np
import soundfile as sf


def recover_label_with_second_probability(model, denoised_mel_spectrogram):
    # 使用第二概率法恢复标签
    probabilities = model(denoised_mel_spectrogram)
    sorted_probabilities, sorted_labels = torch.sort(probabilities, descending=True)
    second_most_probable_label = sorted_labels[1]
    return second_most_probable_label


def compute_clever_distance(model, input, num_classes, norm):
    # 计算CLEVER距离
    distances = []
    for target_class in range(num_classes):
        # 输入对抗样本
        adversarial_example = input

        # 计算CLEVER距离
        distance = torch.norm(input - adversarial_example)
        distances.append(distance)

    return distances


def recover_label_with_clever(model, denoised_mel_spectrogram, num_classes, norm):
    # 使用CLEVER方法恢复标签
    distances = compute_clever_distance(model, denoised_mel_spectrogram, num_classes, norm)
    recovered_label = torch.argmin(distances)
    return recovered_label


# 定义C值和恢复方法的权重

# C：阈值，用于判断是否对对抗样本进行恢复。
# weight_clever：CLEVER 方法的权重。
# weight_second_prob：第二概率方法的权重。
C = 0.5
weight_clever = 0.7
weight_second_prob = 0.3


def recovery_system(outputs, weights, C):
    # 计算Judge_R
    Judge_R = [output * weight for output, weight in zip(outputs, weights)]
    max_judge_r = max(Judge_R)

    if max_judge_r < C:
        # Judge_R小于阈值C，拒绝对该对抗样本进行恢复
        print("Sample rejected, send it for manual review or discard.")
        return None
    else:
        # 找到所有可能的分类标签
        possible_labels = [output for output, judge_r in zip(outputs, Judge_R) if judge_r == max_judge_r]
        if not possible_labels:
            print("No possible labels found.")
            return None
        # 随机选取一个分类标签作为恢复结果
        true_label = np.random.choice(possible_labels)
        print(f"Recovered label: {true_label}")
        return true_label


# 定义GAN模型
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class MyGANModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super(MyGANModel, self).__init__()
        self.G = Generator(input_dim, hidden_dim, output_dim)
        self.D = Discriminator(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        return self.D(self.G(x))


# 加载GAN模型和音频文件

# GAN_model：GAN 模型，用于去除音频文件中的噪声。
# audio_file：音频文件，需要进行去噪和恢复标签。
GAN_model = MyGANModel()
audio_file = "path/to/audio/file.wav"


def denoise_with_GAN(GAN_model, audio_file):
    # 加载音频文件并转换为Mel频谱图
    y, sr = librosa.load(audio_file)
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr)
    mel_spectrogram_tensor = torch.from_numpy(mel_spectrogram).unsqueeze(0).float()
    if torch.cuda.is_available():
        mel_spectrogram_tensor = mel_spectrogram_tensor.cuda()
    mel_spectrogram_variable = Variable(mel_spectrogram_tensor, requires_grad=True)

    # 使用GAN模型去除噪声
    GAN_model.G.eval()
    denoised_mel_spectrogram = GAN_model.G(mel_spectrogram_variable)

    # 使用CLEVER方法恢复标签
    num_classes = ...
    norm = ...
    GAN_model.D.eval()
    recovered_label_clever = recover_label_with_clever(GAN_model.D, denoised_mel_spectrogram, num_classes, norm)
    print("Recovered label with CLEVER method:", recovered_label_clever)

    # 使用第二概率恢复标签
    recovered_label_second_prob = recover_label_with_second_probability(GAN_model.D, denoised_mel_spectrogram)
    print("Recovered label with second probability method:", recovered_label_second_prob)

    # 使用CLEVER方法和第二概率方法恢复标签
    outputs = [recovered_label_clever.item(), recovered_label_second_prob.item()]
    weights = [weight_clever, weight_second_prob]

    # 使用加权投票系统恢复标签
    recovered_label = recovery_system(outputs, weights, C)
    print("Final recovered label:", recovered_label)

    # 保存去噪后的Mel频谱图并转换回音频
    save_image(denoised_mel_spectrogram.data, 'denoised_mel_spectrogram.png')
    denoised_mel_spectrogram_np = denoised_mel_spectrogram.cpu().detach().numpy().squeeze()
    denoised_audio = librosa.feature.inverse.mel_to_audio(denoised_mel_spectrogram_np, sr=sr)
    sf.write('denoised_audio.wav', denoised_audio, sr)


# 使用GAN模型和音频文件进行去噪
denoise_with_GAN(GAN_model, audio_file)
