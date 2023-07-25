import torch
from torch.autograd.variable import Variable
from torchvision.utils import save_image
import librosa
import numpy as np
import soundfile as sf
from my_gan_model import MyGANModel  # 导入自定义的我的GAN模型


# 使用第二大可能性恢复标签
def recover_label_with_second_probability(model, denoised_mel_spectrogram):
    probabilities = model(denoised_mel_spectrogram)  # 获取模型对去噪音频的预测概率
    sorted_probabilities, sorted_labels = torch.sort(probabilities, descending=True)  # 按概率值降序排列
    second_most_probable_label = sorted_labels[1]  # 获取第二大可能性的标签
    return second_most_probable_label


# 计算CLEVER距离（计算每个类别与输入之间的距离）
# def compute_clever_distance(model, input, num_classes, norm):
#     distances = []
#     for target_class in range(num_classes):
#         distance = torch.norm(input - input)  # 这里的逻辑存在问题，因为这样计算的结果始终为0，需要检查
#         distances.append(distance)
#     return distances

# 模型输出是一个向量，其中每个元素表示对应类别的预测值，且模型可以处理批量输入，即可以一次性处理多个输入。
# 在计算CLEVER距离时使用了一个名为norm的参数，但是在计算距离时并没有用到。
# 假设norm是用于计算范数的参数，所以在调用torch.norm时使用了它。如果norm的含义不是用于计算范数的参数，则需要根据其实际含义进行修改。
def compute_clever_distance(model, input, num_classes, norm):
    distances = []
    for target_class in range(num_classes):
        output = model(input)  # 使用模型预测输入的输出
        class_output = output[:, target_class]  # 获取目标类别的预测输出
        distance = torch.norm(input - class_output, p=norm)  # 计算输入和目标类别预测输出之间的距离
        distances.append(distance)
    return distances


# 使用CLEVER方法恢复标签
def recover_label_with_clever(model, denoised_mel_spectrogram, num_classes, norm):
    distances = compute_clever_distance(model, denoised_mel_spectrogram, num_classes, norm)  # 计算CLEVER距离
    recovered_label = torch.argmin(distances)  # 选择距离最小的标签作为恢复的标签
    return recovered_label


# 扰动的置信度（计算模型对扰动输入的置信度）
def compute_disturbed_confidence(model, input, num_samples=100, epsilon=0.01):
    # Generate random noise
    noise = torch.randn(num_samples, *input.shape).cuda() * epsilon

    # Add noise to the input
    disturbed_inputs = input + noise

    # Compute the model's predictions
    predictions = model(disturbed_inputs)

    # Compute the confidence of the predictions
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
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Optionally, you can apply some audio processing steps here, for example:
    # - Apply short-time Fourier transform (STFT) to convert the audio to the frequency domain
    # - Apply Mel-frequency cepstral coefficients (MFCC) to extract features from the audio

    # Convert the audio data to a Torch tensor and add an extra dimension for the batch size
    input_tensor = torch.tensor(y).unsqueeze(0)

    return input_tensor

# 用GAN去噪
def denoise_with_GAN(GAN_model, audio_file, C=0.5, weight_clever=0.7, weight_second_prob=0.3):
    y, sr = librosa.load(audio_file)  # 加载音频文件
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr)  # 计算音频的梅尔频谱
    mel_spectrogram_tensor = torch.from_numpy(mel_spectrogram).unsqueeze(0).float()  # 将梅尔频谱转换为张量

    if torch.cuda.is_available():
        mel_spectrogram_tensor = mel_spectrogram_tensor.cuda()  # 如果GPU可用，将张量放到GPU上

    mel_spectrogram_variable = Variable(mel_spectrogram_tensor, requires_grad=True)  # 创建Variable
    GAN_model.G.eval()  # 设置生成器为评估模式
    denoised_mel_spectrogram = GAN_model.G(mel_spectrogram_variable)  # 使用生成器去噪

    num_classes = GAN_model.num_classes
    norm = GAN_model.norm
    GAN_model.D.eval()  # 设置鉴别器为评估模式

    # 使用CLEVER方法恢复标签
    recovered_label_clever = recover_label_with_clever(GAN_model.D, denoised_mel_spectrogram, num_classes, norm)
    print("Recovered label with CLEVER method:", recovered_label_clever)
    # 使用第二大概率恢复标签
    recovered_label_second_prob = recover_label_with_second_probability(GAN_model.D, denoised_mel_spectrogram)
    print("Recovered label with second probability method:", recovered_label_second_prob)

    # 计算输出和权重
    outputs = [recovered_label_clever.item(), recovered_label_second_prob.item()]
    weights = [weight_clever, weight_second_prob]

    # 使用恢复系统得到最终的恢复标签
    recovered_label = recovery_system(outputs, weights, C)
    print("Final recovered label:", recovered_label)

    # 保存去噪后的梅尔频谱图像
    save_image(denoised_mel_spectrogram.data, 'denoised_mel_spectrogram.png')
    # 将去噪后的梅尔频谱转换为numpy数组，并去掉多余的维度
    denoised_mel_spectrogram_np = denoised_mel_spectrogram.cpu().detach().numpy().squeeze()
    # 从去噪后的梅尔频谱回转为音频
    denoised_audio = librosa.feature.inverse.mel_to_audio(denoised_mel_spectrogram_np, sr=sr)
    # 保存去噪后的音频
    sf.write('denoised_audio.wav', denoised_audio, sr)


if __name__ == "__main__":
    gan_model = MyGANModel()  # 实例化GAN模型
    checkpoint = torch.load('gan_model.pt')  # 加载模型的权重
    gan_model.load_state_dict(checkpoint['model_state_dict'])  # 设置模型的状态字典

    audio_file = "path/to/audio/file.wav"  # 音频文件路径
    denoised_audio = denoise_with_GAN(gan_model, audio_file)  # 使用GAN模型去噪音频文件

    # 假设你有一个函数可以将去噪后的音频转化为输入张量
    input = prepare_input(denoised_audio)

    # Compute the outputs using your three methods
    # You should replace these functions with your real functions
    label_with_second_probability = recover_label_with_second_probability(input, gan_model)
    label_with_clever = recover_label_with_clever(input, gan_model)
    label_with_disturbed_confidence = compute_disturbed_confidence(gan_model, input)

    # Specify the weights for each method
    weight_for_second_probability = 0.3
    weight_for_clever = 0.3
    weight_for_disturbed_confidence = 0.4

    # Apply the recovery system
    outputs = [label_with_second_probability, label_with_clever, label_with_disturbed_confidence]
    weights = [weight_for_second_probability, weight_for_clever, weight_for_disturbed_confidence]
    threshold = 0.5  # You may need to adjust this value based on your specific requirements
    recovered_label = recovery_system(outputs, weights, threshold)
