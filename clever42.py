import torch
from torch.autograd.variable import Variable
from torchvision.utils import save_image
import librosa
import numpy as np

def denoise_with_GAN(GAN_model, audio_file):
    # 使用librosa加载音频文件
    y, sr = librosa.load(audio_file)

    # 将音频转换为Mel频谱图
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr)

    # 将Mel频谱图转换为适合模型输入的张量
    mel_spectrogram_tensor = torch.from_numpy(mel_spectrogram).unsqueeze(0).float()

    if torch.cuda.is_available():
        mel_spectrogram_tensor = mel_spectrogram_tensor.cuda()

    # 将Mel频谱图转换为 Variable
    mel_spectrogram_variable = Variable(mel_spectrogram_tensor, requires_grad=True)

    # 使用GAN模型的生成器去除对抗样本的噪声
    denoised_mel_spectrogram = GAN_model.G(mel_spectrogram_variable)

    # 将去噪后的Mel频谱图保存为图像
    save_image(denoised_mel_spectrogram.data, 'denoised_mel_spectrogram.png')

    # 将去噪后的Mel频谱图转换回numpy数组
    denoised_mel_spectrogram_np = denoised_mel_spectrogram.cpu().detach().numpy().squeeze()

    # 使用Griffin-Lim算法将去噪后的Mel频谱图转换回音频
    denoised_audio = librosa.feature.inverse.mel_to_audio(denoised_mel_spectrogram_np, sr=sr)

    # 保存去噪后的音频
    librosa.output.write_wav('denoised_audio.wav', denoised_audio, sr)

    return denoised_audio