import pickle
from scipy.signal import butter, filtfilt
import pandas as pd
import re
import numpy as np


# 设计Butterworth滤波器
def butter_highpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    # 高通滤波
    b, a = butter_highpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    
    # 低通滤波
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, y)
    
    return y

def data_preparation(data_path):

    #每人有四个位置的pulse
    cun1 = pd.read_excel('')
    cun2 = pd.read_excel('')
    cun3 = pd.read_excel('')
    cun4 = pd.read_excel('')
    guan1 = pd.read_excel('')
    guan2 = pd.read_excel('')
    guan3 = pd.read_excel('')
    guan4 = pd.read_excel('')


    # 以条为单位进行训练并计算模型指标
    # 1左寸2右关3左关4右寸
    data = []
    label = []
    label_patient = []
    cun = pd.concat([cun1, cun2, cun3, cun4], ignore_index=True)
    guan = pd.concat([guan1, guan2, guan3, guan4], ignore_index=True)

    with open(data_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    for key in loaded_dict.keys():
        temp_data_dict = loaded_dict[key]
        temp_label = temp_data_dict['label']
        label_patient = label_patient + [temp_label]
        # 提取中文字符的正则表达式
        chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
        # 提取键中的中文字符
        name = ''.join(chinese_pattern.findall(key))
        feature1 = cun[(cun['姓名'].str.contains(name, na=False)) & (cun['左（右）手'] == 2)].head(1)
        feature1 = feature1.drop(columns=['姓名','标签']).values.squeeze()

        feature2 = guan[(guan['姓名'].str.contains(name, na=False)) & (guan['左（右）手'] == 1)].head(1)
        feature2 = feature2.drop(columns=['姓名','标签']).values.squeeze()

        feature3 = guan[(guan['姓名'].str.contains(name, na=False)) & (guan['左（右）手'] == 2)].head(1)
        feature3 = feature3.drop(columns=['姓名','标签']).values.squeeze()

        feature4 = cun[(cun['姓名'].str.contains(name, na=False)) & (cun['左（右）手'] == 1)].head(1)
        feature4 = feature4.drop(columns=['姓名','标签']).values.squeeze()


        for key, values in temp_data_dict.items():
            if key != 'label' and key[-1] == '1':
                label = label + [temp_label]
                values = (values - np.mean(values)).tolist() + feature1.tolist()
                data = data + [values]
            if key != 'label' and key[-1] == '2':
                label = label + [temp_label]
                values = (values - np.mean(values)).tolist() + feature2.tolist()
                data = data + [values]      
            if key != 'label' and key[-1] == '3':
                label = label + [temp_label]
                values = (values - np.mean(values)).tolist() + feature3.tolist()
                data = data + [values]
            if key != 'label' and key[-1] == '4':
                label = label + [temp_label]
                values = (values - np.mean(values)).tolist() + feature4.tolist()
                data = data + [values]

    data = np.array(data)
    data[:, :4000] = data[:, :4000]/np.std(data[:, :4000])
    data[:, :4000] = butter_bandpass_filter(data[:, :4000], lowcut=0.5, highcut=50, fs=250, order=4)
    data = np.copy(data)
    # data = data[:, 4000:]
    # 选择后27列进行归一化
    data_last_27 = data[:, -27:]
    # 对每一列进行归一化到 [0, 1]
    data_last_27_normalized = (data_last_27 - data_last_27.min(axis=0)) / (data_last_27.max(axis=0) - data_last_27.min(axis=0))
    # 将归一化后的数据替换回原始矩阵
    data[:, -27:] = data_last_27_normalized
    label = np.array(label)

    return data, label, label_patient