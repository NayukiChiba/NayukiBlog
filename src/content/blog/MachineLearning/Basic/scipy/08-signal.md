---
title: SciPy 信号处理基础
date: 2026-01-16
category: MachineLearning/Basic/scipy
tags:
  - Python
  - SciPy
description: 掌握滤波器设计、傅里叶变换和峰值检测
image: https://img.yumeko.site/file/blog/ScipyLearning.jpg
status: public
---

# 信号处理基础

## 滤波器设计 (Filter Design)

滤波器用于从信号中提取或去除特定频率成分。

### 1. Butterworth 低通滤波器

```python
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# 生成带噪声的信号
fs = 1000  # 采样率
t = np.linspace(0, 1, fs)
# 10Hz的正弦波 + 100Hz的噪声
signal_clean = np.sin(2 * np.pi * 10 * t)
noise = 0.5 * np.sin(2 * np.pi * 100 * t)
signal_noisy = signal_clean + noise

# 设计Butterworth低通滤波器
order = 4         # 阶数
cutoff = 20       # 截止频率(Hz)
b, a = signal.butter(order, cutoff, btype='low', fs=fs)

# 应用滤波器
filtered = signal.filtfilt(b, a, signal_noisy)  # filtfilt零相移

print(f"原始信号长度: {len(signal_noisy)}")
print(f"滤波后信号长度: {len(filtered)}")
print(f"噪声去除效果: {np.corrcoef(signal_clean, filtered)[0,1]:.4f}")
```

**输出**:

```
原始信号长度: 1000
滤波后信号长度: 1000
噪声去除效果: 0.9998
```

**参数说明**:

- `order`: 阶数越高，过渡带越陡
- `cutoff`: 截止频率
- `btype`: `'low'`(低通), `'high'`(高通), `'band'`(带通), `'bandstop'`(带阻)

### 2. 其他滤波器类型

```python
# 高通滤波器(去除低频)
b_high, a_high = signal.butter(4, 50, btype='high', fs=1000)

# 带通滤波器(保留特定频段)
b_band, a_band = signal.butter(4, [40, 60], btype='band', fs=1000)

# Chebyshev滤波器(更陡的过渡带)
b_cheby, a_cheby = signal.cheby1(4, 0.5, 20, btype='low', fs=1000)

print("各种滤波器设计完成")
```

**应用**: 音频处理、生物信号、传感器数据清洗。

### 滤波器可视化

下图展示了滤波前后的时域对比和频率响应：

![08_filter](https://img.yumeko.site/file/articles/scipylearn/08_filter.png)

## 卷积 (Convolution)

卷积是信号处理的基本操作。

### 1. 一维卷积

```python
# 信号和系统冲激响应
x = np.array([1, 2, 3, 4, 5])
h = np.array([0.5, 0.5])  # 移动平均滤波器

# 全卷积
y_full = signal.convolve(x, h, mode='full')
print("全卷积(full):", y_full)
print("长度:", len(y_full), "= len(x) + len(h) - 1")

# 同长卷积
y_same = signal.convolve(x, h, mode='same')
print("\n同长卷积(same):", y_same)
print("长度:", len(y_same), "= len(x)")

# 有效卷积
y_valid = signal.convolve(x, h, mode='valid')
print("\n有效卷积(valid):", y_valid)
print("长度:", len(y_valid), "= len(x) - len(h) + 1")
```

**输出**:

```
全卷积(full): [0.5 1.5 2.5 3.5 4.5 2.5]
长度: 6 = len(x) + len(h) - 1

同长卷积(same): [0.5 1.5 2.5 3.5 4.5]
长度: 5 = len(x)

有效卷积(valid): [1.5 2.5 3.5 4.5]
长度: 4 = len(x) - len(h) + 1
```

### 2. 相关分析

```python
# 自相关
sig = np.array([1, 2, 1, 0, 1, 2, 1])
autocorr = signal.correlate(sig, sig, mode='full')
print("自相关:", autocorr)

# 互相关(寻找相似模式)
sig1 = np.array([1, 2, 3, 2, 1])
sig2 = np.array([0, 1, 2, 3, 2, 1, 0])
crosscorr = signal.correlate(sig2, sig1, mode='same')
print("互相关:", crosscorr)
print("最大相关位置:", np.argmax(crosscorr))
```

**应用**: 模式识别、时延估计、雷达信号处理。

### 卷积可视化

下图展示了卷积运算结果：

![08_conv](https://img.yumeko.site/file/articles/scipylearn/08_conv.png)

## 傅里叶变换 (FFT)

频率域分析是信号处理的核心。

```python
from scipy import fft

# 生成复合信号
fs = 1000
t = np.linspace(0, 1, fs)
sig = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)

# FFT
yf = fft.fft(sig)
xf = fft.fftfreq(len(t), 1/fs)

# 只看正频
xf_pos = xf[:len(xf)//2]
yf_pos = np.abs(yf[:len(yf)//2])

# 找到主频率
peaks_idx = np.argsort(yf_pos)[-2:]  # 最大的2个峰
freqs = xf_pos[peaks_idx]

print("检测到的主频率:", sorted(freqs))
print("真实频率: [50, 120] Hz")
```

**输出**:

```
检测到的主频率: [50.0, 120.0]
真实频率: [50, 120] Hz
```

**应用**: 频谱分析、噪声识别、音频处理。

### FFT 可视化

下图展示了时域信号和频谱分析：

![08_fft](https://img.yumeko.site/file/articles/scipylearn/08_fft.png)

## 峰值检测 (Peak Detection)

自动找到信号中的峰值点。

```python
# 生成带峰值的信号
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.3*np.sin(5*x) + 0.1*np.random.randn(100)

# 找峰值
peaks, properties = signal.find_peaks(y, height=0.5, distance=10)

print(f"找到 {len(peaks)} 个峰值")
print("峰值位置:", peaks)
print("峰值高度:", properties['peak_heights'])
```

**输出**:

```
627e到 3 个峰值
峰值位置: [16 47 79]
峰值高度: [0.87 0.95 0.82]
```

**参数**:

- `height`: 最小高度
- `distance`: 峰间最小距离
- `prominence`: 突出度
- `width`: 峰宽

**应用**: ECG分析、语音识别、色谱分析。

### 峰值检测可视化

下图展示了峰值检测结果：

![08_peaks](https://img.yumeko.site/file/articles/scipylearn/08_peaks.png)

## 窗函数 (Window Functions)

减少FFT的频谱泄漏。

```python
# 常用窗函数
windows = {
    'Hann': signal.hann(100),
    'Hamming': signal.hamming(100),
    'Blackman': signal.blackman(100),
    'Kaiser': signal.kaiser(100, beta=5)
}

for name, window in windows.items():
    print(f"{name} 窗: 中心值={window[50]:.3f}, 边缘值={window[0]:.3f}")
```

**输出**:

```
Hann 窗: 中心值=1.000, 边缘值=0.000
Hamming 窗: 中心值=1.000, 边缘值=0.080
Blackman 窗: 中心值=1.000, 边缘值=0.000
Kaiser 窗: 中心值=1.000, 边缘值=0.001
```

## 信号重采样

```python
# 上采样
x = np.linspace(0, 1, 10)
y = np.sin(2*np.pi*x)
y_up = signal.resample(y, 50)  # 10 -> 50点

print(f"原始采样点: {len(y)}")
print(f"上采样后: {len(y_up)}")

# 下采样
y_down = signal.resample(y_up, 10)  # 50 -> 10点
print(f"下采样后: {len(y_down)}")
print(f"重建误差: {np.max(np.abs(y - y_down)):.6f}")
```

**输出**:

```
原始采样点: 10
上采样后: 50
下采样后: 10
重建误差: 0.000012
```

## 练习

```bash
python Basic/Scipy/08_signal.py
```
