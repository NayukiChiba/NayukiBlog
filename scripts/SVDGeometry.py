"""
SVD 三步变化可视化 (n=3 → m=2)

直观展示奇异值分解 A = U Σ V^T 的几何含义（A 为 2×3 矩阵）：
  输入:  R^3 中的单位球面
  V^T:  旋转/反射（球面形状不变，仅方向改变）
  Σ:    缩放 + 降维（3D 球面压扁为 2D 椭圆，半轴 = σ₁, σ₂)
  U:    在 R^2 中旋转椭圆

2×2 网格布局：上排 3D，下排 2D。

使用方法:
  python scripts/SVDGeometry.py

依赖:
  pip install numpy matplotlib
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

# 配置中文字体（Windows 下使用微软雅黑）
_chineseFonts = [f for f in fm.fontManager.ttflist
                 if 'Microsoft YaHei' in f.name or 'SimHei' in f.name]
if _chineseFonts:
    plt.rcParams['font.family'] = _chineseFonts[0].name
    plt.rcParams['axes.unicode_minus'] = False

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ====================== 数据生成 ======================

def generateSphereSurface(nTheta=60, nPhi=40):
    """生成 R^3 单位球面网格点"""
    theta = np.linspace(0, 2 * np.pi, nTheta)
    phi   = np.linspace(0, np.pi, nPhi)
    T, P = np.meshgrid(theta, phi)
    x = np.cos(T) * np.sin(P)
    y = np.sin(T) * np.sin(P)
    z = np.cos(P)
    shape = x.shape
    pointsFlat = np.stack([x.ravel(), y.ravel(), z.ravel()])
    return pointsFlat, theta, phi, shape


def generateSphereSamples(nPts=2000):
    """在 R^3 单位球面上均匀采样（Fibonacci 球）"""
    rng = np.random.default_rng(42)
    indices = np.arange(nPts)
    phi = np.arccos(1 - 2 * (indices + 0.5) / nPts)
    theta = np.pi * (1 + 5**0.5) * indices
    # 添加微小随机扰动使填充更自然
    phi += rng.normal(0, 0.003, nPts)
    theta += rng.normal(0, 0.003, nPts)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z])


def generateLatitudeLines(nLines=10, nPts=200):
    """生成球面纬线"""
    lines = []
    for phi in np.linspace(0.1, np.pi - 0.1, nLines):
        t = np.linspace(0, 2 * np.pi, nPts)
        lines.append(np.stack([
            np.cos(t) * np.sin(phi),
            np.sin(t) * np.sin(phi),
            np.full_like(t, np.cos(phi))
        ]))
    return lines


def generateMeridianLines(nLines=14, nPts=100):
    """生成球面经线"""
    lines = []
    for theta in np.linspace(0, 2 * np.pi, nLines, endpoint=False):
        p = np.linspace(0, np.pi, nPts)
        lines.append(np.stack([
            np.cos(theta) * np.sin(p),
            np.sin(theta) * np.sin(p),
            np.cos(p)
        ]))
    return lines


# ====================== 绘制函数 ======================

AXIS_COLORS = ['#E74C3C', '#27AE60', '#2980B9']  # 红 e1, 绿 e2, 蓝 e3
AXIS_LABELS = ['$e_1$', '$e_2$', '$e_3$']


def plotSphere3D(ax, pointsFlat, shape, title, axesVectors=None):
    """绘制 3D 球面（上排）"""
    xs = pointsFlat[0].reshape(shape)
    ys = pointsFlat[1].reshape(shape)
    zs = pointsFlat[2].reshape(shape)

    ax.plot_surface(xs, ys, zs, color='#85C1E9', alpha=0.22,
                    linewidth=0, antialiased=True, shade=True,
                    rstride=1, cstride=1)

    sR, sC = max(1, shape[0] // 15), max(1, shape[1] // 25)
    ax.plot_wireframe(xs, ys, zs, color='#1A5276', alpha=0.10,
                      linewidth=0.3, rstride=sR, cstride=sC)

    if axesVectors is not None:
        for i in range(3):
            v = axesVectors[:, i]
            ax.quiver(0, 0, 0, v[0], v[1], v[2],
                      color=AXIS_COLORS[i], linewidth=2.5,
                      arrow_length_ratio=0.10, alpha=0.92)

    r = 1.25
    ax.set_xlim([-r, r]); ax.set_ylim([-r, r]); ax.set_zlim([-r, r])
    ax.set_xlabel('$x_1$', fontsize=8, labelpad=-2)
    ax.set_ylabel('$x_2$', fontsize=8, labelpad=-2)
    ax.set_zlabel('$x_3$', fontsize=8, labelpad=-2)
    ax.set_title(title, fontsize=11, pad=12, fontweight='bold')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=22, azim=-55)


def plotEllipse2D(ax, scatterPts, title, axesVectors=None,
                  sigma=None, rotation=None,
                  latLines=None, merLines=None):
    """绘制 2D 椭圆投影（下排）"""
    # 纬线 / 经线投影
    if latLines is not None:
        for line in latLines:
            ax.plot(line[0], line[1], color='#85C1E9', alpha=0.15, lw=0.4)
    if merLines is not None:
        for line in merLines:
            ax.plot(line[0], line[1], color='#5DADE2', alpha=0.12, lw=0.4)

    # 散点填充
    ax.scatter(scatterPts[0], scatterPts[1], c='#85C1E9', s=0.6,
               alpha=0.20, edgecolors='none')

    # 椭圆边界虚线
    if sigma is not None:
        t = np.linspace(0, 2 * np.pi, 360)
        ell = np.stack([sigma[0] * np.cos(t), sigma[1] * np.sin(t)])
        if rotation is not None:
            ell = rotation @ ell
        ax.plot(ell[0], ell[1], '--', color='#C0392B', lw=2.2, alpha=0.75,
                label=f'$\\sigma_1={sigma[0]:.2f}$  '
                      f'$\\sigma_2={sigma[1]:.2f}$')
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.75,
                  edgecolor='#CCCCCC')

    # 基向量（仅 e1, e2，因为 m=2）
    if axesVectors is not None:
        for i in range(2):
            v = axesVectors[:, i]
            ax.arrow(0, 0, v[0], v[1], head_width=0.05, head_length=0.09,
                     fc=AXIS_COLORS[i], ec=AXIS_COLORS[i], lw=2.2,
                     alpha=0.88, length_includes_head=True, zorder=5)

    # 坐标轴
    m = max(np.max(np.abs(scatterPts)) * 1.22, 2.5)
    ax.set_xlim([-m, m]); ax.set_ylim([-m, m])
    ax.set_xlabel('$x_1$', fontsize=9)
    ax.set_ylabel('$x_2$', fontsize=9)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, pad=10, fontweight='bold')
    ax.grid(True, alpha=0.2, ls='--')
    ax.axhline(0, color='gray', lw=0.5, alpha=0.35)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.35)


# ====================== 主逻辑 ======================

def main():
    np.random.seed(42)

    # ---- 构造 2×3 矩阵 A 并计算 SVD ----
    A = np.array([
        [2.0, 0.8, 0.3],
        [0.4, 1.6, 0.2]
    ], dtype=float)

    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    # U: 2×2, s: [σ1, σ2], Vt: 3×3
    SigmaMat = np.zeros((2, 3))
    np.fill_diagonal(SigmaMat, s)

    assert np.allclose(A, U @ SigmaMat @ Vt), "SVD 重构失败"

    print("=" * 60)
    print("SVD 三步变化可视化 (n=3 → m=2)")
    print("=" * 60)
    print(f"\nA (2×3) =\n{A}")
    print(f"\nU (2×2) =\n{np.round(U, 4)}")
    print(f"σ = {np.round(s, 4)}")
    print(f"V^T (3×3) =\n{np.round(Vt, 4)}")
    print(f"\nU @ U^T =\n{np.round(U @ U.T, 4)}")
    print(f"V @ V^T =\n{np.round(Vt.T @ Vt, 4)}")

    # ---- 生成数据 ----
    spherePts, theta, phi, shape = generateSphereSurface(60, 40)
    samplePts = generateSphereSamples(2000)
    latLines3D = generateLatitudeLines(10, 200)
    merLines3D = generateMeridianLines(14, 100)
    axes3D = np.eye(3)

    # ---- 三步变换 ----
    # Step 1: V^T 旋转 (3D → 3D)
    s1Pts     = Vt @ spherePts
    s1Samples = Vt @ samplePts
    s1Lat     = [Vt @ L for L in latLines3D]
    s1Mer     = [Vt @ L for L in merLines3D]
    s1Ax      = Vt @ axes3D

    # Step 2: Σ 缩放 + 降维 (3D → 2D)
    s2Pts     = SigmaMat @ s1Pts
    s2Samples = SigmaMat @ s1Samples
    s2Lat     = [SigmaMat @ L for L in s1Lat]
    s2Mer     = [SigmaMat @ L for L in s1Mer]
    s2Ax      = SigmaMat @ s1Ax      # 2×3

    # Step 3: U 旋转 (2D → 2D)
    s3Pts     = U @ s2Pts
    s3Samples = U @ s2Samples
    s3Lat     = [U @ L for L in s2Lat]
    s3Mer     = [U @ L for L in s2Mer]
    s3Ax      = U @ s2Ax             # 2×3

    # ---- 基向量追踪 ----
    print(f"\n基向量追踪:")
    for i, label in enumerate(AXIS_LABELS):
        if i < 2:
            print(f"  {label}={axes3D[:,i]} → V^T → {np.round(s1Ax[:,i], 4)}"
                  f" → Σ → {np.round(s2Ax[:,i], 4)} → U → {np.round(s3Ax[:,i], 4)}")
        else:
            print(f"  {label}={axes3D[:,i]} → V^T → {np.round(s1Ax[:,i], 4)}"
                  f" → Σ → {np.round(s2Ax[:,i], 4)} (降维)")

    # ---- 绘图 (2×2 网格) ----
    fig = plt.figure(figsize=(13, 12), facecolor='white')
    gs = GridSpec(2, 2, figure=fig, wspace=0.10, hspace=0.28,
                  left=0.04, right=0.96, top=0.88, bottom=0.06)

    # 左上: 输入单位球面 (3D)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    plotSphere3D(ax1, spherePts, shape,
                 '$\\mathbb{S}^2$ 单位球面\n($\\mathbb{R}^3$ 输入)',
                 axes3D)

    # 右上: V^T 旋转后 (3D)
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    plotSphere3D(ax2, s1Pts, shape,
                 '$V^\\top$ 旋转\n(球面形状不变)',
                 s1Ax)

    # 左下: Σ 缩放降维后 (2D)
    ax3 = fig.add_subplot(gs[1, 0])
    plotEllipse2D(ax3, s2Samples,
                  '$\\Sigma$ 缩放 & 降维\n($\\mathbb{R}^3$ → $\\mathbb{R}^2$ 椭圆)',
                  s2Ax, sigma=s, latLines=s2Lat, merLines=s2Mer)

    # 右下: U 旋转后 (2D)
    ax4 = fig.add_subplot(gs[1, 1])
    plotEllipse2D(ax4, s3Samples,
                  '$U$ 旋转\n(最终 $\\mathbb{R}^2$ 输出)',
                  s3Ax, sigma=s, rotation=U, latLines=s3Lat, merLines=s3Mer)

    # ---- 总标题 ----
    sigmaStr = '    '.join(
        f'$\\sigma_{{{i+1}}}={s[i]:.3f}$' for i in range(len(s))
    )
    fig.suptitle(
        f'SVD 三步变换:  $A = U \\Sigma V^\\top$    '
        f'($n=3 \\to m=2$)        {sigmaStr}',
        fontsize=14, fontweight='bold', y=0.97)

    # ---- 底部图例 ----
    fig.text(0.5, 0.015,
             '红 $e_1$   绿 $e_2$   蓝 $e_3$    |    '
             '红虚线 = 椭圆边界  |  线 = 纬线/经线投影',
             ha='center', fontsize=8.5, color='#7F8C8D')

    # ---- 保存 ----
    scriptDir = os.path.dirname(os.path.abspath(__file__))
    outputDir = os.path.join(scriptDir, 'outputs', 'figures')
    os.makedirs(outputDir, exist_ok=True)
    outputPath = os.path.join(outputDir, 'SVDGeometry.png')
    plt.savefig(outputPath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] 图像已保存至: {outputPath}")
    plt.show()


if __name__ == '__main__':
    main()
