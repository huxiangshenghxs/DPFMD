# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:19:18 2025

@author: User
"""
import numpy as np
from scipy.integrate import solve_ivp
from autograd import grad
import math
import os
import matplotlib.pyplot as plt



def generate_dhtm(nl):
    # 生成行索引 l（从1到nl）
    l = np.arange(1, nl + 1).reshape(-1, 1)  # 列向量 (nl, 1)
    # 生成列索引 l1（从0到nl）
    l1 = np.arange(0, nl + 1)                # 行向量 (1, nl+1)
    # 计算分母矩阵（利用广播）
    denominator = l1 - l + 0.5
    # 避免除以零（此处无需处理，因为分母不可能为零）
    dhtm = 1.0 / denominator
    return dhtm


def discrete_peierls(t, u, p):
    # 参数解包
    ωr, ωs, uzbound, uybound, σ, dhtm, gamma = p
    
    # 检查参数维度
    n_l = dhtm.shape[0]
    if not (ωr.shape[0] == 2 and ωs.shape[0] == 2 and uzbound.shape[0] == 2 
            and uybound.shape[0] == 2 and σ.shape[0] == 2):
        print("Wrong parameter size!")
        return np.zeros_like(u)
    
    # 参数分解
    ωr_screw, ωr_edge = ωr
    ωs_screw, ωs_edge = ωs
    uzl, uzr = uzbound
    uyl, uyr = uybound
    σ_screw, σ_edge = σ
    
    # 位移场分割与边界扩展
    N = len(u) // 2
    uz = np.concatenate([[uzl], u[:N], [uzr]])
    uy = np.concatenate([[uyl], u[N:], [uyr]])
    
    # 计算恢复力 F(u)（假设 F 是某种函数，此处需具体实现）
    # 示例：假设 F 返回两列，需替换为实际逻辑
    Fu = np.zeros((N, 2))  # 占位符，需实现具体计算
    Fuz = Fu[:, 0]
    Fuy = Fu[:, 1]
    
    # 计算导数项 duz 和 duy
    duz = (
        0.5 * ωr_screw * (uz[0:nl] + uz[2:nl+2] - 2 * uz[1:nl+1]) +  # 二阶差分
        0.5 * ωs_screw * dhtm @(uz[1:nl+2] - uz[0:nl+1] ) / np.pi +  # Hilbert项（示例）
        Fuz + np.full(N, σ_screw)  # 外加应力
    )
    
    duy = (
        0.5 * ωr_edge * (uy[:-2] + uy[2:] - 2 * uy[1:-1]) +
        0.5 * ωs_edge * dhtm @ (uy[1:] - uy[:-1]) / np.pi +
        Fuy +   np.full(N, σ_edge)
    )
    
    # 合并并除以 gamma
    du = np.concatenate([duz, duy]) / gamma
    return du




def LRHS(u, p):
    # 参数解包
    ωr, ωs, uzbound, uybound, σ, dhtm, gamma = p
    N_l = dhtm.shape[0]  # 假设 dhtm 是 (N_l, N_l+1) 的矩阵
    
    # 参数校验
    if not (ωr.shape[0] == 2 and ωs.shape[0] == 2 and uzbound.shape[0] == 2 
            and uybound.shape[0] == 2 and σ.shape[0] == 2):
        print("Wrong parameter size!")
        return None, None
    
    # 参数分解
    ωr_screw, ωr_edge = ωr
    ωs_screw, ωs_edge = ωs
    uzl, uzr = uzbound
    uyl, uyr = uybound
    σ_screw, σ_edge = σ
    
    # 位移场扩展（添加边界）
    uz = np.concatenate([[uzl], u[:N_l], [uzr]])
    uy = np.concatenate([[uyl], u[N_l:2*N_l], [uyr]])
    
    # 计算恢复力 Fu（假设 F 接受 [uz, uy] 返回二维数组）
    Fu = np.array([F([uz[l+1], uy[l+1]]) for l in range(N_l)])
    Fuz = Fu[:, 0]
    Fuy = Fu[:, 1]
    
    # 计算 LHS（左边项）
    # z方向
    LHS_z = (
        0.5 * ωr_screw * (uz[:-2] + uz[2:] - 2 * uz[1:-1]) +  # 二阶差分项
        0.5 * ωs_screw * dhtm[:, 0] * (uz[1:] - uz[:-1]) / np.pi  # Hilbert项（示例）
    )
    # y方向
    LHS_y = (
        0.5 * ωr_edge * (uy[:-2] + uy[2:] - 2 * uy[1:-1]) +
        0.5 * ωs_edge * dhtm[:, 0] * (uy[1:] - uy[:-1]) / np.pi
    )
    
    # 计算 RHS（右边项）
    half = N_l // 2
    RHS_z = -Fuz - np.concatenate([-σ_screw * np.ones(half), σ_screw * np.ones(half)])
    RHS_y = -Fuy - np.concatenate([-σ_edge * np.ones(half), σ_edge * np.ones(half)])
    
    # 合并结果
    return np.concatenate([LHS_z, LHS_y]), np.concatenate([RHS_z, RHS_y])



# ====================== 晶格常数定义 ======================
b = 1.0                # 单位: b (基本长度单位)
a = math.sqrt(2) * b / 2  # 根据b计算a，单位: b

# ====================== 波矢参数 ======================
p = 2 * math.pi / math.sqrt(3)  # 与晶格对称性相关的波矢分量
q = 2 * math.pi                # 另一方向波矢分量

# ====================== 材料参数 ======================
nu = 0.297039            # 泊松比 (无量纲)

# 弹性系数 (单位: μ)
κs= 1.0           # 螺旋位错相关刚度
κe= 124.446 / 72.732  # 刃型位错相关刚度

# 非线性系数 (单位与μ和λ相关)
βs = 0.5 * math.sqrt(3) / 4  # 螺旋位错非线性项系数 (单位: 0.5*μ*λ)
βe = math.sqrt(3) / 4        # 刃型位错非线性项系数 (单位: μ*λ)

# ====================== 参数定义 ======================
# γc 数组 (单位: μ*b)
gamma_c = [
    -0.0019125519595932596,
    0.0003005935369634915,
    -0.004239113307857148,
    0.000025312895300520926,
    -0.000016013823539506667,
    -0.0007614840227797769,
    0.00020200193876531548,
    -0.000027877026670403876,
    4.032454755765513e-6,
    0.0035967678463422782,
    -0.00695607350257013,
    -0.0004427695369179309,
    2.4014461498135325e-7,
    -0.00014243187185675744,
    -0.000057645719547335114,
    0.01919935925806344
]  # 共 16 个元素，索引为 0~15

λ= math.sqrt(3) / 4

# 阻尼系数 gamma (单位: t~b/ct~1)
gamma =λ / 2

# 离散点数
nl = 100

# 生成离散坐标数组 [-50, -49, ..., 48, 49] (共 100 个元素)
lspan = np.arange(-nl//2, nl//2)  # 使用整数除法确保结果为整数


# ====================== 核心代码转换 ======================
# 1. 计算 ωr_se 和 ωs_se
ωr_se = np.array([βs, βe]) / λ**2  # 弹性系数项，形状 (2,)
ωs_se = np.array([κs, κe]) / λ     # 非线性系数项，形状 (2,)

# 2. 边界条件
uzbound = np.array([0, 1])  # z方向边界值 [uzl, uzr]
uybound = np.array([0, 0])  # y方向边界值 [uyl, uyr]

# 3. 初始位移场 u0
# 计算 atan 项（向量化操作）
part1 = (np.arctan(lspan + 3.5)/(2*np.pi))+(np.arctan(lspan - 3.5)/(2*np.pi))+0.5

# 拼接零数组
u0 = np.concatenate([part1, np.zeros(nl)])  # 形状 (2*NL,)

# 4. 时间范围
tspan = (0.0, 600.0)

# 5. 应力模式标志
single_stress = False  # False表示多应力，True表示单应力

# ====================== 验证输出 ======================
if __name__ == "__main__":
    print("ωr_se:", ωr_se)      # 应输出 [βs/λ², βe/λ²]
    print("ωs_se:", ωs_se)      # 应输出 [κs/λ, κe/λ]
    print("uzbound:", uzbound)  # 应输出 [0, 1]
    print("u0 shape:", u0.shape) # 应输出 (200,)（假设 NL=100）
    print("tspan:", tspan)       # 应输出 (0.0, 6000.0)

def Gamma(u):
    """标量势能函数，对应Julia中的Γ"""
    # 注意：u[0]对应Julia的u[1]，u[1]对应Julia的u[2]
    term = gamma_c[15]  # γc[16]作为常数项
    
    # γc[1]项
    term += gamma_c[0] * (
        np.cos(2.0 * p * u[1]) + 
        np.cos(1.0 * p * u[1] + 1.0 * q * u[0]) + 
        np.cos(1.0 * p * u[1] - 1.0 * q * u[0])
    )
    
    # γc[2]项
    term += gamma_c[1] * (
        np.cos(2.0 * q * u[0]) + 
        np.cos(3.0 * p * u[1] + 1.0 * q * u[0]) + 
        np.cos(3.0 * p * u[1] - 1.0 * q * u[0])
    )
    
    # γc[3]项
    term += gamma_c[2] * (
        np.cos(4.0 * p * u[1]) + 
        np.cos(2.0 * p * u[1] + 2.0 * q * u[0]) + 
        np.cos(2.0 * p * u[1] - 2.0 * q * u[0])
    )
    
    # γc[4]项（复杂项，需仔细核对）
    term += gamma_c[3] * (
        np.cos(5.0 * p * u[1] + 1.0 * q * u[0]) + 
        np.cos(4.0 * p * u[1] + 2.0 * q * u[0]) + 
        np.cos(1.0 * p * u[1] + 3.0 * q * u[0]) + 
        np.cos(1.0 * p * u[1] - 3.0 * q * u[0]) + 
        np.cos(4.0 * p * u[1] - 2.0 * q * u[0]) + 
        np.cos(5.0 * p * u[1] - 1.0 * q * u[0])
    )
    
    # γc[5]项
    term += gamma_c[4] * (
        np.cos(6.0 * p * u[1]) + 
        np.cos(3.0 * p * u[1] + 3.0 * q * u[0]) + 
        np.cos(3.0 * p * u[1] - 3.0 * q * u[0])
    )
    
    # γc[6]项
    term += gamma_c[5] * (
        np.cos(4.0 * q * u[0]) + 
        np.cos(6.0 * p * u[1] + 2.0 * q * u[0]) + 
        np.cos(6.0 * p * u[1] - 2.0 * q * u[0])
    )
    
    # γc[7]项
    term += gamma_c[6] * (
        np.cos(8.0 * p * u[1]) + 
        np.cos(4.0 * p * u[1] + 4.0 * q * u[0]) + 
        np.cos(4.0 * p * u[1] - 4.0 * q * u[0])
    )
    
    # γc[8]项
    term += gamma_c[7] * (
        np.cos(6.0 * q * u[0]) + 
        np.cos(9.0 * p * u[1] + 3.0 * q * u[0]) + 
        np.cos(9.0 * p * u[1] - 3.0 * q * u[0])
    )
    
    # γc[9]项
    term += gamma_c[8] * (
        np.cos(10.0 * p * u[1]) + 
        np.cos(5.0 * p * u[1] + 5.0 * q * u[0]) + 
        np.cos(5.0 * p * u[1] - 5.0 * q * u[0])
    )
    
    # γc[10]项（sin项）
    term += gamma_c[9] * (
        np.sin(2.0 * p * u[1]) - 
        np.sin(1.0 * p * u[1] + 1.0 * q * u[0]) - 
        np.sin(1.0 * p * u[1] - 1.0 * q * u[0])
    )
    
    # γc[11]项
    term += gamma_c[10] * (
        np.sin(4.0 * p * u[1]) - 
        np.sin(2.0 * p * u[1] + 2.0 * q * u[0]) - 
        np.sin(2.0 * p * u[1] - 2.0 * q * u[0])
    )
    
    # γc[12]项（复杂sin项）
    term += gamma_c[11] * (
        np.sin(5.0 * p * u[1] + 1.0 * q * u[0]) - 
        np.sin(4.0 * p * u[1] + 2.0 * q * u[0]) - 
        np.sin(1.0 * p * u[1] + 3.0 * q * u[0]) - 
        np.sin(1.0 * p * u[1] - 3.0 * q * u[0]) - 
        np.sin(4.0 * p * u[1] - 2.0 * q * u[0]) + 
        np.sin(5.0 * p * u[1] - 1.0 * q * u[0])
    )
    
    # γc[13]项
    term += gamma_c[12] * (
        np.sin(6.0 * p * u[1]) - 
        np.sin(3.0 * p * u[1] + 3.0 * q * u[0]) - 
        np.sin(3.0 * p * u[1] - 3.0 * q * u[0])
    )
    
    # γc[14]项
    term += gamma_c[13] * (
        np.sin(8.0 * p * u[1]) - 
        np.sin(4.0 * p * u[1] + 4.0 * q * u[0]) - 
        np.sin(4.0 * p * u[1] - 4.0 * q * u[0])
    )
    
    # γc[15]项
    term += gamma_c[14] * (
        np.sin(10.0 * p * u[1]) - 
        np.sin(5.0 * p * u[1] + 5.0 * q * u[0]) - 
        np.sin(5.0 * p * u[1] - 5.0 * q * u[0])
    )
    
    return term

# 计算负梯度
F = lambda u: -grad(Gamma)(u)

σ_se = np.array([0.0, 0.0])          # 外部应力 [σ_screw, σ_edge]


# 打包参数（注意顺序与Julia一致）
par = (
    ωr_se,      # 弹性系数数组 [βs/λ², βe/λ²]
    ωs_se,      # 非线性系数数组 [κs/λ, κe/λ]
    uzbound,    # z边界 [0, 1]
    uybound,    # y边界 [0, 0]
    σ_se,       # 外部应力 [0.0, 0.0]
    generate_dhtm(nl),       # 离散参数矩阵
    [gamma]     # 阻尼系数（包装成列表以匹配Julia结构）
)
# 求解时间点设置
t_eval = np.arange(tspan[0], tspan[1]+1, 10)  # 对应 saveat=10
# 调用求解器
sol = solve_ivp(
    fun=discrete_peierls,
    t_span=tspan,
    y0=u0,
    args=(par,),          # 传递额外参数
    method='BDF',      # 适合刚性问题
    rtol=1e-6,         # 对应 reltol=1e-6
    t_eval=t_eval      # 保存指定时间点
)

# ====================== 结果提取 ======================
# 解对象 sol 包含以下属性：
# - sol.t : 时间点数组
# - sol.y : 状态变量数组（每列对应一个时间点）
# - sol.success : 是否成功求解

lspan = lspan.reshape(-1, 1)  
# 螺旋分量 uz
uz_data = np.hstack([lspan, sol.y[:nl, :]])  # 形状 (100, 601 + 1)
np.savetxt(
    "uzL12_structure_0K.dat",
    uz_data.T,
    fmt="%.6f",
    header="l uz(t=0) uz(t=10) ..."
)

# 刃型分量 uy
uy_data = np.hstack([lspan, sol.y[nl:2*nl, :]])
np.savetxt(
    "uyL12_structure_0K.dat",
    uy_data.T,
    fmt="%.6f",
    header="l uy(t=0) uy(t=10) ..."
)

lspan = uz_data[:, 0]       # 第 0 列是 l，形状 (100,)
uz_values = uz_data[:, 1:]  # 第 1 列开始是 uz 的时间步数据，形状 (100, nt)
uy_values = uy_data[:, 1:]  # 第 1 列开始是 uy 的时间步数据，形状 (100, nt)
nt = uz_values.shape[1]
tspan = np.linspace(0, 600, nt)  # 形状 (nt,)


# 找到中心点的索引（例如 l=0）
center_idx = np.argmin(np.abs(lspan))  # 最接近 0 的索引

# 提取中心点的位移数据
uz_center = uz_values[center_idx, :]  # 形状 (nt,)
uy_center = uy_values[center_idx, :]  # 形状 (nt,)

# 绘制时间演化曲线
plt.figure(figsize=(10, 6))
plt.plot(tspan, uz_center, label=r"$u_z$ (Center)", color="blue", linewidth=2)
plt.plot(tspan, uy_center, label=r"$u_y$ (Center)", color="red", linewidth=2, linestyle="--")
plt.xlabel("Time (t)", fontsize=12)
plt.ylabel("Displacement", fontsize=12)
plt.title("Displacement Evolution at Center Point", fontsize=14)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# 选择时间步索引
time_indices = [0, nt//2, -1]  # 初始、中间、最终时间步
time_labels = [f"t = {tspan[i]:.0f}" for i in time_indices]

# 绘制 uz 的空间分布
plt.figure(figsize=(10, 6))
for i, idx in enumerate(time_indices):
    plt.plot(lspan, uz_values[:, idx], 
             label=time_labels[i], 
             linewidth=2,
             linestyle=["-", "--", "-."][i])
plt.xlabel("Spatial Coordinate (l)", fontsize=12)
plt.ylabel(r"$u_z$", fontsize=12)
plt.title("Spatial Distribution of $u_z$ at Different Times", fontsize=14)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# 绘制 uy 的空间分布
plt.figure(figsize=(10, 6))
for i, idx in enumerate(time_indices):
    plt.plot(lspan, uy_values[:, idx], 
             label=time_labels[i], 
             linewidth=2,
             linestyle=["-", "--", "-."][i])
plt.xlabel("Spatial Coordinate (l)", fontsize=12)
plt.ylabel(r"$u_y$", fontsize=12)
plt.title("Spatial Distribution of $u_y$ at Different Times", fontsize=14)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

from matplotlib.colors import Normalize


