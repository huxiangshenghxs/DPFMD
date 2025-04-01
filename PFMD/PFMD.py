import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from .utils import generate_dhtm, grad_Gamma2d

class DiscretePeierls:
    def __init__(self, u: np.ndarray, Gamma2d, gamma_c, params: dict):
        """
        初始化 DiscretePeierls 类
        u: np.ndarray -> (nl, 2) 初始位移
        Gamma2d: function -> Gamma 面的势能函数
        gamma_c: np.ndarray -> Gamma 表面的参数
        params: dict -> 各种物理参数 omega, boundary conditions, stress
        """
        assert u.shape[1] == 2, "u should be a 2D array with shape (nl, 2)"
        self.nl = u.shape[0]
        self.u = u
        self.dhtm = generate_dhtm(self.nl)
        self.Gamma2d = Gamma2d
        self.gamma_c = gamma_c
        self.params = params  
        self.unpack_params()
    
    def unpack_params(self):
        """解包参数字典"""
        self.omega_r_screw, self.omega_r_edge = self.params['omega_r']
        self.omega_s_screw, self.omega_s_edge = self.params['omega_s']
        self.uz_l, self.uz_r = self.params['uz_bound']
        self.uy_l, self.uy_r = self.params['uy_bound']
        self.sigma_screw, self.sigma_edge = self.params['sigma']

    def equation(self, t, u_flat):
        """
        计算 du/dt
        这里必须使用 `t, u` 作为参数，以便 `solve_ivp` 调用
        u_flat: 1D 数组，需要 reshape 成 (nl, 2)
        """
        u = u_flat.reshape(self.nl, 2)  # 将 1D 变回 (nl, 2)
        
        # 处理边界条件
        uz = np.concatenate(([self.uz_l], u[:, 0], [self.uz_r]))
        uy = np.concatenate(([self.uy_l], u[:, 1], [self.uy_r]))
        
        # 计算 Gamma 面力
        force = -grad_Gamma2d(np.column_stack((uz[1:self.nl+1], uy[1:self.nl+1])), self.Gamma2d, self.gamma_c)
        
        # 计算 duz/dt 和 duy/dt
        duz = (
            0.5 * self.omega_r_screw * (uz[:-2] + uz[2:] - 2 * uz[1:-1]) +
            0.5 * self.omega_s_screw * self.dhtm @ (uz[1:] - uz[:-1]) / np.pi +
            force[:, 0] + np.full(self.nl, self.sigma_screw)
        )
        
        duy = (
            0.5 * self.omega_r_edge * (uy[:-2] + uy[2:] - 2 * uy[1:-1]) +
            0.5 * self.omega_s_edge * self.dhtm @ (uy[1:] - uy[:-1]) / np.pi +
            force[:, 1] + np.full(self.nl, self.sigma_edge)
        )

        du = np.column_stack((duz, duy)).flatten()  # 变成 1D 数组
        return du
        
    def run(self, tspan, t_eval):
        """
        运行微分方程求解
        tspan: (t0, tf) -> 时间范围
        t_eval: np.ndarray -> 计算时间点
        """
        sol = solve_ivp(
            fun=self.equation,  # 传入方程
            t_span=tspan,
            y0=self.u.flatten(),  # 初始条件必须是 1D
            method='BDF',     
            rtol=1e-6,        
            t_eval=t_eval
        )
        
        # 结果 reshape 回 (nl, 2, time)
        return sol.t, sol.y.reshape(self.nl, 2, -1)
