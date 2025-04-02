import jax.numpy as jnp
from .utils import generate_dhtm, grad_Gamma2d
from jax.experimental.ode import odeint
# from scipy.integrate import solve_ivp
# import diffrax

class DiscretePeierls:
    def __init__(self, u: jnp.ndarray, Gamma2d, gamma_c, params: dict):
        """
        初始化 DiscretePeierls 类
        u: jnp.ndarray -> (nl, 2) 初始位移
        Gamma2d: function -> Gamma 面的势能函数
        gamma_c: jnp.ndarray -> Gamma 表面的参数
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

    def equation(self, u_flat, t):
        """
        计算 du/dt
        这里必须使用 `t, u` 作为参数，以便 `solve_ivp` 调用
        u_flat: 1D 数组，需要 reshape 成 (nl, 2)
        """
        u = u_flat.reshape(self.nl, 2, order='F')  # 将 1D 变回 (nl, 2)
        
        # 处理边界条件
        uz = jnp.concatenate((jnp.array([self.uz_l]), u[:, 0], jnp.array([self.uz_r])))
        uy = jnp.concatenate((jnp.array([self.uy_l]), u[:, 1], jnp.array([self.uy_r])))
        
        # 计算 Gamma 面力
        force = -grad_Gamma2d(jnp.column_stack((u[:, 0], u[:, 1])), self.Gamma2d, self.gamma_c)
        
        # 计算 duz/dt 和 duy/dt
        duz = (
            0.5 * self.omega_r_screw * (uz[:-2] + uz[2:] - 2 * uz[1:-1]) +
            0.5 * self.omega_s_screw * self.dhtm @ (uz[1:] - uz[:-1]) / jnp.pi +
            force[:, 0] + jnp.full(self.nl, self.sigma_screw)
        )
        
        duy = (
            0.5 * self.omega_r_edge * (uy[:-2] + uy[2:] - 2 * uy[1:-1]) +
            0.5 * self.omega_s_edge * self.dhtm @ (uy[1:] - uy[:-1]) / jnp.pi +
            force[:, 1] + jnp.full(self.nl, self.sigma_edge)
        )

        du = jnp.column_stack((duz, duy)).flatten(order='F') # 变成 1D 数组
        return du
    
    
    def run(self, t_eval):
        sol = odeint(self.equation, self.u.flatten(order='F'), t_eval)
        return t_eval, sol.reshape(sol.shape[0], self.nl, 2, order='F')
        
    # def run(self, tspan, t_eval):
    #     t0, tf = tspan
    #     y0 = self.u.flatten(order='F')

    #     term = diffrax.ODETerm(self.equation)

    #     solver = diffrax.Tsit5()

    #     dt0 = 1e-3 
    #     stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)

    #     saveat = diffrax.SaveAt(ts=t_eval)

    #     solution = diffrax.diffeqsolve(
    #         term,
    #         solver,
    #         t0=t0,
    #         t1=tf,
    #         dt0=dt0,
    #         y0=y0,
    #         saveat=saveat,
    #         stepsize_controller=stepsize_controller
    #     )

    #     u_result = solution.ys.reshape(len(t_eval), self.nl, 2, order='F')

    #     return solution.ts, u_result    
    # def run(self, tspan, t_eval):
    #     """
    #     运行微分方程求解
    #     tspan: (t0, tf) -> 时间范围
    #     t_eval: jnp.ndarray -> 计算时间点
    #     """
    #     sol = solve_ivp(
    #         fun=self.equation,  # 传入方程
    #         t_span=tspan,
    #         y0=self.u.flatten(),  # 初始条件必须是 1D
    #         method='BDF',     
    #         rtol=1e-6,        
    #         t_eval=t_eval
    #     )
        
    #     # 结果 reshape 回 (nl, 2, time)
    #     return sol.t, sol.y.reshape(self.nl, 2, -1)
