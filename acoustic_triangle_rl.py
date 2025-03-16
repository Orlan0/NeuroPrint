#!/usr/bin/env python3

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from fenics import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

###############################################
# 1) PLOT CALLBACK (SAVE FIGURES EVERY N STEPS)
###############################################
class PlotCallback(BaseCallback):
    """
    Saves a plot of the pressure field every `check_freq` training steps.
    """
    def __init__(self, env, check_freq=500, save_dir='plots', verbose=0):
        super().__init__(verbose)
        self.env = env
        self.check_freq = check_freq
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate policy on one env reset
            obs, _ = self.env.reset()
            action, _ = self.model.predict(obs)
            new_obs, reward, terminated, truncated, info = self.env.step(action)

            Nx, Ny = self.env.nx, self.env.ny
            final_pressure_2d = new_obs.reshape((Ny, Nx))

            plt.figure(figsize=(6,5))
            plt.imshow(
                final_pressure_2d,
                origin='lower',
                extent=[0, self.env.L, 0, self.env.H],
                aspect='equal'
            )
            plt.colorbar(label='Pressure (normalized scale)')
            plt.title(f'Pressure Slice (Step {self.n_calls})')
            plt.xlabel('x [m]')
            plt.ylabel('y [m]')
            plt.tight_layout()

            filename = os.path.join(self.save_dir, f'pressure_field_step_{self.n_calls}.png')
            plt.savefig(filename)
            plt.close()

            if self.verbose > 0:
                print(f'[PlotCallback] Saved: {filename}')

        return True

##################################################
# 2) TRAINING MONITOR CALLBACK (PRINT PROGRESS)
##################################################
class TrainingMonitorCallback(BaseCallback):
    """
    Prints training progress (epochs, step count, approximate losses)
    every `check_freq` steps.
    """
    def __init__(self, check_freq=500, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.epoch = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.epoch += 1

            policy_loss = self.model.logger.name_to_value.get("train/policy_loss", None)
            value_loss  = self.model.logger.name_to_value.get("train/value_loss", None)
            entropy_loss= self.model.logger.name_to_value.get("train/entropy_loss", None)

            print(f"\n--- EPOCH {self.epoch} (Global Step: {self.n_calls}) ---")
            if policy_loss is not None:
                print(f"Policy loss:   {policy_loss:.4f}")
            if value_loss is not None:
                print(f"Value loss:    {value_loss:.4f}")
            if entropy_loss is not None:
                print(f"Entropy loss:  {entropy_loss:.4f}")
            print("-----------------------------------\n")

        return True

#############################
# 3) CREATE TRIANGLE MASK
#############################
def create_triangle_mask(Nx, Ny, L, H):
    x_vals = np.linspace(0, L, Nx)
    y_vals = np.linspace(0, H, Ny)
    X, Y = np.meshgrid(x_vals, y_vals)

    v1 = np.array([L/2, 0.8*H])
    v2 = np.array([0.2*L, 0.2*H])
    v3 = np.array([0.8*L, 0.2*H])

    def is_inside_triangle(pt, v1, v2, v3):
        d1 = (pt[0] - v2[0])*(v1[1] - v2[1]) - (v1[0] - v2[0])*(pt[1] - v2[1])
        d2 = (pt[0] - v3[0])*(v2[1] - v3[1]) - (v2[0] - v3[0])*(pt[1] - v3[1])
        d3 = (pt[0] - v1[0])*(v3[1] - v1[1]) - (v3[0] - v1[0])*(pt[1] - v1[1])
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    mask = np.zeros((Ny, Nx), dtype=np.float64)
    for j in range(Ny):
        for i in range(Nx):
            pt = np.array([X[j, i], Y[j, i]])
            mask[j, i] = 1.0 if is_inside_triangle(pt, v1, v2, v3) else 0.0

    return mask

####################################
# 4) CUSTOM FENICS EXPRESSION CLASS
####################################
class StaticTransducerWave(UserExpression):
    """
    Summation of Gaussian wave sources at each transducer position.
    amplitude = thresholded from [0..1].
    """
    def __init__(self, A, px, py, pz, **kwargs):
        super().__init__(**kwargs)
        self.A  = A
        self.px = px
        self.py = py
        self.pz = pz

    def eval(self, value, x):
        val = 0.0
        for Ai, px_i, py_i, pz_i in zip(self.A, self.px, self.py, self.pz):
            dx = x[0] - px_i
            dy = x[1] - py_i
            dz = x[2] - pz_i
            r2 = dx*dx + dy*dy + dz*dz
            val += Ai * np.exp(-100.0 * r2)
        value[0] = val

#################################
# 5) GYM ENVIRONMENT DEFINITION
#################################
class AcousticEnv(gym.Env):
    """
    Use a continuous Box(0..1) action space with thresholding => 0 or 1e6 amplitude.
    Must return (obs, reward, terminated, truncated, info) for Gymnasium 0.26+.
    """
    def __init__(self):
        super().__init__()

        # Domain geometry
        self.L, self.H, self.W = 0.1, 0.1, 0.1
        self.nx, self.ny, self.nz = 30, 30, 30
        self.mesh = BoxMesh(Point(0,0,0), Point(self.L,self.H,self.W),
                            self.nx, self.ny, self.nz)
        self.V = FunctionSpace(self.mesh, "P", 1)
        self.bc = DirichletBC(self.V, Constant(0.0), "on_boundary")

        # Transducer positions
        self.transducer_positions = [
            (0.02, 0.02, 0.09), (0.02, 0.08, 0.09), (0.05, 0.05, 0.09),
            (0.08, 0.02, 0.09), (0.08, 0.08, 0.09), (0.02, 0.05, 0.09),
            (0.08, 0.05, 0.09), (0.01, 0.02, 0.05), (0.01, 0.08, 0.05),
            (0.01, 0.05, 0.02), (0.01, 0.05, 0.08), (0.09, 0.02, 0.05),
            (0.09, 0.08, 0.05), (0.09, 0.05, 0.02), (0.09, 0.05, 0.08)
        ]
        self.n_transducers = len(self.transducer_positions)

        # Continuous action space [0..1]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_transducers,), dtype=np.float32
        )

        # Observations: Nx×Ny slice at z=0.05
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.nx*self.ny,),
            dtype=np.float32
        )

        # Precompute the triangle mask
        self.triangle_mask = create_triangle_mask(self.nx, self.ny, self.L, self.H).flatten()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        init_action = np.ones(self.n_transducers, dtype=float)*0.5
        p_sol = self.solve_acoustic_field(init_action)
        obs = self.sample_pressure_slice(p_sol)
        return obs, {}

    def step(self, action):
        p_sol = self.solve_acoustic_field(action)
        obs = self.sample_pressure_slice(p_sol)
        reward = self.calculate_reward(obs)

        # Gymnasium: return 5 items
        terminated = False   # No terminal condition
        truncated  = False   # No time-limit condition
        info = {}

        return obs, reward, terminated, truncated, info

    def solve_acoustic_field(self, action):
        # threshold => 0 or 1
        binary_action = (action > 0.5).astype(float)
        amplitudes = binary_action*1e6

        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        wave_expr = StaticTransducerWave(
            A=amplitudes,
            px=[p[0] for p in self.transducer_positions],
            py=[p[1] for p in self.transducer_positions],
            pz=[p[2] for p in self.transducer_positions],
            degree=2
        )

        a = inner(grad(u), grad(v))*dx
        rhs = wave_expr*v*dx
        u_sol = Function(self.V)
        solve(a == rhs, u_sol, self.bc,
              solver_parameters={"linear_solver": "gmres",
                                 "preconditioner": "ilu"})
        return u_sol

    def sample_pressure_slice(self, u_sol, z_slice=0.05):
        x_vals = np.linspace(0, self.L, self.nx)
        y_vals = np.linspace(0, self.H, self.ny)
        P = np.zeros((self.ny, self.nx))
        for j, yy in enumerate(y_vals):
            for i, xx in enumerate(x_vals):
                P[j, i] = u_sol(xx, yy, z_slice)
        return P.flatten()

    def calculate_reward(self, pressure_slice):
        p_min, p_max = pressure_slice.min(), pressure_slice.max()
        if abs(p_max - p_min) < 1e-12:
            p_norm = np.zeros_like(pressure_slice)
        else:
            p_norm = (pressure_slice - p_min)/(p_max - p_min)
        mse = np.mean((p_norm - self.triangle_mask)**2)
        return -mse

#####################################
# 6) MAIN FUNCTION: TRAIN & DISPLAY
#####################################
def main():
    env = AcousticEnv()

    # Create PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # Callbacks
    plot_callback = PlotCallback(env, check_freq=500, save_dir='plots', verbose=1)
    train_monitor_callback = TrainingMonitorCallback(check_freq=500, verbose=1)
    callback = CallbackList([plot_callback, train_monitor_callback])

    # Train for 5000 steps
    model.learn(total_timesteps=5000, callback=callback)

    # Test final policy
    obs, _ = env.reset()
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Final action (float in [0..1]): {action}")
    print(f"Final reward: {reward:.4f}")

    Nx, Ny = env.nx, env.ny
    final_pressure_2d = obs.reshape((Ny, Nx))

    plt.figure(figsize=(6,5))
    plt.imshow(
        final_pressure_2d,
        origin='lower',
        extent=[0, env.L, 0, env.H],
        aspect='equal'
    )
    plt.colorbar(label='Pressure (normalized scale)')
    plt.title('Final Pressure Slice')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.tight_layout()
    plt.savefig('plots/final_pressure.png')
    plt.show()
    print("Final pressure plot saved to plots/final_pressure.png")


if __name__ == '__main__':
    main()
