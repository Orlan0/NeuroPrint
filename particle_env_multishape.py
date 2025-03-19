import os
import time
import cv2
import numpy as np
import open3d as o3d
import gym
from gym import spaces
from stable_baselines3 import PPO

########################################
# 1. Additional 3D Shapes
########################################

def generate_pyramid(num_particles, base_size=5, height=5):
    """
    Square-based pyramid approximation in 'layers'.
    """
    positions = []
    layers = int(np.sqrt(num_particles))
    if layers < 1:
        layers = 1

    half_base = base_size / 2
    count = 0

    for layer_idx in range(layers):
        frac = layer_idx / (layers - 1) if layers > 1 else 0
        y = frac * height
        layer_size = base_size * (1 - frac)
        half_layer = layer_size / 2
        layer_points = int(num_particles / layers)
        if layer_idx == layers - 1:
            layer_points = num_particles - count

        for _ in range(layer_points):
            x = np.random.uniform(-half_layer, half_layer)
            z = np.random.uniform(-half_layer, half_layer)
            positions.append(np.array([x, y, z]))
        count += layer_points

    # Force apex if needed
    if len(positions) < num_particles:
        positions.append(np.array([0, height, 0]))
    positions = positions[:num_particles]
    return positions

def generate_cylinder(num_particles, radius=3, height=5):
    """
    Approximate a vertical cylinder of given radius, height.
    We'll do 'layers' along the Y-axis from 0 to height.
    """
    positions = []
    layers = int(np.sqrt(num_particles))
    if layers < 1:
        layers = 1

    points_per_layer = int(num_particles / layers)
    y_step = height / layers
    count = 0

    for layer_idx in range(layers):
        y = layer_idx * y_step
        # generate points around circle of 'radius'
        for _ in range(points_per_layer):
            theta = np.random.uniform(0, 2*np.pi)
            r = radius
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            positions.append(np.array([x, y, z]))
        count += points_per_layer

    # If any leftover
    while len(positions) < num_particles:
        theta = np.random.uniform(0, 2*np.pi)
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        positions.append(np.array([x, height, z]))

    return positions[:num_particles]

def generate_random_shape(num_particles, scale=5):
    """
    Sample points in a random 3D 'blob' (sphere volume).
    """
    positions = []
    for _ in range(num_particles):
        r = scale * (np.random.rand() ** (1/3))
        theta = np.random.rand() * 2*np.pi
        phi = np.random.rand() * np.pi
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        positions.append(np.array([x, y, z]))
    return positions

def generate_cad_shape(filename, num_particles, scale=1.0):
    """
    Load a CAD file (STL, OBJ, etc.) and sample 'num_particles' points on its surface.
    """
    mesh = o3d.io.read_triangle_mesh(filename)
    if mesh.is_empty():
        raise ValueError(f"Mesh file '{filename}' is empty or invalid.")

    pcd = mesh.sample_points_poisson_disk(num_particles)
    pts = np.asarray(pcd.points)
    pts *= scale
    center = np.mean(pts, axis=0)
    pts -= center  # center it at (0,0,0)
    return [np.array(p) for p in pts]

########################################
# 2. Existing Basic Shapes (Sphere, Cube, Letter_A)
########################################

def generate_sphere(num_particles, radius=5):
    positions = []
    phi = np.pi * (3 - np.sqrt(5))
    for i in range(num_particles):
        y = 1 - (i / float(num_particles - 1)) * 2
        r = np.sqrt(1 - y * y) * radius
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        positions.append(np.array([x, y * radius, z]))
    return positions

def generate_cube(num_particles, cube_size=5):
    positions = []
    grid_size = int(np.ceil(num_particles ** (1/3)))
    spacing = cube_size / (grid_size - 1) if grid_size > 1 else 0
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                if len(positions) < num_particles:
                    positions.append(np.array([
                        (x - grid_size/2) * spacing,
                        (y - grid_size/2) * spacing,
                        (z - grid_size/2) * spacing
                    ]))
    return positions

def generate_letter_A(num_particles, scale=5):
    positions = []
    n1 = int(num_particles * 0.4)
    n2 = int(num_particles * 0.4)
    n3 = num_particles - n1 - n2
    for i in range(n1):
        t = i / (n1 - 1) if n1 > 1 else 0
        x = -scale * (1 - t)
        y = -scale + 2 * scale * t
        z = 0
        positions.append(np.array([x, y, z]))
    for i in range(n2):
        t = i / (n2 - 1) if n2 > 1 else 0
        x = scale * (1 - t)
        y = -scale + 2 * scale * t
        z = 0
        positions.append(np.array([x, y, z]))
    for i in range(n3):
        t = i / (n3 - 1) if n3 > 1 else 0
        x = -scale + 2 * scale * t
        y = 0
        z = 0
        positions.append(np.array([x, y, z]))
    return positions

########################################
# 3. Particle & Simulation
########################################

class Particle:
    def __init__(self, position, target):
        self.position = np.array(position, dtype=np.float64)
        self.target = np.array(target, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)

    def update(self, dt, speed=1.0):
        direction = self.target - self.position
        dist = np.linalg.norm(direction)
        if dist > 0:
            direction /= dist
        self.velocity = direction * min(speed, dist / dt)
        self.position += self.velocity * dt

class Simulation:
    def __init__(self, particles):
        self.dt = 0.1
        self.particles = particles

    def update(self):
        for p in self.particles:
            p.update(self.dt)

    def get_positions(self):
        return np.array([p.position for p in self.particles])

########################################
# 4. Multi-Shape RL Environment (No CAD in Training)
########################################

class ParticleEnvMultiShape(gym.Env):
    """
    RL environment with these shapes: sphere, cube, letter_A, pyramid, cylinder, random
    but NOT 'cad'.
    Observations: (positions Nx3 + target Nx3) => (2N, 3)
    Reward = -mean_dist - collision_penalty
    """
    def __init__(
        self,
        num_particles=200,
        shapes_list=None,
        collision_threshold=0.5,
        collision_penalty_value=0.5,
        shape_scale=3.0
    ):
        super().__init__()
        self.num_particles = num_particles
        # Default shapes (no CAD)
        self.shapes_list = shapes_list or ["sphere", "cube", "letter_A", "pyramid", "cylinder", "random"]
        self.collision_threshold = collision_threshold
        self.collision_penalty_value = collision_penalty_value
        self.shape_scale = shape_scale

        self.sim = None
        self.target_shape = None

        self.observation_space = spaces.Box(
            low=-20, high=20,
            shape=(num_particles * 2, 3),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(num_particles, 3),
            dtype=np.float32
        )

    def _load_shape_coords(self, shape_name):
        if shape_name == "sphere":
            return generate_sphere(self.num_particles, radius=self.shape_scale)
        elif shape_name == "cube":
            return generate_cube(self.num_particles, cube_size=self.shape_scale)
        elif shape_name == "letter_A":
            return generate_letter_A(self.num_particles, scale=self.shape_scale)
        elif shape_name == "pyramid":
            return generate_pyramid(self.num_particles, base_size=self.shape_scale, height=self.shape_scale)
        elif shape_name == "cylinder":
            return generate_cylinder(self.num_particles, radius=self.shape_scale, height=self.shape_scale*2)
        elif shape_name == "random":
            return generate_random_shape(self.num_particles, scale=self.shape_scale)
        else:
            raise ValueError(f"Unknown shape: {shape_name}")

    def reset(self):
        shape_name = np.random.choice(self.shapes_list)
        coords = self._load_shape_coords(shape_name)

        particles = []
        for c in coords:
            pos = np.random.uniform(-10, 10, 3)
            particles.append(Particle(pos, c))

        self.sim = Simulation(particles)
        self.target_shape = np.array(coords)
        return self._get_state()

    def _get_state(self):
        current_positions = self.sim.get_positions()
        return np.vstack((current_positions, self.target_shape))

    def step(self, action):
        for i, p in enumerate(self.sim.particles):
            p.velocity = action[i]
            p.update(self.sim.dt)

        # Mean distance
        mean_distance = np.mean([
            np.linalg.norm(p.target - p.position)
            for p in self.sim.particles
        ])

        # Collision penalty
        collision_penalty = 0.0
        for i in range(self.num_particles):
            for j in range(i+1, self.num_particles):
                dist = np.linalg.norm(self.sim.particles[i].position - self.sim.particles[j].position)
                if dist < self.collision_threshold:
                    collision_penalty += (self.collision_threshold - dist) * self.collision_penalty_value

        reward = -mean_distance - collision_penalty

        done = all(np.linalg.norm(p.target - p.position) < 0.1
                   for p in self.sim.particles)

        return self._get_state(), reward, done, {}

########################################
# 5. Training (500 episodes)
########################################

def train_multi_shape_model(
    num_particles=200,
    total_episodes=500,
    timesteps_per_episode=100,
    collision_threshold=0.5,
    collision_penalty_value=0.5,
    shape_scale=3.0,
    model_file="universal_multi_shape_model.zip"
):
    if os.path.exists(model_file):
        print(f"[Train] Model '{model_file}' already exists. Skipping training.")
        return

    print("[Train] Starting multi-shape RL training (no CAD).")

    env = ParticleEnvMultiShape(
        num_particles=num_particles,
        shapes_list=["sphere","cube","letter_A","pyramid","cylinder","random"],
        collision_threshold=collision_threshold,
        collision_penalty_value=collision_penalty_value,
        shape_scale=shape_scale
    )
    model = PPO("MlpPolicy", env, verbose=1)

    episodes_done = 0
    while episodes_done < total_episodes:
        print(f"[Train] Episode {episodes_done+1}/{total_episodes}")
        model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)
        episodes_done += 1

    model.save(model_file)
    print(f"[Train] Model saved as '{model_file}'.")

########################################
# 6. Test (300 steps) on the same shapes
########################################

def create_video_from_frames(frame_folder="frames_test", output_video="test_run.mp4", fps=30):
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])
    if not frame_files:
        print("[Video] No frames found!")
        return
    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    for frame_file in frame_files:
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)
    video.release()
    print(f"[Video] Video saved as '{output_video}'.")

def test_multi_shape_model(model_file="universal_multi_shape_model.zip", steps=300):
    """
    Test on the same environment shapes for 300 steps, capturing frames.
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model '{model_file}' not found!")

    model = PPO.load(model_file)
    env = ParticleEnvMultiShape(
        num_particles=200,
        shapes_list=["sphere","cube","letter_A","pyramid","cylinder","random"],
        collision_threshold=0.5,
        collision_penalty_value=0.5,
        shape_scale=3.0
    )
    obs = env.reset()

    save_path = "frames_test"
    os.makedirs(save_path, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    point_cloud = o3d.geometry.PointCloud()

    num_particles = env.num_particles
    current_positions = obs[:num_particles]
    point_cloud.points = o3d.utility.Vector3dVector(current_positions)
    vis.add_geometry(point_cloud)

    vis.update_geometry(point_cloud)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    for i in range(steps):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        current_positions = obs[:num_particles]
        point_cloud.points = o3d.utility.Vector3dVector(current_positions)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # float buffer => avoid black frames
        float_buffer = vis.capture_screen_float_buffer(True)
        img = (np.asarray(float_buffer) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        frame_path = os.path.join(save_path, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, img)

        if done:
            print(f"[Test] Done at step {i}, reward={reward:.3f}")
            break

    vis.destroy_window()
    create_video_from_frames(save_path, "test_run.mp4", fps=30)

########################################
# 7. Test CAD Shape (not in training)
########################################

class ParticleEnvCADTest(gym.Env):
    """
    Environment that loads ONLY a CAD shape (not used in training).
    """
    def __init__(
        self,
        num_particles=200,
        cad_path="my_cad_model.stl",
        collision_threshold=0.5,
        collision_penalty_value=0.5,
        shape_scale=3.0
    ):
        super().__init__()
        self.num_particles = num_particles
        self.collision_threshold = collision_threshold
        self.collision_penalty_value = collision_penalty_value
        self.shape_scale = shape_scale

        self.sim = None
        self.target_shape = None

        self.observation_space = spaces.Box(
            low=-20, high=20,
            shape=(num_particles * 2, 3),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(num_particles, 3),
            dtype=np.float32
        )

        # Load CAD shape coords once
        coords = generate_cad_shape(cad_path, num_particles, scale=shape_scale)
        self.cad_coords = np.array(coords)

    def reset(self):
        # random initial positions
        particles = []
        for c in self.cad_coords:
            pos = np.random.uniform(-10, 10, 3)
            particles.append(Particle(pos, c))

        self.sim = Simulation(particles)
        self.target_shape = self.cad_coords
        return self._get_state()

    def _get_state(self):
        current_positions = self.sim.get_positions()
        return np.vstack((current_positions, self.target_shape))

    def step(self, action):
        for i, p in enumerate(self.sim.particles):
            p.velocity = action[i]
            p.update(self.sim.dt)

        mean_distance = np.mean([
            np.linalg.norm(p.target - p.position)
            for p in self.sim.particles
        ])

        collision_penalty = 0.0
        for i in range(self.num_particles):
            for j in range(i+1, self.num_particles):
                dist = np.linalg.norm(self.sim.particles[i].position - self.sim.particles[j].position)
                if dist < self.collision_threshold:
                    collision_penalty += (self.collision_threshold - dist)*self.collision_penalty_value

        reward = -mean_distance - collision_penalty
        done = all(np.linalg.norm(p.target - p.position) < 0.1
                   for p in self.sim.particles)

        return self._get_state(), reward, done, {}

def test_cad_shape_model(
    model_file="universal_multi_shape_model.zip",
    cad_path="my_cad_model.stl",
    steps=300
):
    """
    Use the final model on a new environment that loads a CAD shape.
    The agent attempts to form that shape from random initial positions.
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model '{model_file}' not found!")

    model = PPO.load(model_file)
    env = ParticleEnvCADTest(
        num_particles=200,
        cad_path=cad_path,
        collision_threshold=0.5,
        collision_penalty_value=0.5,
        shape_scale=3.0
    )
    obs = env.reset()

    save_path = "frames_cad_test"
    os.makedirs(save_path, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    point_cloud = o3d.geometry.PointCloud()

    num_particles = env.num_particles
    current_positions = obs[:num_particles]
    point_cloud.points = o3d.utility.Vector3dVector(current_positions)
    vis.add_geometry(point_cloud)

    vis.update_geometry(point_cloud)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    for i in range(steps):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

        current_positions = obs[:num_particles]
        point_cloud.points = o3d.utility.Vector3dVector(current_positions)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # float buffer capture
        float_buffer = vis.capture_screen_float_buffer(True)
        img = (np.asarray(float_buffer) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        frame_path = os.path.join(save_path, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, img)

        if done:
            print(f"[CAD Test] Done at step {i}, reward={reward:.3f}")
            break

    vis.destroy_window()
    create_video_from_frames(save_path, "cad_test_run.mp4", fps=30)

########################################
# 8. Main
########################################

if __name__ == "__main__":
    # 1) Train the multi-shape model (no CAD in shapes list) for 500 episodes
    train_multi_shape_model(
        num_particles=200,
        total_episodes=500,          # increased to 500
        timesteps_per_episode=100,
        collision_threshold=0.5,
        collision_penalty_value=0.5,
        shape_scale=3.0,
        model_file="universal_multi_shape_model.zip"
    )

    # 2) Test on the same environment shapes for 300 steps
    test_multi_shape_model(
        model_file="universal_multi_shape_model.zip",
        steps=300   # increased to 300
    )

    # 3) Test on a CAD shape (not trained on) for 300 steps
    # Provide your actual CAD file path below
    # test_cad_shape_model(
    #     model_file="universal_multi_shape_model.zip",
    #     cad_path="my_cad_model.stl",
    #     steps=300
    # )
