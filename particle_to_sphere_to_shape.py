import os
import time
import cv2
import gym
import numpy as np
from gym import spaces
import open3d as o3d
from stable_baselines3 import PPO

###################################
# Shape Generators
###################################
def generate_sphere(num_particles, radius=5):
    positions = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
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
    n1 = int(num_particles * 0.4)  # left diagonal
    n2 = int(num_particles * 0.4)  # right diagonal
    n3 = num_particles - n1 - n2   # crossbar

    # Left diagonal
    for i in range(n1):
        t = i / (n1 - 1) if n1 > 1 else 0
        x = -scale * (1 - t)
        y = -scale + 2 * scale * t
        z = 0
        positions.append(np.array([x, y, z]))

    # Right diagonal
    for i in range(n2):
        t = i / (n2 - 1) if n2 > 1 else 0
        x = scale * (1 - t)
        y = -scale + 2 * scale * t
        z = 0
        positions.append(np.array([x, y, z]))

    # Horizontal crossbar
    for i in range(n3):
        t = i / (n3 - 1) if n3 > 1 else 0
        x = -scale + 2 * scale * t
        y = 0
        z = 0
        positions.append(np.array([x, y, z]))

    return positions

def load_custom_shape(shape_name, num_particles):
    if shape_name == "sphere":
        return generate_sphere(num_particles)
    elif shape_name == "cube":
        return generate_cube(num_particles)
    elif shape_name == "letter_A":
        return generate_letter_A(num_particles)
    else:
        raise ValueError(f"Unknown shape: {shape_name}")

###################################
# Particle & Simulation
###################################
class Particle:
    def __init__(self, position, target):
        self.position = np.array(position, dtype=np.float64)
        self.target = np.array(target, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)

    def update(self, dt, speed=1.0):
        direction = self.target - self.position
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction /= distance
        self.velocity = direction * min(speed, distance / dt)
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

###################################
# Visualization: create_video_from_frames
###################################
def create_video_from_frames(frame_folder="frames", output_video="particle_simulation.mp4", fps=30):
    """
    Compiles PNG frames into an MP4 video.
    """
    print(f"[Video] Creating video from frames in '{frame_folder}' ...")
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

###################################
# Collision-Penalizing RL Environment
###################################
from stable_baselines3 import PPO

class ParticleEnvCollision(gym.Env):
    """
    Single RL agent controlling all particles (N x 3 action).
    Reward = - (mean distance) - (collision penalty).
    """
    def __init__(self, num_particles=200, collision_threshold=1.0, collision_penalty=0.5):
        super(ParticleEnvCollision, self).__init__()
        self.num_particles = num_particles
        self.collision_threshold = collision_threshold
        self.collision_penalty_value = collision_penalty

        self.sim = None
        self.target_shape = None

        # Observations: positions (N x 3) + target (N x 3)
        self.observation_space = spaces.Box(low=-20, high=20,
                                            shape=(num_particles * 2, 3),
                                            dtype=np.float32)
        # Actions: velocity for each particle
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(num_particles, 3),
                                       dtype=np.float32)

    def load_shape(self, shape_name):
        coords = load_custom_shape(shape_name, self.num_particles)
        particles = []
        for c in coords:
            pos = np.random.uniform(-10, 10, 3)
            p = Particle(pos, c)
            particles.append(p)
        self.sim = Simulation(particles)
        self.target_shape = coords

    def step(self, action):
        # 1) Apply velocities
        for i, p in enumerate(self.sim.particles):
            p.velocity = action[i]
            p.update(self.sim.dt)

        # 2) Mean distance to target
        mean_distance = np.mean([
            np.linalg.norm(p.target - p.position)
            for p in self.sim.particles
        ])

        # 3) Collision penalty
        collision_penalty = 0.0
        for i in range(self.num_particles):
            for j in range(i+1, self.num_particles):
                dist = np.linalg.norm(self.sim.particles[i].position
                                      - self.sim.particles[j].position)
                if dist < self.collision_threshold:
                    collision_penalty += (self.collision_threshold - dist) * self.collision_penalty_value

        # 4) Final reward
        reward = -mean_distance - collision_penalty

        # 5) Done if all near target
        done = all(np.linalg.norm(p.target - p.position) < 0.1
                   for p in self.sim.particles)

        # 6) Build next state
        current_positions = self.sim.get_positions()
        state = np.vstack((current_positions, self.target_shape))
        return state, reward, done, {}

    def reset(self, shape_name="sphere"):
        self.load_shape(shape_name)
        state = np.vstack((self.sim.get_positions(), self.target_shape))
        return state

def train_rl_model_collision(model_file="particle_model_collision.zip"):
    """
    Trains a collision-penalizing RL model on multiple shapes.
    Saves as 'particle_model_collision.zip'.
    """
    if os.path.exists(model_file):
        print(f"[Training] Model file '{model_file}' already exists. Skipping training.")
        return

    print("[Training] Starting RL training with collision penalty...")
    env = ParticleEnvCollision(num_particles=200, collision_threshold=1.0, collision_penalty=0.5)
    shapes = ["sphere", "cube", "letter_A"]
    model = PPO("MlpPolicy", env, verbose=1)

    total_timesteps = 30000
    timesteps_per_episode = 100
    episodes = total_timesteps // timesteps_per_episode

    for i in range(episodes):
        shape = np.random.choice(shapes)
        env.reset(shape)
        print(f"[Training] Episode {i+1}/{episodes} - shape: {shape}")
        model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)

    model.save("particle_model_collision")
    print(f"[Training] Model saved as '{model_file}'.")

###################################
# Start as Sphere -> Another Shape
###################################
class SimulationStartAsSphere:
    """
    Particles start exactly on sphere coords => perfect sphere arrangement.
    """
    def __init__(self, sphere_coords):
        self.dt = 0.1
        self.particles = []
        for coord in sphere_coords:
            self.particles.append(Particle(position=coord, target=coord))

    def update(self):
        for p in self.particles:
            p.update(self.dt)

    def get_positions(self):
        return np.array([p.position for p in self.particles])

###################################
# Float-Buffer Single-Phase Run
###################################
def run_collision_sphere_to_shape(
    shape_name="letter_A",
    steps=200,
    model_file="particle_model_collision.zip"
):
    """
    1) Loads/Trains a collision RL model from 'particle_model_collision.zip'.
    2) Particles start in a perfect sphere arrangement (200).
    3) Immediately assign shape_name coords as targets.
    4) Use RL to move from sphere -> shape, capturing frames via float buffer.
    5) Produce an MP4: 'sphere_to_<shape_name>_collision.mp4'
    """
    # 1) Check if model file exists
    if not os.path.exists(model_file):
        print(f"[Run] Model file '{model_file}' not found. Training now...")
        train_rl_model_collision(model_file=model_file)

    print(f"[Run] Loading RL model from '{model_file}'...")
    model = PPO.load(model_file)

    # 2) Create sphere arrangement
    num_particles = 200
    sphere_coords = load_custom_shape("sphere", num_particles)
    sim_sphere = SimulationStartAsSphere(sphere_coords)

    # 3) Assign new shape's coords as targets
    new_coords = load_custom_shape(shape_name, num_particles)
    for i, p in enumerate(sim_sphere.particles):
        p.target = new_coords[i]

    # 4) Set up Open3D for capturing frames with float buffer
    save_path = f"frames_sphere_to_{shape_name}_collision"
    os.makedirs(save_path, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    point_cloud = o3d.geometry.PointCloud()
    points = sim_sphere.get_positions()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(point_cloud)

    # Reset camera so we see geometry
    vis.update_geometry(point_cloud)
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    def get_state():
        cur_pos = sim_sphere.get_positions()
        return np.vstack((cur_pos, new_coords))

    state = get_state()

    for step_i in range(steps):
        # RL step
        action, _ = model.predict(state)
        for j, p in enumerate(sim_sphere.particles):
            p.velocity = action[j]
            p.update(sim_sphere.dt)

        # Update geometry
        points = sim_sphere.get_positions()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # Capture float buffer => avoid black frames
        float_buffer = vis.capture_screen_float_buffer(True)
        img = (np.asarray(float_buffer) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        frame_path = os.path.join(save_path, f"frame_{step_i:04d}.png")
        cv2.imwrite(frame_path, img)
        print(f"[Visualization] Saved frame {step_i+1}/{steps}")

        time.sleep(0.03)

        # Next RL state
        state = get_state()

    vis.destroy_window()

    # 5) Create a single MP4
    output_video = f"sphere_to_{shape_name}_collision.mp4"
    create_video_from_frames(save_path, output_video, fps=30)
    print(f"[Run] Done! Video '{output_video}' shows sphere -> {shape_name} with collision penalty.")

###################################
# Main Execution
###################################
if __name__ == "__main__":
    # Example: sphere -> letter_A with collision RL model
    # run_collision_sphere_to_shape("letter_A", steps=200)
    # Or sphere -> cube:
     run_collision_sphere_to_shape("cube", steps=200)
