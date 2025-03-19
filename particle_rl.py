import os
import time
import cv2
import gym
import numpy as np
from gym import spaces
import open3d as o3d
from stable_baselines3 import PPO

# =========================
# Shape Generators
# =========================

def generate_sphere(num_particles, radius=5):
    """Generates evenly distributed target points on a sphere."""
    positions = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
    for i in range(num_particles):
        y = 1 - (i / float(num_particles - 1)) * 2  # y goes from 1 to -1
        r = np.sqrt(1 - y * y) * radius
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        positions.append(np.array([x, y * radius, z]))
    return positions

def generate_cube(num_particles, cube_size=5):
    """Generates target positions in a cube formation."""
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
    """
    Generates a simple point cloud approximating a letter A using
    two diagonal lines and a horizontal crossbar.
    """
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
    """Returns a target shape based on the given name."""
    if shape_name == "sphere":
        return generate_sphere(num_particles)
    elif shape_name == "cube":
        return generate_cube(num_particles)
    elif shape_name == "letter_A":
        return generate_letter_A(num_particles)
    else:
        raise ValueError("Unknown shape: " + shape_name)

# =========================
# Particle and Simulation
# =========================

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

# =========================
# Simulation that Starts as a Sphere
# =========================

class SimulationStartAsSphere:
    """
    This simulation sets the initial positions to the sphere shape,
    so we begin with a perfect sphere arrangement. 
    Then we can reassign each particle's target to a new shape.
    """
    def __init__(self, sphere_coords):
        self.dt = 0.1
        self.particles = []
        # Each particle starts at the sphere coordinate
        for coord in sphere_coords:
            self.particles.append(Particle(position=coord, target=coord))
    
    def update(self):
        for p in self.particles:
            p.update(self.dt)
    
    def get_positions(self):
        return np.array([p.position for p in self.particles])

# =========================
# Visualization & Video
# =========================

def visualize_and_record(sim, save_path="frames", num_frames=100):
    """Visualizes the simulation in 3D and saves frames as images."""
    os.makedirs(save_path, exist_ok=True)
    print(f"[Visualization] Saving frames to '{save_path}' ...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    point_cloud = o3d.geometry.PointCloud()
    points = sim.get_positions()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(point_cloud)
    
    for i in range(num_frames):
        sim.update()
        points = sim.get_positions()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        frame_path = os.path.join(save_path, f"frame_{i:04d}.png")
        vis.capture_screen_image(frame_path)
        print(f"[Visualization] Saved frame {i+1}/{num_frames}")
        time.sleep(0.05)
    vis.destroy_window()
    print("[Visualization] Finished recording frames.")

def create_video_from_frames(frame_folder="frames", output_video="particle_simulation.mp4", fps=30):
    """Creates a video from saved frame images."""
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
        frame = cv2.imread(os.path.join(frame_folder, frame_file))
        video.write(frame)
    video.release()
    print(f"[Video] Video saved as '{output_video}'.")

# =========================
# Optional: RL Environment for Training
# =========================

class ParticleEnv(gym.Env):
    """
    Environment used to train a universal RL model.
    Particles start at random positions and have shape-based targets.
    """
    def __init__(self, num_particles):
        super(ParticleEnv, self).__init__()
        self.num_particles = num_particles
        self.sim = None
        self.target_shape = None

        # Observations: current positions (N x 3) + target positions (N x 3)
        self.observation_space = spaces.Box(low=-20, high=20,
                                            shape=(num_particles * 2, 3),
                                            dtype=np.float32)
        # Actions: velocity for each particle
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(num_particles, 3),
                                       dtype=np.float32)
    
    def load_shape(self, shape_name):
        coords = load_custom_shape(shape_name, self.num_particles)
        # Create random initial positions
        particles = []
        for c in coords:
            pos = np.random.uniform(-10, 10, 3)
            p = Particle(pos, c)
            particles.append(p)
        self.sim = Simulation(particles)
        self.target_shape = coords
    
    def step(self, action):
        for i, p in enumerate(self.sim.particles):
            p.velocity = action[i]
            p.update(self.sim.dt)
        current_positions = self.sim.get_positions()
        state = np.vstack((current_positions, self.target_shape))
        # Reward: negative mean distance
        reward = -np.mean([np.linalg.norm(p.target - p.position) for p in self.sim.particles])
        done = all(np.linalg.norm(p.target - p.position) < 0.1 for p in self.sim.particles)
        return state, reward, done, {}
    
    def reset(self, shape_name="sphere"):
        self.load_shape(shape_name)
        state = np.vstack((self.sim.get_positions(), self.target_shape))
        return state

def train_rl_model():
    """Trains a universal RL model on multiple shapes, saved as universal_particle_model.zip."""
    print("[Training] Checking for universal_particle_model.zip...")
    if os.path.exists("universal_particle_model.zip"):
        print("[Training] Model file already exists. Skipping training.")
        return None  # No need to re-train
    print("[Training] Starting RL training on multiple shapes...")

    env = ParticleEnv(500)
    shapes = ["sphere", "cube", "letter_A"]
    model = PPO("MlpPolicy", env, verbose=1)
    
    total_timesteps = 30000  # Adjust as needed
    timesteps_per_episode = 100
    episodes = total_timesteps // timesteps_per_episode
    for i in range(episodes):
        shape = np.random.choice(shapes)
        env.reset(shape)
        print(f"[Training] Episode {i+1}/{episodes} - shape: {shape}")
        model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)
    model.save("universal_particle_model")
    print("[Training] Model saved as 'universal_particle_model.zip'.")

# =========================
# Main: Single-Phase Transition (Sphere -> New Shape)
# =========================

def run_model_sphere_to_shape(shape_name="letter_A", steps=200):
    """
    1) Particles START in a perfect sphere arrangement (positions == sphere coords).
    2) Instantly assign new shape as target.
    3) Use RL model to move from sphere -> shape in one phase.
    4) Record frames into a single MP4.
    """
    print("[Run] Checking for 'universal_particle_model.zip'...")
    if not os.path.exists("universal_particle_model.zip"):
        print("[Run] No model found. Training now...")
        train_rl_model()

    print(f"[Run] Loading RL model from 'universal_particle_model.zip'...")
    model = PPO.load("universal_particle_model")

    # Create a perfect sphere arrangement as initial positions
    print("[Run] Creating initial sphere arrangement for 500 particles...")
    sphere_coords = load_custom_shape("sphere", 500)
    sim_sphere = SimulationStartAsSphere(sphere_coords)

    # Assign the new shape's coords as targets
    new_coords = load_custom_shape(shape_name, 500)
    for i, p in enumerate(sim_sphere.particles):
        p.target = new_coords[i]

    # We'll record frames while we let RL control velocities
    os.makedirs("frames_sphere_to_shape", exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    point_cloud = o3d.geometry.PointCloud()
    points = sim_sphere.get_positions()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(point_cloud)

    # Helper to create observation for RL (positions + targets)
    def get_state():
        cur_pos = sim_sphere.get_positions()
        return np.vstack((cur_pos, new_coords))

    state = get_state()
    for i in range(steps):
        # RL model predicts velocities
        action, _ = model.predict(state)
        # Apply to each particle
        for j, p in enumerate(sim_sphere.particles):
            p.velocity = action[j]
            p.update(sim_sphere.dt)
        # Visualization
        points = sim_sphere.get_positions()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        frame_path = os.path.join("frames_sphere_to_shape", f"frame_{i:04d}.png")
        vis.capture_screen_image(frame_path)
        time.sleep(0.03)
        # Next state
        state = get_state()

    vis.destroy_window()
    output_video = f"sphere_to_{shape_name}.mp4"
    create_video_from_frames("frames_sphere_to_shape", output_video, fps=30)
    print(f"[Run] Done! Video '{output_video}' shows sphere -> {shape_name} in one phase.")

# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    # If the model doesn't exist, we'll train automatically.
    # Then we do a single-phase run from sphere -> letter_A.
    run_model_sphere_to_shape("letter_A", steps=200)
    # You can change "letter_A" to "cube" or "sphere" to see different final shapes.
