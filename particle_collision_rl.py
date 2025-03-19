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
# Particle & Simulation
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
        # Move with limited speed; if very close, take a smaller step.
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
# RL Environment with Collision Penalty
# =========================

class ParticleEnvCollision(gym.Env):
    """
    Environment that penalizes collisions among particles.
    Single RL agent controlling all particles (action space = Nx3).
    Reward = -(mean distance to target) + collision penalty.
    """
    def __init__(self, num_particles, collision_threshold=1.0, collision_penalty=1.0):
        super(ParticleEnvCollision, self).__init__()
        self.num_particles = num_particles
        self.collision_threshold = collision_threshold
        self.collision_penalty_value = collision_penalty

        self.sim = None
        self.target_shape = None
        
        # Observations: current positions (N x 3) + target positions (N x 3)
        self.observation_space = spaces.Box(low=-20, high=20,
                                            shape=(num_particles * 2, 3),
                                            dtype=np.float32)
        # Actions: velocity for each particle (N x 3)
        self.action_space = spaces.Box(low=-1, high=1,
                                       shape=(num_particles, 3),
                                       dtype=np.float32)
    
    def load_shape(self, shape_name):
        coords = load_custom_shape(shape_name, self.num_particles)
        # Create random initial positions + shape-based targets
        particles = []
        for c in coords:
            pos = np.random.uniform(-10, 10, 3)
            p = Particle(pos, c)
            particles.append(p)
        self.sim = Simulation(particles)
        self.target_shape = coords
    
    def step(self, action):
        # 1) Apply the action to each particle
        for i, p in enumerate(self.sim.particles):
            p.velocity = action[i]
            p.update(self.sim.dt)

        # 2) Calculate distances to target
        current_positions = self.sim.get_positions()
        mean_distance = np.mean([
            np.linalg.norm(p.target - p.position)
            for p in self.sim.particles
        ])

        # 3) Compute collision penalty
        #    For each pair of particles, if they're closer than threshold, penalize.
        collision_penalty = 0.0
        for i in range(self.num_particles):
            for j in range(i+1, self.num_particles):
                dist = np.linalg.norm(self.sim.particles[i].position
                                      - self.sim.particles[j].position)
                if dist < self.collision_threshold:
                    # The closer they are, the bigger the penalty.
                    # Example: penalty grows linearly as distance goes below threshold.
                    collision_penalty += self.collision_penalty_value * (self.collision_threshold - dist)

        # 4) Combine into total reward
        #    We use negative mean distance plus negative collision penalty
        #    so that the agent tries to reduce distance to target and avoid collisions.
        reward = -mean_distance - collision_penalty

        # 5) Check if done (all particles near target)
        done = all(
            np.linalg.norm(p.target - p.position) < 0.1
            for p in self.sim.particles
        )

        # 6) Build next state
        state = np.vstack((current_positions, self.target_shape))

        return state, reward, done, {}
    
    def reset(self, shape_name="sphere"):
        self.load_shape(shape_name)
        state = np.vstack((self.sim.get_positions(), self.target_shape))
        return state

# =========================
# Training and Single-Phase Run
# =========================

def train_rl_model_collision():
    """
    Trains an RL model on multiple shapes with collision penalties.
    Saves the model as 'particle_model_collision.zip'.
    """
    if os.path.exists("particle_model_collision.zip"):
        print("[Training] 'particle_model_collision.zip' already exists. Skipping re-training.")
        return

    print("[Training] Starting RL training with collision penalty...")
    env = ParticleEnvCollision(
        num_particles=200,    # you can change the number of particles
        collision_threshold=1.0,
        collision_penalty=0.5
    )

    # We'll train on multiple shapes randomly
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
    print("[Training] Model saved as 'particle_model_collision.zip'.")

def run_collision_sphere_to_shape(shape_name="letter_A", steps=200):
    """
    1) Loads the collision-penalizing RL model.
    2) Starts with a perfect sphere arrangement.
    3) Assigns the new shape as targets.
    4) Moves from sphere -> shape in one phase, penalizing collisions.
    """
    # 1) If the model doesn't exist, train it
    if not os.path.exists("particle_model_collision.zip"):
        train_rl_model_collision()

    print("[Run] Loading model with collision penalty: 'particle_model_collision.zip'...")
    model = PPO.load("particle_model_collision")

    # 2) Start as a perfect sphere arrangement
    print("[Run] Creating initial sphere arrangement.")
    sphere_coords = load_custom_shape("sphere", 200)  # 200 must match the environment's particle count
    # Create particles at sphere coords
    from_sphere_particles = []
    for coord in sphere_coords:
        # start position = sphere coordinate, target = sphere coordinate
        from_sphere_particles.append(Particle(position=coord, target=coord))

    # 3) Assign the new shape
    new_coords = load_custom_shape(shape_name, 200)
    for i, p in enumerate(from_sphere_particles):
        p.target = new_coords[i]

    # Create a simulation object for visualization
    sim = Simulation(from_sphere_particles)

    # 4) Record frames while letting RL model control velocities
    os.makedirs("frames_sphere_to_shape_collision", exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    point_cloud = o3d.geometry.PointCloud()
    points = sim.get_positions()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(point_cloud)

    # Build a function to get the RL "state": positions + target coords
    def get_state():
        cur_pos = sim.get_positions()
        return np.vstack((cur_pos, new_coords))

    state = get_state()
    for i in range(steps):
        # RL model predicts velocity for each particle
        action, _ = model.predict(state)
        # Apply
        for j, p in enumerate(sim.particles):
            p.velocity = action[j]
            p.update(sim.dt)

        # Visualization
        points = sim.get_positions()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        frame_path = os.path.join("frames_sphere_to_shape_collision", f"frame_{i:04d}.png")
        vis.capture_screen_image(frame_path)
        time.sleep(0.03)

        # Next state
        state = get_state()

    vis.destroy_window()

    # 5) Compile frames into video
    output_video = f"sphere_to_{shape_name}_collision.mp4"
    create_video_from_frames("frames_sphere_to_shape_collision", output_video, fps=30)
    print(f"[Run] Done! Video '{output_video}' shows sphere -> {shape_name} with collision penalty.")

# =========================
# Main Execution
# =========================

if __name__ == "__main__":
    # 1) Train or skip if model already exists
    # train_rl_model_collision()  # uncomment if you want to force training now

    # 2) Single-phase transition from sphere -> letter_A with collision avoidance
    run_collision_sphere_to_shape("letter_A", steps=200)
    # You can also try "cube" or "sphere"
    # run_collision_sphere_to_shape("cube", steps=200)
