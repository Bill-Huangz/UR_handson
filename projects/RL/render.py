import os
# Force headless rendering backends immediately before any other library imports
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Now import the rest of your libraries safely
import numpy as np
import gymnasium as gym
import imageio
from tqdm import tqdm
# 1. Initialize the environment with explicit render mode
env_name = "HalfCheetah-v5"
print(f"🎬 Initializing {env_name}...")

# 'rgb_array' tells Gymnasium to render to an offscreen buffer (perfect for our EGL setup)
env = gym.make(env_name, render_mode="rgb_array")

obs, info = env.reset()
frames = []

# Optional: Capture the very first frame at step 0
first_frame = env.render()
if first_frame is not None:
    frames.append(first_frame)

print("🏃 Running simulation steps...")
for _ in tqdm(range(200), desc="Simulating"):
    # Sample a random action (or insert your RL agent's action here)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Capture the current frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)
        
    if terminated or truncated:
        obs, info = env.reset()

env.close()

# 2. Compile and save the video using imageio
if len(frames) > 0:
    actual_shape = frames[0].shape
    print(f"📊 Captured {len(frames)} frames. Internal layout shape: {actual_shape}")
    
    video_path = "halfcheetah_v5_perfect.mp4"
    print(f"💾 Compiling video container via imageio...")
    
    # imageio writes standard RGB arrays flawlessly on headless AMD GPUs
    imageio.mimsave(video_path, frames, fps=30)
    
    print(f"🎉 Success! Playable video saved at: {os.path.abspath(video_path)}")
else:
    print("❌ Critical Error: No frames were captured. Verify that render_mode='rgb_array' is set.")