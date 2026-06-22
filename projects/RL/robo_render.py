import os
import numpy as np
import robosuite as suite
import imageio  # 👈 Bypasses picky OpenCV codecs completely

# 1. Initialize Robosuite (Using plural keywords to ensure size safety)
env = suite.make(
    env_name="Lift",                 
    robots="Panda",                  
    has_renderer=False,              
    has_offscreen_renderer=True,     # Harnesses your AMD EGL setup
    use_camera_obs=True,             
    camera_names="agentview",        
    # camera_heights=512,              # Plural to support modern robosuite specs
    # camera_widths=512,
)

obs = env.reset()
frames = []

print("🎥 Running simulation steps...")
for _ in range(100):
    action = np.random.uniform(env.action_spec[0], env.action_spec[1])
    obs, reward, done, info = env.step(action)
    
    # Extract the camera image matrix
    raw_frame = obs["agentview_image"]
    
    # Robosuite renders upside down natively, flip it right side up
    raw_frame = np.flipud(raw_frame)
    
    # Ensure values are strictly standard 8-bit integers (0-255)
    if raw_frame.dtype != np.uint8:
        if raw_frame.max() <= 1.0:
            raw_frame = (raw_frame * 255).astype(np.uint8)
        else:
            raw_frame = raw_frame.astype(np.uint8)
            
    frames.append(raw_frame)

env.close()

# 2. Compile and save the video using imageio
if len(frames) > 0:
    actual_shape = frames[0].shape
    print(f"📊 Captured {len(frames)} frames. Internal layout shape: {actual_shape}")
    
    video_path = "robosuite_perfect.mp4"
    print(f"💾 Compiling video container via embedded ffmpeg...")
    
    # imageio infers size and handles encoding flawlessly in headless environments
    # Note: imageio natively uses RGB, so we don't need to shuffle channels to BGR!
    imageio.mimsave(video_path, frames, fps=20)
    
    print(f"🎉 Success! Playable video saved at: {video_path}")
else:
    print("❌ Critical Error: No frames were captured from the environment.")