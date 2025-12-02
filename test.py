import gymnasium as gym
import torch
import numpy as np
import argparse
import glfw
import mujoco

# Import our custom modules
import module1 as m1
import module2 as m2
import module3 as m3

def parse_args():
    parser = argparse.ArgumentParser(description="Test PPO Agent for Humanoid Locomotion")
    parser.add_argument("--model-path", required=True, help="Path to the trained model file (.pt)")
    parser.add_argument("--image-path", help="Path to an image for the initial pose")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = m2.HumanoidEnvWrapper(render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    agent = m3.PPOAgent(obs_dim, act_dim).to(device)

    try:
        agent.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        exit()

    agent.eval() # Set the agent to evaluation mode

    initial_pose = None
    if args.image_path:
        initial_pose = m1.get_initial_pose_from_image(args.image_path)
        print(f"Using initial pose from {args.image_path}")
    else:
        print("No image path provided. Using default initial pose.")

    state, _ = env.reset(initial_pose=initial_pose)
    done = False
    step = 0

    # Set up camera view
    viewer = env.unwrapped.mujoco_renderer.viewer
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = 0
    glfw.maximize_window(viewer.window)

    print("Simulation running. Close the window to exit.")
    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            # Get the mean action (deterministic) for testing
            action, _, _, _ = agent.get_action_and_value(state_tensor)
        
        next_state, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated
        
        state = next_state
        step += 1
        env.render()

    env.close()
    print("Simulation finished.")