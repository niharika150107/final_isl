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

# --- VISUAL CUSTOMIZATION FUNCTIONS (from final.py) ---

def set_robot_color(env, color):
    """
    Set the color of all robot geoms in the MuJoCo environment.

    Args:
        env: The gymnasium environment instance.
        color: A tuple of (r, g, b, a) values, each between 0 and 1.
               For example, (1.0, 0.0, 0.0, 1.0) is opaque red.
    """
    model = env.unwrapped.model
    # Iterate through all geoms in the model
    for i in range(model.ngeom):
        # The worldbody (which often contains the floor) has an ID of 0.
        # We skip it to avoid changing the floor color here.
        if model.geom_bodyid[i] != 0:
            model.geom_rgba[i] = color

def set_floor_color(env, color):
    """
    Set the color of the floor in the MuJoCo environment and remove any texture.

    Args:
        env: The gymnasium environment instance.
        color: A list of [r, g, b, a] values, each between 0 and 1.
    """
    model = env.unwrapped.model
    for i in range(model.ngeom):
        # The floor is part of the worldbody, which has an ID of 0.
        if model.geom_bodyid[i] == 0:
            # Check if the floor geom has a material
            mat_id = model.geom_matid[i]
            if mat_id != -1:  # -1 means no material is assigned
                # Get the material object
                mat = model.mat(mat_id)
                # Set the material's texture to -1 (no texture). This removes the checkerboard.
                mat.texid = -1
                # Set the material's base color
                mat.rgba = color
            break  # Exit after finding the floor

# --- END VISUAL CUSTOMIZATION FUNCTIONS ---


def parse_args():
    parser = argparse.ArgumentParser(description="Test PPO Agent for Humanoid Locomotion")
    parser.add_argument("--model-path", required=True, help="Path to the trained model file (.pt)")
    parser.add_argument("--image-path", help="Path to an image for the initial pose")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = m2.HumanoidEnvWrapper(render_mode="human")

    # --- APPLY VISUAL CUSTOMIZATION ---
    # 1. Set the entire robot to a nice blue
    robot_color = (0.2, 0.6, 1.0, 1.0)  # A nice blue
    set_robot_color(env, robot_color)

    # 2. Set the floor to solid white.
    floor_color = [1, 1, 1, 1]  # Solid white
    set_floor_color(env, floor_color)
    # ---------------------------------

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
