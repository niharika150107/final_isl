import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from tqdm import tqdm

# Import our custom modules
import module1 as m1
import module2 as m2
import module3 as m3

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training for Humanoid Locomotion")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps to run per environment per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size for PPO updates")
    parser.add_argument("--n-update-epochs", type=int, default=10, help="Number of epochs to update the policy")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-kl", type=float, default=0.01, help="Maximum KL divergence for early stopping")
    parser.add_argument("--save-dir", default="models", help="Directory to save models")
    parser.add_argument("--image-dir", default="pose_images", help="Directory with initial pose images")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.exists(args.image_dir) or not os.listdir(args.image_dir):
        print(f"Warning: Image directory '{args.image_dir}' is empty or does not exist.")
        print("The agent will always start from the default pose.")

    env = m2.HumanoidEnvWrapper()
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = m3.PPOAgent(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    
    buffer = m3.PPOBuffer(obs_dim, act_dim, args.n_steps, 1, device, args.gamma, args.gae_lambda)

    pose_images = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        initial_pose = None
        if pose_images:
            image_path = np.random.choice(pose_images)
            initial_pose = m1.get_initial_pose_from_image(image_path)
            print(f"Using initial pose from {os.path.basename(image_path)}")

        obs, _ = env.reset(initial_pose=initial_pose)
        episode_reward = 0

        # 1. Collect Trajectories
        for step in range(args.n_steps):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)
                
            next_obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            episode_reward += reward
            
            buffer.store(obs, action, reward, value.squeeze(0).cpu().numpy(), terminated, truncated, logprob.squeeze(0).cpu().numpy())
            
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        # 2. Calculate Advantage
        with torch.no_grad():
            last_obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = agent.get_value(last_obs_tensor).squeeze(0).cpu().numpy()
        
        advantages, returns = buffer.calculate_advantages(
            torch.tensor(last_value), 
            torch.tensor([terminated], dtype=torch.float32), 
            torch.tensor([truncated], dtype=torch.float32)
        )

        # 3. Update Policy
        obs_batch, act_batch, old_logprob_batch, _, _, _, _ = buffer.get()
        
        dataset = torch.utils.data.TensorDataset(obs_batch, act_batch, old_logprob_batch, advantages, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        for _ in range(args.n_update_epochs):
            for obs_batch, act_batch, old_logprob_batch, adv_batch, ret_batch in dataloader:
                _, new_logprob, entropy, new_value = agent.get_action_and_value(obs_batch, act_batch)
                
                logratio = new_logprob - old_logprob_batch
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                
                policy_loss1 = -adv_batch * ratio
                policy_loss2 = -adv_batch * torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                # CORRECTED: Squeeze ret_batch to match new_value.squeeze() shape
                value_loss = nn.functional.mse_loss(new_value.squeeze(), ret_batch.squeeze())
                
                # CORRECTED: Take the mean of the entropy tensor to get a scalar loss
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy.mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
                if approx_kl > args.max_kl:
                    break

        print(f"Epoch Reward: {episode_reward:.2f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")

        if (epoch + 1) % 50 == 0:
            model_path = os.path.join(args.save_dir, f"ppo_humanoid_epoch_{epoch+1}.pt")
            torch.save(agent.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    env.close()
    print("Training finished.")