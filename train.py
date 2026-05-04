from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
ANALYSIS_DIR = BASE_DIR / "analysis"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)


class BiologicallyMotivatedWrapper(gym.Wrapper):
    """
    Custom reward wrapper implementing biologically-motivated reward function
    based on the proposal requirements.
    
    Reward components:
    - Forward velocity reward: Encourages movement in target direction
    - Upright posture reward: Penalizes excessive trunk lean
    - Energy penalty: Penalizes large joint torques
    - Foot contact reward: Bonus for heel-to-toe strikes
    - Fall penalty: Large negative reward if agent falls
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.forward_reward_weight = 1.25
        self.upright_reward_weight = 0.8
        self.energy_penalty_weight = 0.0005
        self.fall_penalty_weight = 1.0
        self.foot_contact_weight = 0.1
        self.torso_height_threshold = 0.8
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
       
        qpos = self.env.unwrapped.data.qpos
        qvel = self.env.unwrapped.data.qvel
        
       
        torso_x_pos = qpos[0]
        torso_y_pos = qpos[1]  # Height
        
        
        custom_reward = 0.0
        
       
        forward_velocity = qvel[0]
        forward_reward = self.forward_reward_weight * forward_velocity
        
        
        torso_orientation = qpos[3:7]  
        
        torso_quat = torso_orientation / np.linalg.norm(torso_orientation)
        
        pitch = 2 * np.arctan2(torso_quat[1], torso_quat[0])
        upright_penalty = -self.upright_reward_weight * np.abs(pitch)
        
        
        # Using action magnitude as proxy for energy consumption
        action_mag = np.sqrt(np.sum(action ** 2))
        energy_penalty = -self.energy_penalty_weight * action_mag
        
       
        fall_penalty = 0.0
        if torso_y_pos < self.torso_height_threshold:
            fall_penalty = -self.fall_penalty_weight * (self.torso_height_threshold - torso_y_pos)
        
       
        foot_contact_reward = 0.0
        if len(qvel) > 2 and qvel[2] < 0.1:  # Low vertical velocity
            foot_contact_reward = self.foot_contact_weight
        
        
        custom_reward = (forward_reward + upright_penalty + energy_penalty + 
                        fall_penalty + foot_contact_reward)
        
        
        total_reward = reward + custom_reward
        
        
        info['forward_reward'] = forward_reward
        info['upright_penalty'] = upright_penalty
        info['energy_penalty'] = energy_penalty
        info['fall_penalty'] = fall_penalty
        info['foot_contact_reward'] = foot_contact_reward
        info['custom_reward'] = custom_reward
        
        return obs, total_reward, terminated, truncated, info


def main() -> None:
    n_envs = 4
    env_name = "Humanoid-v5"  
    
    print(f"Setting up {n_envs} parallel environments: {env_name}")
    
    # Create training environment with custom reward wrapper
    train_env = make_vec_env(env_name, n_envs=n_envs, wrapper_class=BiologicallyMotivatedWrapper)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create evaluation environment
    eval_env = make_vec_env(env_name, n_envs=1, wrapper_class=BiologicallyMotivatedWrapper)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False
    
    # Setup callbacks for training
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_humanoid_biologically_motivated"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best_model"),
        log_path=str(LOG_DIR),
        eval_freq=25000,
        deterministic=True,
        render=False,
    )
    
    # Initialize PPO model with optimized hyperparameters
    print("Initializing PPO model with biologically-motivated reward wrapper...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        device="cpu",
        # Note: Removed tensorboard_log to avoid dependency issues
    )
    
    # Training configuration
    total_timesteps = 2_000_000
    print(f"\n{'='*60}")
    print(f"Training {env_name} for {total_timesteps:,} timesteps")
    print(f"Training with biologically-motivated custom reward function")
    print(f"{'='*60}\n")

    print("Custom Reward Components:")
    print("  Forward velocity reward (weight: 1.25)")
    print("  Upright posture reward (weight: 0.8)")
    print("  Energy penalty (weight: 0.0005)")
    print("  Foot contact reward (weight: 0.1)")
    print("  Fall penalty (weight: 1.0)\n")
    
    print(f"Training Configuration:")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Algorithm: PPO (Proximal Policy Optimization)")
    print(f"  Learning rate: 3e-4")
    print(f"  Steps per environment: 2048")
    print(f"  Batch size: 256")
    print(f"  Epochs per update: 10")
    print(f"  Discount factor (gamma): 0.99")
    print(f"  GAE lambda: 0.95")
    print(f"  Device: CPU")
    print()
    
    print("This training will:")
    print("  Save checkpoints every 50,000 timesteps")
    print("  Evaluate model every 25,000 timesteps")
    print("  Save best model automatically")
    print("  Estimated time: 4-8 hours on CPU\n")
    
    # Train the model
    print("Starting training...")
    print("="*60)
    print()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,  # Disable progress bar to avoid tqdm/rich dependency
    )
    
    # Save final model and normalization statistics
    final_model_path = MODEL_DIR / "ppo_humanoid_biologically_motivated_final"
    model.save(final_model_path)
    train_env.save(str(MODEL_DIR / "vecnormalize.pkl"))
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final model saved to: {final_model_path}.zip")
    print(f"Normalization stats saved to: {MODEL_DIR / 'vecnormalize.pkl'}")
    print(f"Best model saved to: {MODEL_DIR / 'best_model'}")
    print(f"Checkpoints saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
