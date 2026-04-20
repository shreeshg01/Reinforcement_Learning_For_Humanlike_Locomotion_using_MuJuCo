from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


def main() -> None:
    n_envs = 4

    train_env = make_vec_env("Humanoid-v5", n_envs=n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env("Humanoid-v5", n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False
    eval_env.norm_reward = False

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_humanoid"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODEL_DIR / "best_model"),
        log_path=str(LOG_DIR),
        eval_freq=25000,
        deterministic=True,
        render=False,
    )

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
    )

    total_timesteps = 2_000_000
    print(f"Training Humanoid-v5 for {total_timesteps} timesteps...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,
    )

    model.save(MODEL_DIR / "ppo_humanoid_final")
    train_env.save(str(MODEL_DIR / "vecnormalize.pkl"))

    train_env.close()
    eval_env.close()

    print("Training complete.")
    print(f"Saved final model to: {MODEL_DIR / 'ppo_humanoid_final.zip'}")
    print(f"Saved normalization stats to: {MODEL_DIR / 'vecnormalize.pkl'}")


if __name__ == "__main__":
    main()