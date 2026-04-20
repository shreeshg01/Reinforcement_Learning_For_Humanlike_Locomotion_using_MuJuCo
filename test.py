from pathlib import Path

import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
VIDEO_DIR = BASE_DIR / "videos"
VIDEO_DIR.mkdir(exist_ok=True)


def main() -> None:
    model_path = MODEL_DIR / "ppo_humanoid_final.zip"
    norm_path = MODEL_DIR / "vecnormalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing normalization file: {norm_path}")

    env = make_vec_env("Humanoid-v5", n_envs=1)
    env = VecNormalize.load(str(norm_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path)

    obs = env.reset()
    frames = []
    total_reward = 0.0
    max_steps = 1000

    raw_env = env.venv.envs[0]

    for _ in range(max_steps):
        frame = raw_env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward[0])

        if done[0]:
            break

    video_path = VIDEO_DIR / "humanoid_test.mp4"
    if frames:
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Saved video to: {video_path}")
    else:
        print("No frames captured. Video not saved.")

    print(f"Episode reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()