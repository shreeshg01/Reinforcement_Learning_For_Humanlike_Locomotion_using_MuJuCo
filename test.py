from pathlib import Path
import numpy as np
import json
from typing import Dict, List, Tuple
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder
import gymnasium as gym


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
VIDEO_DIR = BASE_DIR / "videos"
ANALYSIS_DIR = BASE_DIR / "analysis"
VIDEO_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)


class GaitAnalyzer:
    """
    Analyzes the trained humanoid agent's walking gait and compares it to
    known human locomotion principles.
    """
    
    def __init__(self):
        self.joint_angles = []
        self.joint_velocities = []
        self.center_of_mass = []
        self.actions = []
        self.rewards = []
        self.contact_forces = []
        self.time_steps = []
        self.torso_heights = []
        self.forward_velocities = []
        
    def record_frame(self, env, obs, action, reward, step):
        """Record gait data for a single timestep."""
        data = env.unwrapped.data
        
        
        qpos = data.qpos
        self.joint_angles.append(qpos.copy())
        
        
        qvel = data.qvel
        self.joint_velocities.append(qvel.copy())
        
       
        com_x = qpos[0]
        com_y = qpos[1]
        self.center_of_mass.append([com_x, com_y])
        
        # Record torso height
        self.torso_heights.append(com_y)
        
        
        self.forward_velocities.append(qvel[0])
        
       
        self.actions.append(action.copy())
        
       
        self.rewards.append(reward)
        
       
        if hasattr(data, 'cfrc_ext'):
            self.contact_forces.append(data.cfrc_ext.copy())
        
        self.time_steps.append(step)
    
    def compute_statistics(self) -> Dict:
        """Compute gait statistics from recorded data."""
        statistics = {}
        
        
        joint_angles = np.array(self.joint_angles)
        joint_velocities = np.array(self.joint_velocities)
        center_of_mass = np.array(self.center_of_mass)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        torso_heights = np.array(self.torso_heights)
        forward_velocities = np.array(self.forward_velocities)
        
       
        statistics['forward_velocity'] = {
            'mean': float(np.mean(forward_velocities)),
            'std': float(np.std(forward_velocities)),
            'max': float(np.max(forward_velocities)),
            'min': float(np.min(forward_velocities))
        }
        
       
        statistics['torso_height'] = {
            'mean': float(np.mean(torso_heights)),
            'std': float(np.std(torso_heights)),
            'max': float(np.max(torso_heights)),
            'min': float(np.min(torso_heights))
        }
        
       
        statistics['joint_angles'] = {
            'num_joints': int(joint_angles.shape[1]),
            'mean_per_joint': joint_angles.mean(axis=0).tolist(),
            'std_per_joint': joint_angles.std(axis=0).tolist(),
            'range_per_joint': [
                [float(joint_angles[:, i].min()), float(joint_angles[:, i].max())]
                for i in range(joint_angles.shape[1])
            ]
        }
        
       
        action_magnitudes = np.sqrt(np.sum(actions ** 2, axis=1))
        statistics['energy_cost'] = {
            'mean_action_magnitude': float(np.mean(action_magnitudes)),
            'std_action_magnitude': float(np.std(action_magnitudes)),
            'total_action_magnitude': float(np.sum(action_magnitudes))
        }
        
        
        statistics['episode_reward'] = {
            'total': float(np.sum(rewards)),
            'mean_per_step': float(np.mean(rewards)),
            'std': float(np.std(rewards))
        }
        
       
        com_displacement = np.linalg.norm(center_of_mass[-1] - center_of_mass[0])
        statistics['center_of_mass'] = {
            'total_displacement': float(com_displacement),
            'mean_height': float(np.mean(center_of_mass[:, 1])),
            'height_oscillation': float(np.std(center_of_mass[:, 1]))
        }
        
        
        if joint_angles.shape[1] > 5:
            
            statistics['leg_symmetry'] = {
                'note': 'Computed from alternating joint angles',
                'mean_difference': float(np.mean(np.abs(joint_angles[:, 4] - joint_angles[:, 10]))) if joint_angles.shape[1] > 10 else None
            }
        
        return statistics
    
    def save_analysis(self, filename: str = "gait_analysis.json"):
        """Save gait analysis to JSON file."""
        analysis = {
            'statistics': self.compute_statistics(),
            'frame_count': len(self.time_steps),
            'duration_seconds': len(self.time_steps) / 30.0,  # Assuming 30 FPS
        }
        
        output_path = ANALYSIS_DIR / filename
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"✓ Gait analysis saved to: {output_path}")
        return analysis
    
    def print_comparison_to_human_gait(self, analysis: Dict):
        """
        Compare the agent's gait to known human locomotion principles.
        
        Reference: Human gait typically shows:
        - Forward velocity: 1.0-1.5 m/s for normal walking
        - Stance phase: ~60% of gait cycle
        - Double support: ~20% of gait cycle
        - Torso height variation: 3-5 cm during normal gait
        - Energy cost: proportional to walking speed
        """
        
        print("\n" + "="*70)
        print("GAIT ANALYSIS: Comparison to Human Locomotion Principles")
        print("="*70 + "\n")
        
        stats = analysis['statistics']
        
       
        print("1. FORWARD LOCOMOTION:")
        print("-" * 70)
        fv = stats['forward_velocity']['mean']
        print(f"   Agent forward velocity: {fv:.4f} m/s")
        print(f"   Human normal walking:  1.0-1.5 m/s")
        if 0.5 < fv < 2.0:
            print("   ✓ Within reasonable human walking range")
        else:
            print("   ⚠ Atypical velocity (too slow or too fast)")
        print()
        
      
        print("2. POSTURAL STABILITY:")
        print("-" * 70)
        height_std = stats['torso_height']['std']
        mean_height = stats['torso_height']['mean']
        print(f"   Mean torso height: {mean_height:.4f} m")
        print(f"   Height oscillation (std): {height_std:.4f} m")
        print(f"   Human torso oscillation: 0.03-0.05 m")
        if height_std < 0.1:
            print("   Reasonable postural stability")
        else:
            print("   Excessive torso bobbing (unstable gait)")
        print()
        
       
        print("3. ENERGY EFFICIENCY:")
        print("-" * 70)
        energy = stats['energy_cost']['mean_action_magnitude']
        print(f"   Mean action magnitude (energy proxy): {energy:.4f}")
        print(f"   Human walking follows minimal energy principle")
        print("    Lower values indicate more efficient locomotion")
        print()
        
        
        print("4. CENTER OF MASS DYNAMICS:")
        print("-" * 70)
        com_stats = stats['center_of_mass']
        print(f"   Total COM displacement: {com_stats['total_displacement']:.4f} m")
        print(f"   COM height oscillation: {com_stats['height_oscillation']:.4f} m")
        print(f"   Human COM oscillation: ~0.05 m vertically")
        print()
        
        
        print("5. JOINT ANGLE RANGES:")
        print("-" * 70)
        ja = stats['joint_angles']
        print(f"   Number of joints: {ja['num_joints']}")
        print(f"   First 3 joint ranges:")
        for i in range(min(3, ja['num_joints'])):
            r = ja['range_per_joint'][i]
            print(f"      Joint {i}: [{r[0]:.4f}, {r[1]:.4f}] rad")
        print()
        
        
        print("="*70)
        print("OVERALL ASSESSMENT:")
        print("="*70)
        if all([
            0.5 < fv < 2.0,
            height_std < 0.1,
            com_stats['height_oscillation'] < 0.15
        ]):
            print("Agent demonstrates biologically plausible locomotion!")
            print("  The emerged gait shows characteristics aligned with human walking.")
        else:
            print("Agent's gait shows some atypical characteristics.")
            print("  Consider tuning reward function for biological realism.")
        print("="*70 + "\n")


def run_evaluation(model_path: str, norm_path: str, max_steps: int = 1000, 
                   render: bool = True) -> Tuple[float, Dict, List]:
    """
    Run evaluation of trained model and perform gait analysis.
    
    Returns:
        - Total episode reward
        - Gait analysis statistics
        - Recorded frames
    """
    
    print(f"\nLoading model from: {model_path}")
    print(f"Loading normalization from: {norm_path}")
    
   
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(norm_path).exists():
        raise FileNotFoundError(f"Normalization file not found: {norm_path}")
    
   
    env = make_vec_env("Humanoid-v5", n_envs=1)
    env = VecNormalize.load(str(norm_path), env)
    env.training = False
    env.norm_reward = False
    
    
    model = PPO.load(model_path)
    
   
    analyzer = GaitAnalyzer()
    
   
    print("\nRunning evaluation and recording gait data...")
    obs = env.reset()
    frames = []
    total_reward = 0.0
    step_count = 0
    
    raw_env = env.venv.envs[0]
    
    for step in range(max_steps):
       
        if render:
            frame = raw_env.render()
            if frame is not None:
                frames.append(frame)
        
       
        action, _ = model.predict(obs, deterministic=True)
        
      
        obs, reward, done, info = env.step(action)
        total_reward += float(reward[0])
        
       
        analyzer.record_frame(raw_env, obs, action[0], float(reward[0]), step)
        
        step_count += 1
        
        if done[0]:
            print(f"Episode finished after {step_count} steps")
            break
    
    
    if frames:
        video_path = VIDEO_DIR / "humanoid_gait_analysis.mp4"
        imageio.mimsave(video_path, frames, fps=30)
        print(f"✓ Video saved to: {video_path}")
    else:
        print("⚠ No frames captured for video")
    
    
    print("\nAnalyzing gait characteristics...")
    analysis = analyzer.save_analysis("gait_analysis.json")
    analyzer.print_comparison_to_human_gait(analysis)
    
    
    env.close()
    
    return total_reward, analysis, frames


def main() -> None:
    model_path = MODEL_DIR / "ppo_humanoid_biologically_motivated_final.zip"
    norm_path = MODEL_DIR / "vecnormalize.pkl"
    
    print("="*70)
    print("HUMANOID GAIT ANALYSIS AND COMPARISON")
    print("="*70)
    print("\nThis script evaluates the trained RL agent and performs detailed")
    print("gait analysis, comparing the emergent locomotion to human walking principles.")
    print()
    
    
    total_reward, analysis, frames = run_evaluation(
        str(model_path), 
        str(norm_path), 
        max_steps=1000,
        render=True
    )
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Total Episode Reward: {total_reward:.2f}")
    print(f"Steps Executed: {analysis['frame_count']}")
    print(f"Episode Duration: {analysis['duration_seconds']:.2f} seconds")
    print("="*70)
    
    print("\n✓ All analysis files saved to:", ANALYSIS_DIR)
    print("  - gait_analysis.json: Detailed statistics")
    print("  - humanoid_gait_analysis.mp4: Visual gait recording")


if __name__ == "__main__":
    main()
