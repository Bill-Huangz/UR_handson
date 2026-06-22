import csv
import numpy as np
import torch
import random 
import os 
from typing import NamedTuple
import imageio
from tqdm import tqdm

# import sys
# sys.path.append("..")
# from lite6_lift_env import Lite6LiftEnv

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

class PPOBuffer(object):
    def __init__(self, num_steps, obs_dim, act_dim, device):
        self.num_steps = num_steps
        self.obs = torch.zeros((num_steps, obs_dim)).to(device)
        self.actions = torch.zeros((num_steps, act_dim)).to(device)
        self.next_obs = torch.zeros((num_steps, obs_dim)).to(device)
        self.logprobs = torch.zeros((num_steps,)).to(device)
        self.rewards = torch.zeros((num_steps,)).to(device)
        self.dones = torch.zeros((num_steps,)).to(device)
        self.values = torch.zeros((num_steps,)).to(device)
        self.next_obs_values = torch.zeros((num_steps,)).to(device)

    def store(self, step, obs, action, next_obs, reward, done, value, next_obs_value,  logprob):
        self.obs[step] = obs
        self.actions[step] = action
        self.rewards[step] = reward
        self.dones[step] = done
        self.values[step] = value
        self.next_obs[step] = next_obs
        self.next_obs_values[step] = next_obs_value
        self.logprobs[step] = logprob

class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    rewards: torch.Tensor
        
class ReplayBuffer(object):
    def __init__(self, buffer_size, obs_dim, act_dim, device):

        self.buffer_size = buffer_size
        self.device = device

        self.obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, act_dim), dtype=torch.float32, device=device)

        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.terminations = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.truncations = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, termination, truncation):
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action

        self.rewards[self.pos] = reward
        self.terminations[self.pos] = termination
        self.truncations[self.pos] = truncation

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    @torch.no_grad()
    def load_dataset(self, obs, next_obs, actions, rewards, terminations, truncations):
        n = obs.shape[0]
        if n > self.buffer_size:
            raise ValueError(f"dataset size {n} > buffer_size {self.buffer_size}")

        obs = obs.to(self.device, non_blocking=True)
        next_obs = next_obs.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True)
        terminations = terminations.to(self.device, non_blocking=True)
        truncations = truncations.to(self.device, non_blocking=True)

        self.obs[:n].copy_(obs)
        self.next_obs[:n].copy_(next_obs)
        self.actions[:n].copy_(actions)
        self.rewards[:n].copy_(rewards)
        self.terminations[:n].copy_(terminations)
        self.truncations[:n].copy_(truncations)

        self.pos = n % self.buffer_size
        self.full = (n == self.buffer_size)

    @torch.no_grad()
    def save_buffer(self, path):
        n = self.buffer_size if self.full else self.pos

        data = {
            "obs": self.obs[:n].clone().cpu(),
            "next_obs": self.next_obs[:n].clone().cpu(),
            "actions": self.actions[:n].clone().cpu(),
            "rewards": self.rewards[:n].clone().cpu(),
            "terminations": self.terminations[:n].clone().cpu(),
            "truncations": self.truncations[:n].clone().cpu(),
            "pos": self.pos,
            "full": self.full,
            "buffer_size": self.buffer_size,
        }
        torch.save(data, path)
    

    def sample(self, batch_size, enc_batch=False):
        if self.full:
            if enc_batch:
                batch_inds = (torch.randint(1, self.buffer_size, (batch_size - 1,), device=self.device) + self.pos) % self.buffer_size
                batch_inds = torch.cat([batch_inds, torch.tensor([self.pos - 1], device=self.device)])
            else:
                batch_inds = (torch.randint(1, self.buffer_size, (batch_size,), device=self.device) + self.pos) % self.buffer_size
        else:
            if enc_batch:
                batch_inds = torch.randint(0, self.pos, (batch_size - 1,), device=self.device)
                batch_inds = torch.cat([batch_inds, torch.tensor([self.pos - 1], device=self.device)])
            else:
                batch_inds = torch.randint(0, self.pos, (batch_size,), device=self.device)
        return self._get_samples(batch_inds)
    
    
    def _get_samples(self, batch_inds):
        return ReplayBufferSamples(
            self.obs[batch_inds],
            self.actions[batch_inds],
            self.next_obs[batch_inds],
            self.terminations[batch_inds],
            self.truncations[batch_inds],
            self.rewards[batch_inds],
        )


class RunningMeanStd(object):
    def __init__(self, shape, device, dtype=torch.float32, epsilon=1e-4):
        self.device = device
        self.dtype = dtype

        self.mean = torch.zeros(shape, device=device, dtype=dtype)
        self.var = torch.ones(shape, device=device, dtype=dtype)

        self.count = torch.tensor(epsilon, device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], device=self.device, dtype=self.dtype)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    @torch.no_grad()
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count

        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
        
    @torch.no_grad()
    def normalize(self, x, clip_range=None, eps=1e-8):
        x = (x - self.mean) / (torch.sqrt(self.var)+ eps)
        if clip_range is not None:
            x = torch.clamp(x, -clip_range, clip_range)
        return x
        
class NormalizeReward(object):
    def __init__(self, gamma = 0.99, epsilon = 1e-8):
        self.return_rms = RunningMeanStd(shape=())
        self.discounted_reward = np.array([0.0])
        self.gamma = gamma
        self.epsilon = epsilon
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the reward statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the reward statistics."""
        self._update_running_mean = setting

    def normalize(self, reward, terminated):
        # Using the `discounted_reward` rather than `reward` makes no sense but for backward compatibility, it is being kept
        self.discounted_reward = self.discounted_reward * self.gamma * (1 - terminated) + float(reward)
        if self._update_running_mean:
            self.return_rms.update(self.discounted_reward)

        normalized_reward = reward / np.sqrt(self.return_rms.var + self.epsilon)
        return normalized_reward
    

class evaluate_management(object):
    def __init__(self, env, device, reset_seed, eval_episodes, log_path):
        super().__init__()
        self.env = env
        self.action_low, self.action_high = self.env.action_space.low, self.env.action_space.high

        if reset_seed is not None:
            self.obs, _ = self.env.reset(seed=reset_seed)
        else:
            self.obs, _ = self.env.reset()

        self.device = device
        self.eval_episodes = eval_episodes
        self.results = {}
        
        self.save_count = 0
        self.eval_csv_path = f"{log_path}/eval.csv"
        dir_path = os.path.dirname(self.eval_csv_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        if os.path.exists(self.eval_csv_path):
            os.remove(self.eval_csv_path)
        self.save_flag = False
    
    def evaluate(self, agent, steps): # Removed writer parameter
        episodic_count = 0
        rewards = 0.0
        lens = 0.0
        episode_success = False
        num_success = 0
        
        # evaluate
        while episodic_count < self.eval_episodes:
            with torch.inference_mode():
                actions = agent.get_action(torch.as_tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0), test=True)
            next_obs, reward, terminations, truncations, _ = self.env.step(np.clip(actions[0].cpu().numpy(), self.action_low, self.action_high))
            rewards += reward
            lens += 1
            self.obs = next_obs

            if hasattr(self.env, "_check_success"):
                episode_success = episode_success or self.env._check_success()
                
            if terminations or truncations:
                num_success += int(episode_success)
                self.obs, _ = self.env.reset()
                episodic_count += 1
                episode_success = False

        eval_rews, eval_lens, sucess = rewards/self.eval_episodes, lens/self.eval_episodes, num_success/self.eval_episodes

        # Record internally
        self.results[steps] = {"rewards": eval_rews, "lengths": eval_lens}
        
        # Check if the environment tracks success rates
        has_success = hasattr(self.env, "_check_success")
        success_val = sucess if has_success else "N/A"

        # Console Logging (Keep this so you can watch progress in the terminal!)
        if has_success:
            tqdm.write(f"[Eval] step={steps} | rew={eval_rews:.2f} | len={eval_lens:.1f} | acc={sucess:.1f}")  
        else:
            tqdm.write(f"[Eval] step={steps} | rew={eval_rews:.2f} | len={eval_lens:.1f}") 

        # CSV Logging
        
        file_exists = os.path.isfile(self.eval_csv_path)
        with open(self.eval_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                # Create a consistent layout for your columns
                writer.writerow(["step", "eval_rewards", "eval_lengths", "success_rate"])
            
            # Append the evaluation metrics
            writer.writerow([steps, eval_rews, eval_lens, success_val])

        return eval_rews, eval_lens, sucess
    
def construct_env(env_name, render_mode=None):
    import gymnasium as gym
    # import centipede
    import robosuite as suite
    from robosuite.wrappers import GymWrapper

    # env = Lite6LiftEnv(
    #         show_viewer=False,
    #         ee_link_name="link6",
    #         max_steps=1000,
    #         render_mode=None,
    #         reward_shaping=True,
    #     )
    # return env
    
    
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assert/")

    if env_name == '2leg_cheetah':
        envs = gym.make('HalfCheetah-v5', render_mode=render_mode)
    elif env_name == '3leg_cheetah':
        envs = gym.make('HalfCheetah-v5', xml_file=file_path + "3leg_cheetah.xml", render_mode=render_mode)
    elif env_name == '4leg_cheetah':
        envs = gym.make('HalfCheetah-v5', xml_file=file_path + "4leg_cheetah.xml", render_mode=render_mode)
    elif env_name == '6leg_ant':
        envs = gym.make('Ant-v5', xml_file=file_path + "6leg_ant.xml",healthy_z_range=(0.2, 1.2), render_mode=render_mode)
    elif env_name == '5leg_ant':
        envs = gym.make('Ant-v5', xml_file=file_path + "5leg_ant.xml",healthy_z_range=(0.2, 1.2), render_mode=render_mode)
    elif env_name == '4leg_ant':
        envs = gym.make('Ant-v5', render_mode=render_mode)
    elif env_name == '3limb_swimmer':
        envs = gym.make('Swimmer-v5', render_mode=render_mode)
    elif env_name == '4limb_swimmer':
        envs = gym.make('Swimmer-v5', xml_file=file_path + "4limb_swimmer.xml", render_mode=render_mode)
    elif env_name == '5limb_swimmer':
        envs = gym.make('Swimmer-v5', xml_file=file_path + "5limb_swimmer.xml", render_mode=render_mode)
    elif env_name == 'CentipedeFour':
        envs = gym.make('CentipedeFour-v1', render_mode=render_mode)
    elif env_name == 'CentipedeSix':
        envs = gym.make('CentipedeSix-v1', render_mode=render_mode)
    elif env_name == 'CentipedeEight':
        envs = gym.make('CentipedeEight-v1', render_mode=render_mode)
    else:
        task, robot = env_name.split("_")
        if task == "TwoArmLift":
            env = GymWrapper(
                suite.make(
                    "TwoArmLift",
                    robots=[robot, robot],
                    env_configuration="opposed",  # two-arm 常用；兩手面對面
                    reward_shaping=True,
                    use_camera_obs=False,
                    use_object_obs=True,
                    has_offscreen_renderer=False,
                    has_renderer=False,
                    initialization_noise=None,
                )
            )
        else:
            env = GymWrapper(
                suite.make(
                    task,
                    robots=robot,
                    reward_shaping=True,
                    use_camera_obs=False,
                    has_offscreen_renderer=False,
                    has_renderer=False,
                    initialization_noise=None,
                )
            )
        
        if task == 'Wipe':
            import types
            print("Rewrite _check_success for Wiping task.")
            def _check_success(self):
                return len(self.wiped_markers)/self.num_markers
            env._check_success = types.MethodType(_check_success, env)
        return env

    return envs

