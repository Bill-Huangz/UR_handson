import time
import numpy as np
import torch
from utils import seed_everything, evaluate_management, construct_env
from copy import deepcopy
import torch.nn as nn
from tqdm import tqdm
import argparse
import yaml

from SAC import SAC
from QAvatar_SAC import QAvatar_SAC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alg", type=str, default="PPO")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    with open(f"config/{args.alg}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    for k, v in config.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    exp_name = f"{args.env}/{args.alg}/{args.seed}"
    log_path = f"{args.tar_folder}/log/{exp_name}"
    
    
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = construct_env(args.env)
    eval_env = evaluate_management(construct_env(args.env), device=device, reset_seed=args.seed, eval_episodes=5, log_path=log_path)

    if args.alg == 'SAC':
        agent = SAC(args, envs.observation_space.shape[0], envs.action_space.shape[0], envs.action_space.low, envs.action_space.high, device, log_path)
    elif args.alg == 'QAvatar_SAC':
        src_envs = [construct_env(env_id) for env_id in args.src_env]
        src_obs = [src_env.observation_space.shape[0] for src_env in src_envs]
        src_act = [src_env.action_space.shape[0] for src_env in src_envs]
        act_low = [envs.action_space.low] + [src_env.action_space.low for src_env in src_envs]
        act_high = [envs.action_space.high] + [src_env.action_space.high for src_env in src_envs]
        agent = QAvatar_SAC(args, src_obs, src_act, envs.observation_space.shape[0], envs.action_space.shape[0], act_low, act_high, device, log_path)
    else:
        raise ValueError("Not Implementation")
    
    print(args.alg)

    if agent.interact == "ON_POLICY":
        agent.start_time = time.time()
        obs, _ = envs.reset(seed=args.seed)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        num_iterations = args.total_timesteps // args.num_steps

        best_sucess = - 1.0
        for iteration in tqdm(range(1, num_iterations + 1)):
            # Annealing the rate if instructed to do so.
            agent.anneal_anything(iteration, num_iterations)

            for step in range(args.num_steps):
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs)
                next_obs, reward, terminations, truncations, _ = envs.step(np.clip(action[0].cpu().numpy(), envs.action_space.low, envs.action_space.high))

                done = terminations or truncations
                if args.env_reward_norm_value is not None:
                    reward = agent.norm_rew.normalize(reward, done)

                next_obs_value = agent.get_value(torch.as_tensor(next_obs, dtype=torch.float32, device=device))

                agent.store_data(step, obs, action, next_obs, reward, done, value, next_obs_value, logprob)

                if done:
                    next_obs, _ = envs.reset()

                obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

            agent.train(args.num_steps*iteration)
            if args.num_steps*iteration > int(4e5) and  iteration % int(1e4) == 0: 
                eval_rews, eval_lens, sucess = eval_env.evaluate(agent, args.num_steps*iteration)
                if best_sucess > sucess:
                    best_sucess = sucess
                    if best_sucess > 0.4:
                        agent.save(f"{log_path}/model.pt")
                        break
        envs.close()
        
        for step, data in eval_env.results.items():
            print(f"{step}: rewards={data['rewards']}, lengths={data['lengths']}")        
    elif agent.interact == "OFF_POLICY":
        agent.start_time = time.time()
        best_sucess = 0.0

        obs, _ = envs.reset(seed=args.seed)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

        for step in range(1, args.learning_starts+1):
            actions = envs.action_space.sample()
            next_obs, rewards, terminations, truncations, _ = envs.step(actions)
            
            actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            
            agent.store_data(obs, next_obs, actions, rewards, terminations, truncations)
            if terminations or truncations:
                next_obs, _ = envs.reset()
                next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            obs = next_obs
            if step % args.evaluate_freq == 0:
                eval_rews, eval_lens, sucess = eval_env.evaluate(agent, step)
            
        for step in tqdm(range(args.learning_starts + 1, args.total_timesteps + 1)):
            with torch.no_grad():
                actions, _ = agent.get_action(obs.unsqueeze(0))

            next_obs, rewards, terminations, truncations, _ = envs.step(actions[0].cpu().numpy())
            
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            agent.store_data(obs, next_obs, actions, rewards, terminations, truncations)
            
            if terminations or truncations:
                next_obs, _ = envs.reset()
                next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    
            obs = next_obs
            agent.train(step, args.total_timesteps + 1)

            if step % args.evaluate_freq == 0:
                eval_rews, eval_lens, sucess = eval_env.evaluate(agent, step)
                agent.logger_losses(step)
            
        if args.save_model:
            agent.save(f"{log_path}/model.pt")
        envs.close()
    else:
        raise ValueError("Not Implementation")
    
