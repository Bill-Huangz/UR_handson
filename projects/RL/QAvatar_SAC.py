import os
import sys
import csv
sys.path.append(os.path.abspath("/opt/workspace/RL/flow_model"))
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from SAC import SAC
from core.flow.real_nvp import RealNvp
 
class decoder(nn.Module):
    def __init__(self, source_state_dim, source_action_dim, target_state_dim, target_action_dim, useTanh=True, hidden_size=256):
        super(decoder, self).__init__()
        self.state_emb = nn.Sequential(
            nn.Linear(target_state_dim, hidden_size//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2),
        )
        self.action_emb = nn.Sequential(
            nn.Linear(target_action_dim, hidden_size//2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2),
        )

        self.out_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, source_state_dim + source_action_dim),
        )
        self.useTanh = useTanh
        self.record_out = 0.0

    def forward(self, target_state, target_action):
        state_emb = self.state_emb(target_state.float())
        action_emb = self.action_emb(target_action.float())
        input = torch.cat((state_emb, action_emb), dim=1)
        output = self.out_layer(input)
        
        self.record_out = np.mean(abs(output.detach().cpu().numpy()))

        if self.useTanh:
            output = nn.Tanh()(output)
        return output

class QAvatar_SAC(SAC):
    def __init__(self, args, src_obs_dim, src_act_dim, tar_obs_dim, tar_act_dim, action_low, action_high, device, log_path=None):
        super().__init__(args, tar_obs_dim, tar_act_dim, action_low[0], action_high[0], device)  
        self.device = device
        self.num_src_domain = len(src_obs_dim) 
        self.decoders = nn.ModuleList([decoder(so, sa, tar_obs_dim, tar_act_dim) for so, sa in zip(src_obs_dim,src_act_dim)]).to(device)
        self.state_flow = nn.ModuleList([RealNvp.load_module(f"{args.src_model_path[idx]}flow_model/state/flow.pt").float().to(device) for idx in range(self.num_src_domain)]).to(device)
        self.state_flow_stds = [torch.tensor(np.load(f"{args.src_model_path[idx]}flow_model/state/src_info.npz")['std'],dtype=torch.float32).to(device) for idx in range(self.num_src_domain)]
        self.state_flow_means = [torch.tensor(np.load(f"{args.src_model_path[idx]}flow_model/state/src_info.npz")['mean'],dtype=torch.float32).to(device) for idx in range(self.num_src_domain)]
    
        self.action_flow = nn.ModuleList([RealNvp.load_module(f"{args.src_model_path[idx]}flow_model/action/flow.pt").float().to(device) for idx in range(self.num_src_domain)]).to(device)

        for module in self.state_flow:
            for param in module.parameters():
                param.requires_grad = False
        for module in self.action_flow:
            for param in module.parameters():
                param.requires_grad = False

        self.src_obs_dim = src_obs_dim
        
        self.source_models = [SAC(args, obs_dim, act_dim, act_low, act_high, device) \
                                for obs_dim, act_dim, act_low, act_high in zip(src_obs_dim, src_act_dim, action_low[1:], action_high[1:])]
        
        for idx in range(self.num_src_domain):
            self.source_models[idx].load(f"{args.src_model_path[idx]}model.pt")
        
        self.decoder_optimizer = torch.optim.AdamW(self.decoders.parameters(), lr = args.decoder_lr)
        
        self.weight = [0.0 for _ in range(self.num_src_domain)]
        self.source_critic_losses = []

        self.decoder_outs = []
        self.weights_record = []
        for _ in range(self.num_src_domain):
            self.decoder_outs.append([])
            self.weights_record.append([])

        if log_path is not None:
            self.csv_path = f"{log_path}/train.csv"
            dir_path = os.path.dirname(self.csv_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            if os.path.exists(self.csv_path):
                os.remove(self.csv_path)
    
    def evaluate_weight(self): 
        if self.buffer.pos < int(1e5):
            index = np.arange(self.buffer.pos)
        else:
            index = np.random.choice(self.buffer.pos, size=int(1e5), replace=False)
        
        data = self.buffer._get_samples(index)
        trans_losses = []
        with torch.no_grad():
            next_state_actions, _ = self.get_action(data.next_observations)

            q_values = torch.min(*self.Qnetworks(data.observations, data.actions))
            target_q_value = torch.min(*self.Qnetworks(data.next_observations, next_state_actions))
            target_q_value = data.rewards + (1 - data.terminations) * self.gamma * target_q_value

            trans_losses.append(np.mean(torch.clamp(abs(q_values - target_q_value), 1e-8, 1e8).cpu().numpy()))

            for idx in range(self.num_src_domain):
                latent_src_sa = self.decoders[idx](data.observations, data.actions)
                src_s = (self.state_flow[idx].g(latent_src_sa[:, :self.src_obs_dim[idx]])[0]*self.state_flow_stds[idx] + self.state_flow_means[idx])
                src_a = self.action_flow[idx].g(latent_src_sa[:, self.src_obs_dim[idx]:])[0]
                src_a = src_a + (torch.clamp(src_a, -1.0, 1.0) - src_a).detach()
                src_q_value = torch.min(*self.source_models[idx].Qnetworks(src_s, src_a))

                latent_src_sa_ = self.decoders[idx](data.next_observations, next_state_actions)
                src_s_ = (self.state_flow[idx].g(latent_src_sa_[:, :self.src_obs_dim[idx]])[0]*self.state_flow_stds[idx] + self.state_flow_means[idx])
                src_a_ = self.action_flow[idx].g(latent_src_sa_[:, self.src_obs_dim[idx]:])[0]
                src_a_ = src_a_ + (torch.clamp(src_a_, -1.0, 1.0) - src_a_).detach()

                src_target_q_value = torch.min(*self.source_models[idx].Qnetworks(src_s_, src_a_))
                src_target_q_value = data.rewards + (1 - data.terminations) * self.gamma * src_target_q_value

                trans_losses.append(np.mean(torch.clamp(abs(src_q_value - src_target_q_value), 1e-8, 1e8).cpu().numpy()))

            trans_losses = np.array(trans_losses, dtype=np.float32)
            self.weight = ((1 / trans_losses) / np.sum(1 / trans_losses))[1:]

    # re-write the training
    def train(self, step, total_timesteps):
        #------------------ Compute alpha ------------------#
        if step % 1000 == 0:
            self.evaluate_weight()

        data = self.buffer.sample(self.train_batch_size)
        data_enc = self.buffer.sample(self.train_batch_size, enc_batch=True)
        #------------------ Update decoders ------------------# 
        source_critic_loss = 0.0
        for idx in range(self.num_src_domain):
            latent_src_sa = self.decoders[idx](data_enc.observations, data_enc.actions)
            src_s = (self.state_flow[idx].g(latent_src_sa[:, :self.src_obs_dim[idx]])[0]*self.state_flow_stds[idx] + self.state_flow_means[idx])
            src_a = self.action_flow[idx].g(latent_src_sa[:, self.src_obs_dim[idx]:])[0]
            src_a = src_a + (torch.clamp(src_a, -1.0, 1.0) - src_a).detach()

            source_qf1_values, source_qf2_values = self.source_models[idx].Qnetworks(src_s, src_a)

            with torch.no_grad():
                next_state_actions, next_state_log_pi = self.get_action(data_enc.next_observations)
                latent_src_next_sa = self.decoders[idx](data_enc.next_observations, next_state_actions)
                src_s_ = (self.state_flow[idx].g(latent_src_next_sa[:, :self.src_obs_dim[idx]])[0]*self.state_flow_stds[idx] + self.state_flow_means[idx])
                src_a_ = self.action_flow[idx].g(latent_src_next_sa[:, self.src_obs_dim[idx]:])[0]
                src_a_ = src_a_ + (torch.clamp(src_a_, -1.0, 1.0) - src_a_).detach()

                source_next_q_value = torch.min(*self.source_models[idx].Qnetworks(src_s_, src_a_))
                target_source_q_values = data_enc.rewards + (1 - data_enc.terminations) * self.gamma * source_next_q_value

            source_critic_loss += F.mse_loss(source_qf1_values, target_source_q_values) + F.mse_loss(source_qf2_values, target_source_q_values)

        self.decoder_optimizer.zero_grad()
        source_critic_loss.backward()
        self.decoder_optimizer.step()
        self.source_critic_losses.append(source_critic_loss.item()/self.num_src_domain)

        with torch.no_grad():
            next_state_actions, next_state_log_pi = self.get_action(data.next_observations)
            qf1_next_target, qf2_next_target = self.target_Qnetworks(data.next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards + (1 - data.terminations) * self.gamma * min_qf_next_target

        qf1_values, qf2_a_values = self.Qnetworks(data.observations, data.actions)
        qf_loss = F.mse_loss(qf1_values, next_q_value) + F.mse_loss(qf2_a_values, next_q_value)
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
            
        pi, log_pi= self.get_action(data.observations)
        qf1_pi, qf2_pi = self.Qnetworks(data.observations, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        mix_Q = (1 - sum(self.weight)) * min_qf_pi
        for idx in range(self.num_src_domain):
            latent_src_spi = self.decoders[idx](data.observations, pi)
            src_s = (self.state_flow[idx].g(latent_src_spi[:, :self.src_obs_dim[idx]])[0]*self.state_flow_stds[idx] + self.state_flow_means[idx])
            src_pi = self.action_flow[idx].g(latent_src_spi[:, self.src_obs_dim[idx]:])[0]
            src_pi = src_pi + (torch.clamp(src_pi, -1.0, 1.0) - src_pi).detach()

            source_current_q_value = torch.min(*self.source_models[idx].Qnetworks(src_s, src_pi))
            mix_Q += self.weight[idx] * source_current_q_value

        actor_loss = ((self.alpha * log_pi) - mix_Q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.temperature_opt:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.temp_optimizer.zero_grad()
            alpha_loss.backward()
            self.temp_optimizer.step()

        # update the target networks
        with torch.no_grad():
            for param, target_param in zip(self.Qnetworks.parameters(), self.target_Qnetworks.parameters()):
                target_param.mul_(1 - self.tau)
                target_param.add_(self.tau * param)
        
        self.Q_losses.append(qf_loss.mean().item())
        self.actor_losses.append(actor_loss.mean().item())
        self.alphas.append(self.alpha.item())
        if self.temperature_opt:
            self.alpha_losses.append(alpha_loss.item())

        for idx in range(self.num_src_domain):
            self.decoder_outs[idx].append(self.decoders[idx].record_out)
            self.weights_record[idx].append(self.weight[idx])

    def logger_losses(self, step):  # Removed writer parameter
    # 1. Calculate the base means
        q_loss = np.mean(self.Q_losses)
        actor_loss = np.mean(self.actor_losses)
        src_critic_loss = np.mean(self.source_critic_losses)
        alpha = np.mean(self.alphas)
        alpha_loss = np.mean(self.alpha_losses) if self.temperature_opt else "N/A"

        # 2. Gather metrics for each source domain dynamically
        domain_data = []
        for idx in range(self.num_src_domain):
            domain_data.append(np.mean(self.decoder_outs[idx]))
            domain_data.append(np.mean(self.weights_record[idx]))

        # 3. Check if CSV file exists to determine if we need a header
        file_exists = os.path.isfile(self.csv_path)

        # 4. Write to CSV
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                # Build the dynamic header layout
                header = ["step", "Q_loss", "actor_loss", "source_critic_loss", "alpha"]
                for idx in range(self.num_src_domain):
                    header.append(f"decoder{idx}_outs")
                    header.append(f"weight_{idx}")
                header.append("alpha_loss")
                
                writer.writerow(header)
            
            # Assemble the data row matching the header order perfectly
            row = [step, q_loss, actor_loss, src_critic_loss, alpha] + domain_data + [alpha_loss]
            writer.writerow(row)

        # 5. Clear all memory buffers exactly as before
        for idx in range(self.num_src_domain):
            self.decoder_outs[idx].clear()
            self.weights_record[idx].clear()
        
        if self.temperature_opt:
            self.alpha_losses.clear()
            
        self.Q_losses.clear()
        self.actor_losses.clear()
        self.alphas.clear()
        self.source_critic_losses.clear()
