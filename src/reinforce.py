import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from collections import deque
import random

class DJEnvironment:
    """Environment for mixing two songs together - ALWAYS keeps both songs playing"""
    
    def __init__(self, song_a_dir, song_b_dir, sample_rate=44100, window_size=4096):
        self.sample_rate = sample_rate
        self.window_size = window_size  # Process in larger chunks
        
        # Load stems from both songs
        self.song_a_stems = self._load_stems(song_a_dir)
        self.song_b_stems = self._load_stems(song_b_dir)
        
        # Get minimum length across all stems
        min_len = min(
            min(len(stem) for stem in self.song_a_stems.values()),
            min(len(stem) for stem in self.song_b_stems.values())
        )
        
        # Trim all stems to same length
        for key in self.song_a_stems:
            self.song_a_stems[key] = self.song_a_stems[key][:min_len]
        for key in self.song_b_stems:
            self.song_b_stems[key] = self.song_b_stems[key][:min_len]
        
        self.song_length = min_len
        self.num_windows = self.song_length // self.window_size
        
        # Store which stems are available for feature extraction
        # Use union of both songs' stems for consistent feature size
        self.all_stem_names = sorted(set(list(self.song_a_stems.keys()) + list(self.song_b_stems.keys())))
        print(f"Using stems for features: {self.all_stem_names}")
        
        # State: current position in songs
        self.current_window = 0
        
        # Action space: continuous control of mix parameters
        # Instead of discrete play/stop, we control:
        # 0: volume A (0-1)
        # 1: volume B (0-1)  
        # 2: tempo offset A (-0.1 to +0.1)
        # 3: tempo offset B (-0.1 to +0.1)
        # 4: crossfade position (-1 to +1, -1=all A, +1=all B)
        self.action_space = 5
        
        # Output buffer
        self.mix_buffer = []
        
        # Mix history for reward calculation
        self.volume_history = []
        
    def _load_stems(self, directory):
        """Load available stems from directory (vocal, bass, drums, other)"""
        stems = {}
        stem_names = ['vocals', 'bass', 'drums', 'other']
        
        for stem_name in stem_names:
            for ext in ['.wav', '.mp3', '.flac']:
                stem_path = Path(directory) / f"{stem_name}{ext}"
                if stem_path.exists():
                    audio, sr = librosa.load(stem_path, sr=self.sample_rate, mono=True)
                    stems[stem_name] = audio
                    print(f"Loaded {stem_name} from {directory}")
                    break
        
        if not stems:
            raise FileNotFoundError(f"Could not find any stems in {directory}")
        
        print(f"Found {len(stems)} stems in {directory}: {list(stems.keys())}")
        return stems
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_window = 0
        self.mix_buffer = []
        self.volume_history = []
        return self._get_state()
    
    def _get_state(self):
        """Get current state representation"""
        # Calculate state size: 2 features per stem type * number of stems * 2 songs + 2 progress values
        state_size = len(self.all_stem_names) * 2 * 2 + 2
        
        if self.current_window >= self.num_windows:
            # Return zero state with correct size when done
            return np.zeros(state_size, dtype=np.float32)
        
        # Get audio features from current window of both songs
        start_idx = self.current_window * self.window_size
        end_idx = start_idx + self.window_size
        
        features_a = self._extract_features(self.song_a_stems, start_idx, end_idx)
        features_b = self._extract_features(self.song_b_stems, start_idx, end_idx)
        
        # State: features from both songs + progress
        state = np.concatenate([
            features_a,
            features_b,
            [self.current_window / self.num_windows],  # Progress through song
            [len(self.volume_history) / self.num_windows if self.volume_history else 0]
        ]).astype(np.float32)
        
        return state
    
    def _extract_features(self, stems, start, end):
        """Extract audio features from available stems"""
        features = []
        
        # Use the common stem names across both songs for consistency
        for stem_name in self.all_stem_names:
            if stem_name in stems:
                stem = stems[stem_name]
                segment = stem[start:end]
                
                # RMS energy
                rms = np.sqrt(np.mean(segment**2)) if len(segment) > 0 else 0.0
                
                # Spectral centroid (brightness)
                if len(segment) > 0:
                    spec_centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sample_rate)[0, 0]
                else:
                    spec_centroid = 0.0
                
                features.extend([float(rms), float(spec_centroid / 10000.0)])
            else:
                # Stem not available, use zero features
                features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def step(self, action):
        """
        Execute action and return next state, reward, done
        Action is continuous: [vol_a, vol_b, tempo_a, tempo_b, crossfade]
        """
        if self.current_window >= self.num_windows:
            return self._get_state(), 0.0, True
        
        # Parse continuous action (normalized to appropriate ranges)
        vol_a = float(np.clip(action[0], 0, 1))
        vol_b = float(np.clip(action[1], 0, 1))
        tempo_a = float(np.clip(action[2], -0.1, 0.1))
        tempo_b = float(np.clip(action[3], -0.1, 0.1))
        crossfade = float(np.clip(action[4], -1, 1))
        
        # Apply crossfade to volumes
        if crossfade < 0:  # Favor A
            vol_a_adjusted = vol_a
            vol_b_adjusted = vol_b * (1 + crossfade)
        else:  # Favor B
            vol_a_adjusted = vol_a * (1 - crossfade)
            vol_b_adjusted = vol_b
        
        # Get current window
        start_idx = self.current_window * self.window_size
        end_idx = start_idx + self.window_size
        
        # Mix audio with tempo adjustments
        mixed_chunk = self._mix_audio_window(
            start_idx, end_idx, 
            vol_a_adjusted, vol_b_adjusted,
            tempo_a, tempo_b
        )
        
        self.mix_buffer.extend(mixed_chunk)
        self.volume_history.append((vol_a_adjusted, vol_b_adjusted))
        
        # Calculate reward
        reward = float(self._calculate_reward(mixed_chunk, vol_a_adjusted, vol_b_adjusted))
        
        # Move to next window
        self.current_window += 1
        done = self.current_window >= self.num_windows
        
        return self._get_state(), reward, done
    
    def _mix_audio_window(self, start, end, vol_a, vol_b, tempo_a, tempo_b):
        """Mix audio window with volume and tempo control"""
        mixed = np.zeros(self.window_size)
        
        # Mix song A stems with tempo adjustment
        if vol_a > 0.01:
            for stem_name, stem in self.song_a_stems.items():
                segment = stem[start:end]
                if abs(tempo_a) > 0.001:
                    # Apply time stretching
                    segment = librosa.effects.time_stretch(segment, rate=1.0 + tempo_a)
                    segment = segment[:self.window_size]  # Trim to window size
                    if len(segment) < self.window_size:
                        segment = np.pad(segment, (0, self.window_size - len(segment)))
                mixed[:len(segment)] += segment * vol_a
        
        # Mix song B stems with tempo adjustment  
        if vol_b > 0.01:
            for stem_name, stem in self.song_b_stems.items():
                segment = stem[start:end]
                if abs(tempo_b) > 0.001:
                    segment = librosa.effects.time_stretch(segment, rate=1.0 + tempo_b)
                    segment = segment[:self.window_size]
                    if len(segment) < self.window_size:
                        segment = np.pad(segment, (0, self.window_size - len(segment)))
                mixed[:len(segment)] += segment * vol_b
        
        # Normalize to prevent clipping
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak
        
        return mixed
    
    def _calculate_reward(self, mixed_chunk, vol_a, vol_b):
        """Calculate reward - encourage both songs playing at good levels"""
        
        # 1. STRONG penalty for either volume being too low (not mixing)
        min_vol = min(vol_a, vol_b)
        if min_vol < 0.3:
            volume_penalty = -10.0 * (0.3 - min_vol)
        else:
            volume_penalty = 0
        
        # 2. Reward for balanced mixing
        vol_balance = 1.0 - abs(vol_a - vol_b)
        balance_reward = vol_balance * 5.0
        
        # 3. Reward for both volumes being high
        both_high_reward = min(vol_a, vol_b) * 5.0
        
        # 4. Check audio quality
        rms = np.sqrt(np.mean(mixed_chunk**2))
        peak = np.max(np.abs(mixed_chunk))
        
        # Reward good energy
        energy_reward = np.clip(rms * 3, 0, 2)
        
        # Punish clipping
        clipping_penalty = -5.0 if peak > 0.99 else 0
        
        # 5. Slight penalty for silence
        silence_penalty = -3.0 if rms < 0.01 else 0
        
        return (balance_reward + both_high_reward + energy_reward + 
                clipping_penalty + silence_penalty + volume_penalty)
    
    def save_mix(self, filename='output_mix.wav'):
        """Save the mixed audio to file"""
        if len(self.mix_buffer) > 0:
            mix_array = np.array(self.mix_buffer)
            sf.write(filename, mix_array, self.sample_rate)
            print(f"Mix saved to {filename} ({len(mix_array)/self.sample_rate:.2f} seconds)")


class MixingNetwork(nn.Module):
    """Actor network for continuous mixing control"""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super(MixingNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        
        # Separate heads for different action types
        self.volume_head = nn.Linear(hidden_size // 2, 2)  # vol_a, vol_b
        self.tempo_head = nn.Linear(hidden_size // 2, 2)   # tempo_a, tempo_b
        self.crossfade_head = nn.Linear(hidden_size // 2, 1)  # crossfade
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Volume outputs (0-1) - use sigmoid
        volumes = torch.sigmoid(self.volume_head(x))
        
        # Tempo outputs (-0.1 to +0.1) - use tanh and scale
        tempos = torch.tanh(self.tempo_head(x)) * 0.1
        
        # Crossfade (-1 to +1) - use tanh
        crossfade = torch.tanh(self.crossfade_head(x))
        
        return torch.cat([volumes, tempos, crossfade], dim=1)


class CriticNetwork(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays with explicit dtype
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """DDPG Agent for continuous control of mixing"""
    
    def __init__(self, state_size, action_size, lr_actor=0.0001, lr_critic=0.001, 
                 gamma=0.99, tau=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor networks
        self.actor = MixingNetwork(state_size, action_size).to(self.device)
        self.actor_target = MixingNetwork(state_size, action_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic networks
        self.critic = CriticNetwork(state_size, action_size).to(self.device)
        self.critic_target = CriticNetwork(state_size, action_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.replay_buffer = ReplayBuffer()
        
        # Exploration noise
        self.noise_scale = 0.1
        self.noise_decay = 0.9995
        
    def select_action(self, state, add_noise=True):
        """Select action using actor network"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]
            
            if add_noise:
                # Add exploration noise
                noise = np.random.normal(0, self.noise_scale, size=action.shape)
                action = action + noise
                
                # Clip to valid ranges
                action[0:2] = np.clip(action[0:2], 0, 1)  # Volumes
                action[2:4] = np.clip(action[2:4], -0.1, 0.1)  # Tempos
                action[4] = np.clip(action[4], -1, 1)  # Crossfade
            
            return action
    
    def train_step(self, batch_size=256):
        """Perform one training step"""
        if len(self.replay_buffer) < batch_size:
            return 0, 0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, source, target):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def decay_noise(self):
        """Decay exploration noise"""
        self.noise_scale = max(0.01, self.noise_scale * self.noise_decay)


def train_dj_agent(song_a_dir, song_b_dir, episodes=300):
    """Train the DJ agent to mashup two songs"""
    
    # Initialize environment
    env = DJEnvironment(song_a_dir, song_b_dir)
    
    # Get state size from initial state
    initial_state = env.reset()
    state_size = len(initial_state)
    action_size = env.action_space
    
    # Initialize agent
    agent = DDPGAgent(state_size, action_size)
    
    print(f"Training DJ agent with DDPG for {episodes} episodes...")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Song length: {env.song_length / env.sample_rate:.2f} seconds")
    print(f"Number of windows: {env.num_windows}")
    
    best_reward = float('-inf')
    reward_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        actor_losses = []
        critic_losses = []
        
        for step in range(env.num_windows):
            # Select and perform action
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done = env.step(action)
            
            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Train
            actor_loss, critic_loss = agent.train_step(batch_size=256)
            if actor_loss != 0:
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.decay_noise()
        reward_history.append(total_reward)
        
        if total_reward > best_reward:
            best_reward = total_reward
        
        if episode % 10 == 0:
            avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else total_reward
            avg_actor_loss = np.mean(actor_losses) if actor_losses else 0
            avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
            
            # Analyze volume usage
            if env.volume_history:
                vols_a = [v[0] for v in env.volume_history]
                vols_b = [v[1] for v in env.volume_history]
                avg_vol_a = np.mean(vols_a)
                avg_vol_b = np.mean(vols_b)
                both_playing = sum(1 for va, vb in env.volume_history if va > 0.3 and vb > 0.3)
                both_pct = (both_playing / len(env.volume_history)) * 100
                
                print(f"Ep {episode}/{episodes} | Reward: {total_reward:.1f} | Avg: {avg_reward:.1f} | "
                      f"Vol A: {avg_vol_a:.2f} | Vol B: {avg_vol_b:.2f} | Both>0.3: {both_pct:.1f}% | "
                      f"Noise: {agent.noise_scale:.3f}")
    
    # Generate final mix with trained policy (no noise)
    print("\nGenerating final mix with trained policy...")
    state = env.reset()
    
    vol_a_list = []
    vol_b_list = []
    
    for step in range(env.num_windows):
        action = agent.select_action(state, add_noise=False)
        vol_a_list.append(action[0])
        vol_b_list.append(action[1])
        next_state, reward, done = env.step(action)
        state = next_state
        if done:
            break
    
    print(f"\nFinal mix statistics:")
    print(f"Mix duration: {len(env.mix_buffer) / env.sample_rate:.2f} seconds")
    print(f"Original song duration: {env.song_length / env.sample_rate:.2f} seconds")
    print(f"Average volume A: {np.mean(vol_a_list):.3f}")
    print(f"Average volume B: {np.mean(vol_b_list):.3f}")
    both_active = sum(1 for va, vb in zip(vol_a_list, vol_b_list) if va > 0.3 and vb > 0.3)
    print(f"Both songs active (>0.3): {(both_active/len(vol_a_list))*100:.1f}% of the time")
    
    env.save_mix('trained_mashup.wav')
    
    return agent, env


# Example usage
if __name__ == "__main__":
    # Specify directories containing stems
    song_a_directory = "/Users/abcheng/Documents/workspace/mashup_eval/data/auto_preprocess_smol/separated/htdemucs/Bam Bam - Hi-Q"
    song_b_directory = "/Users/abcheng/Documents/workspace/mashup_eval/data/auto_preprocess_smol/separated/htdemucs/Benji Cossa - New Flowers (Fast 4-track Version)"
    
    # Train agent
    agent, env = train_dj_agent(
        song_a_directory, 
        song_b_directory,
        episodes=300
    )
    
    print("Training complete!")