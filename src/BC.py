"""
Behavior Cloning to test ST architecture
"""
import numpy as np
import pickle
import torch
import time
import os
import sys
from stable_baselines3 import PPO

# Import PADRE's existing functions
from evaluate_MANETAgents import AgentResolver, EvalParams, evaluate  # ← Add evaluate here
import MANET.Environments as envs
import Utils.training_utils as ut
import MANET.ObservationWrappers as ow
import MANET.RewardWrappers as rw
import MANET.Agents as ag  # ← Add this
import MANET.ConditionStrategy as cs  # ← Add this

from imitation.data import types
from imitation.algorithms import bc
from stable_baselines3 import PPO
from architectures.ST import ST

def collect_demonstrations(expert_path, n_episodes=50):
    """Collect (obs, action) pairs from expert"""
    
    print(f"\n=== Collecting Demonstrations ===")
    print(f"Expert: {expert_path}")
    print(f"Episodes: {n_episodes}\n")
    
    env_params = {
        'numbOfNodes': 4,
        'numbOfJammers': (0, 5),
        'numbOfUsers': (2, 21),
        'seed': 42,
        'ep_length': 13000,
        'obs_type': 'VS_COW',
    }
    
    # Use PADRE's actual wrapping function!
    env = ut.createAndWrapEnv(
        env_constructor=envs.Static_MANETEnv,
        render_mode=None,
        numbOfNodes=4,
        numbOfJammers=1,
        numbOfUsers=(2, 21),
        seed=42,
        obs_wrapper=ow.VS_COW,
        obs_config=None,
        rew_wrapper=rw.LinearReward,  # ← CLASS not string!
        ep_length=13000,
        obs_type='full',
    )
    
    # Load expert
    eval_params = EvalParams.from_dict(env_params)
    resolver = AgentResolver([expert_path], eval_params)
    expert = resolver.load_agent(expert_path, env)


    device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expert.policy = expert.policy.to(device)
    expert.device = device
    print(f"✅ Expert loaded on: {device}\n")
    
    demonstrations = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 13000:
            t1 = time.time()
            action, _ = expert.predict(obs, deterministic=True)
            t2 = time.time()
            obs, reward, terminated, truncated, _ = env.step(action)
            t3 = time.time()
    
            if steps < 5:
                print(f"  step {steps}: predict={t2-t1:.4f}s, env.step={t3-t2:.4f}s")

            demonstrations.append((obs.copy(), action.copy()))
            done = terminated or truncated
            steps += 1
        
        print(f"  Episode {ep + 1}/{n_episodes}, Steps this ep: {steps}, Total: {len(demonstrations)}")
    
    print(f"\n✅ Collected {len(demonstrations)} demonstrations\n")
    return demonstrations


def train_bc(demo_file="demonstrations.pkl", n_epochs=100):
    """Train ST to imitate expert demonstrations"""
    
    print(f"\n=== Training ST via Behavior Cloning ===")
    
    with open(demo_file, "rb") as f:
        demonstrations = pickle.load(f)
    
    print(f"Loaded {len(demonstrations)} demonstration pairs")
    
    obs_list = np.array([demo[0] for demo in demonstrations])
    acts_list = np.array([demo[1] for demo in demonstrations])
    acts_list = acts_list.reshape(len(acts_list), -1)
    
    print(f"Observation shape: {obs_list.shape}")
    print(f"Action shape: {acts_list.shape}")
    
    transitions = types.Transitions(
        obs=obs_list,
        acts=acts_list,
        infos=np.array([{}] * len(obs_list)),
        next_obs=np.roll(obs_list, -1, axis=0),
        dones=np.zeros(len(obs_list), dtype=bool),
    )
    
    env = ut.createAndWrapEnv(
        env_constructor=envs.Static_MANETEnv,
        render_mode=None,
        numbOfNodes=4,
        numbOfJammers=1,
        numbOfUsers=(2, 21),
        seed=42,
        obs_wrapper=ow.VS_COW,
        obs_config=None,
        rew_wrapper=rw.LinearReward,
        ep_length=13000,
        obs_type='full',
    )
    
    device = 'cuda'
    st_policy = ST(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 3e-4,
        num_nodes=4,
    )
    st_policy = st_policy.to(device)
    
    rng = np.random.default_rng(42)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=st_policy,
        rng=rng,
        device=device,
        batch_size=512,
        optimizer_kwargs={"lr": 1e-4},
    )
    
    save_dir = "../out/BC_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # Track best loss for saving
    best_loss = float('inf')
    epoch_losses = []
    
    def on_epoch_end():
        nonlocal best_loss
        epoch = len(epoch_losses)
        
        # Get current loss from trainer's logger
        current_loss = bc_trainer.logger.default_logger.name_to_value.get("bc/loss", None)
        epoch_losses.append(current_loss)
        
        # Loading bar
        percent = 100 * ((epoch + 1) / n_epochs)
        filled = int(50 * (epoch + 1) // n_epochs)
        bar = '█' * filled + '-' * (50 - filled)
        loss_str = f"Loss: {current_loss:.4f}" if current_loss else "Loss: ???"
        print(f'\rTraining: |{bar}| {percent:.1f}% Epoch {epoch+1}/{n_epochs} - {loss_str}', end='', flush=True)
        
        # Save every epoch
        bc_trainer.policy.save(f"{save_dir}/ST_BC_epoch_{epoch+1}.zip")
        
        # Save best
        if current_loss is not None and current_loss < best_loss:
            best_loss = current_loss
            bc_trainer.policy.save(f"{save_dir}/ST_BC_best.zip")
    
    print(f"Training for {n_epochs} epochs...")
    bc_trainer.train(
        n_epochs=n_epochs,
        progress_bar=False,  # disable imitation's bar
        on_epoch_end=on_epoch_end,
        log_interval=99999,  # suppress imitation's logging
    )
    
    print(f"\n\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    bc_trainer.policy.save(f"{save_dir}/ST_BC_final.zip")
    print(f"Checkpoints saved in {save_dir}/")
    
    return bc_trainer






def evaluate_random_policy(n_episodes=3):
    """Evaluate random policy with visualization"""
    
    print(f"\n=== Random Policy Evaluation ===")
    print(f"Episodes: {n_episodes}\n")
    
    env = ut.createAndWrapEnv(
        env_constructor=envs.Static_MANETEnv,
        render_mode='plot',  # ← Visualization
        numbOfNodes=4,
        numbOfJammers=1,
        numbOfUsers=(2, 21),
        seed=42,
        obs_wrapper=ow.VS_COW,
        obs_config=None,
        rew_wrapper=rw.LinearReward,
        ep_length=1000,  # ← Short for recording
        obs_type='full',
    )
    
    print("Random policy created\n")
    
    all_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        
        while not done:
            action = env.action_space.sample()  # RANDOM!
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        
        all_rewards.append(ep_reward)
        print(f"  Episode {ep+1}/{n_episodes}: steps={steps}, reward={ep_reward:.2f}")
    
    mean_reward = np.mean(all_rewards)
    print(f"\n  Mean reward: {mean_reward:.2f}")
    print(f"\n✅ Evaluation complete!")
    return mean_reward




def evaluate_bc_model_direct(model_path="src/ST_BC_trained/rl_model_100000_steps.zip", n_episodes=3):
    """Evaluate BC-trained ST model directly, bypassing PADRE's eval system"""
    
    print(f"\n=== Direct BC Model Evaluation ===")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}\n")
    
    # Same env as BC training
    env = ut.createAndWrapEnv(
        env_constructor=envs.Static_MANETEnv,
        render_mode="plot",
        numbOfNodes=4,
        numbOfJammers=1,
        numbOfUsers=(2, 21),
        seed=42,
        obs_wrapper=ow.VS_COW,
        obs_config=None,
        rew_wrapper=rw.LinearReward,
        ep_length=500,
        obs_type='full',
    )
    
    # Load model
    
    model = PPO.load(model_path, env=env)
    print(f"Model loaded successfully\n")
    
    # Run episodes
    all_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        
        all_rewards.append(ep_reward)
        print(f"  Episode {ep+1}/{n_episodes}: steps={steps}, reward={ep_reward:.2f}")
    
    mean_reward = np.mean(all_rewards)
    print(f"\n  Mean reward: {mean_reward:.2f}")
    print(f"\n✅ Evaluation complete!")
    return mean_reward




def convert_bc_to_sb3(bc_model_path="ST_BC_trained/rl_model_100000_steps.zip", output_path="ST_BC_trained/rl_model_100000_steps_converted"):
    """Convert imitation library BC model to SB3 format"""
    
    print(f"\n=== Converting BC Model to SB3 Format ===")
    print(f"Input: {bc_model_path}")
    print(f"Output: {output_path}.zip\n")
    
    # Create environment
    env = ut.createAndWrapEnv(
        env_constructor=envs.Static_MANETEnv,
        render_mode=None,
        numbOfNodes=4,
        numbOfJammers=1,
        numbOfUsers=(2, 21),
        seed=42,
        obs_wrapper=ow.VS_COW,
        obs_config=None,
        rew_wrapper=rw.LinearReward,
        ep_length=13000,
        obs_type='full',
    )
    
    # Load the PyTorch policy weights
    print("Loading BC weights...")
    checkpoint = torch.load(bc_model_path, map_location='cpu')
    
    # Extract the actual state dict
    if 'state_dict' in checkpoint:
        policy_weights = checkpoint['state_dict']
    else:
        policy_weights = checkpoint
    
    print(f"Keys in checkpoint: {list(checkpoint.keys())}")
    print(f"Keys in state_dict: {list(policy_weights.keys())[:5]}...")  # Show first 5
    
    # Create a fresh SB3 model with ST
    print("Creating SB3 PPO model with ST policy...")
    model = PPO(
        policy=ST,
        env=env,
        policy_kwargs={"num_nodes": 4},
    )
    
    # Load the BC-trained weights into the policy
    print("Loading weights into policy...")
    model.policy.load_state_dict(policy_weights, strict=False)  # ← Added strict=False
    
    # Save in SB3 format
    print("Saving in SB3 format...")
    model.save(output_path)
    
    print(f"\n✅ Conversion complete! Saved to {output_path}.zip")
    return output_path + ".zip"




def evaluate_greedy_baseline(n_episodes=3):
    """Evaluate Greedy baseline"""
    
    print(f"\n=== Greedy Baseline Evaluation ===")
    print(f"Episodes: {n_episodes}\n")
    
    # Same env setup
    env = ut.createAndWrapEnv(
        env_constructor=envs.Static_MANETEnv,
        render_mode=None,
        numbOfNodes=4,
        numbOfJammers=1,
        numbOfUsers=(2, 21),
        seed=42,
        obs_wrapper=ow.VS_COW,
        obs_config=None,
        rew_wrapper=rw.LinearReward,
        ep_length=1000,
        obs_type='full',
    )
    
    # Create greedy agent
    greedy_agent = ag.GreedyMaximizer(env=env.unwrapped)
    print(f"Greedy agent created\n")
    
    # Run episodes
    all_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_reward = 0
        
        while not done:
            action = greedy_agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        
        all_rewards.append(ep_reward)
        print(f"  Episode {ep+1}/{n_episodes}: steps={steps}, reward={ep_reward:.2f}")
    
    mean_reward = np.mean(all_rewards)
    print(f"\n  Mean reward: {mean_reward:.2f}")
    print(f"\n✅ Evaluation complete!")
    return mean_reward

def printLoadingBar(current, total, loss=None, prefix='Progress', length=50):
    """
    Print a loading bar with optional loss display
    
    Args:
        current: Current iteration (0-indexed or 1-indexed, doesn't matter)
        total: Total iterations
        loss: Optional loss value to display
        prefix: Text before the bar
        length: Length of the bar in characters
    """
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    
    if loss is not None:
        print(f'\r{prefix}: |{bar}| {percent:.1f}% - Loss: {loss:.4f}', end='', flush=True)
    else:
        print(f'\r{prefix}: |{bar}| {percent:.1f}%', end='', flush=True)
    
    # Print newline on completion
    if current == total:
        print()



def test_reshape():
    """Test if reshape breaks action structure"""
    print("\n=== Testing Action Reshape ===\n")
    
    # Sample action like what expert produces
    demo_action = np.array([[0.5, 0.2], [0.1, -0.3], [0.8, 0.1], [-0.2, 0.4]])  # (4, 2)
    
    print(f"Original action (4, 2):")
    print(demo_action)
    print(f"Shape: {demo_action.shape}\n")
    
    # What BC training does
    flat = demo_action.reshape(-1)  # (8,)
    print(f"After flatten (8,):")
    print(flat)
    print(f"Shape: {flat.shape}\n")
    
    # What happens when we reshape back
    back = flat.reshape(4, 2)  # (4, 2)
    print(f"Reshaped back (4, 2):")
    print(back)
    print(f"Shape: {back.shape}\n")
    
    # Are they identical?
    identical = np.array_equal(demo_action, back)
    print(f"Identical to original: {identical}")
    
    if identical:
        print("\n✅ Reshape is reversible.")
        print("The bug is NOT the reshape.")
    else:
        print("\n❌ Reshape changed the action structure!")
        print("The bug IS the reshape.")





if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cpu":
        raise SystemError("GPU device not found. im not training that shit on CPU")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("existing demos found. No new data will be generated")
            train_bc("demonstrations.pkl", n_epochs=100)
        elif sys.argv[1] == "eval":
            print("evaluating BC model with visualization")
            evaluate_bc_model_direct("src/ST_BC_trained/rl_model_100000_steps.zip", n_episodes=30)
        elif sys.argv[1] == "eval_random":
            print("evaluating random policy with visualization")
            evaluate_random_policy(n_episodes=3)
        elif sys.argv[1] == "convert":
            print("converting BC model to SB3 format")
            convert_bc_to_sb3()
        elif sys.argv[1] == "eval_greedy":
            print("evaluating greedy baseline")
            evaluate_greedy_baseline(n_episodes=10)
        elif sys.argv[1] == "test_reshape":
            test_reshape()
    else:
        print("no data found. generating new demos")
        expert_path = "../out/best_agents/27_04_filtered-PPO_4_[0,5]_[2,21]_13k_ultra_final_Sttc_MANE_True/rl_model_10000000_steps.zip"
        demos = collect_demonstrations(expert_path, n_episodes=50)
        
        with open("demonstrations.pkl", "wb") as f:
            pickle.dump(demos, f)
        
        print("✅ Saved to demonstrations.pkl")
        print(f"Total pairs: {len(demos)}")


