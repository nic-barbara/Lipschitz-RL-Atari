import cv2
import envpool
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import pickle
import random
import torch

from cycler import cycler
from pathlib import Path
from liprl.atari_agent import AtariAgent


def seed_everything(seed: int = 42, deterministic: bool = True):
    """Set random seeds and CUDA backend determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def make_env(env_id, seed=1, num_envs=1):
    """Create batched environments with envpool."""
    envs = envpool.make(
        env_id,
        env_type="gym",
        num_envs=num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=seed,
    )
    envs.num_envs = num_envs
    return envs


def save_params_config(save_dir: str, agent: AtariAgent, config: dict, metrics):
    """Save agent params and hyperparameter configs dictionary."""
    
    # Generate file name
    fname = f"{config['env_id']}_{config['network']}"
    if not config["network"] == "cnn":
        fname += f"_g{config['lipschitz']}"
    fname += f"_v{config['seed']}"
    config["fname"] = fname
    
    save_data = {
        "params": agent.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    torch.save(save_data, save_dir + fname + ".pt")


def load_agent(env, params, config):
    """Load a trained agent from its parameters."""
    
    # Agent template
    agent = AtariAgent(env.action_space.n, config["network"], config["lipschitz"])
    agent = agent.to(torch.device("cuda"))
    
    # Call a dummy batch to initialise all params
    obs, _ = env.reset()
    obs = torch.Tensor(obs).to("cuda")
    agent.get_action(obs)
    
    # Return loaded model
    agent.load_state_dict(params)
    agent.get_action(obs)
    agent.eval()
    return agent
    

def load_params_configs(load_path):
    """Load agent params and hyperparameter configs dictionary."""
    return torch.load(load_path)


def empirical_lipschitz(model, x, eps=0.05, verbose=False, keep_graph=False):
    """
    Ray's implementation from:
    https://github.com/acfr/LBDN/blob/main/utils.py#L84
    """

    norms = lambda X: X.view(X.shape[0], -1).norm(dim=1) ** 2
    gam = 0.0
    for r in range(10):
        dx = torch.zeros_like(x)
        dx.uniform_(-eps, eps)
        x.requires_grad = True
        dx.requires_grad = True
        optimizer = torch.optim.Adam([x, dx], lr=1E-2)
        iters, j = 0, 0
        LipMax = 0.0
        while j < 100:
            
            # Estimate Lipschitz bound fot this epoch
            LipMax_1 = LipMax
            optimizer.zero_grad()
            dy = model(x + dx) - model(x)
            Lip = norms(dy) / (norms(dx) + 1e-6)
            Obj = -Lip.sum()
            Obj.backward(retain_graph=keep_graph)
            optimizer.step()
            LipMax = Lip.max().item()
            
            # Update iterations and check for termination condition
            iters += 1
            j += 1
            if j >= 5:
                if LipMax < LipMax_1 + 1E-3:  
                    optimizer.param_groups[0]["lr"] /= 10.0
                    j = 0

                if optimizer.param_groups[0]["lr"] <= 1E-5:
                    break
        
        gam = max(gam, np.sqrt(LipMax))
        if verbose:
            print("iter: {}, lip: {:.3g}".format(r, gam))

    return gam 


def plot_reward_curve(config, metrics):
    
    # Extract data
    x = metrics["steps"]
    y = metrics["rewards"]
    y_err = metrics["stdev"]
    
    # Make a plot and make it pretty
    plt.plot(x, y)
    plt.fill_between(x, y - y_err, y + y_err, alpha=0.2)
    
    fname = f"../results/plots/{config['fname']}_rewards.pdf"
    title = f"Reward curve for {config['network'].upper()} on {config['env_id']}"
    if not config["network"] == "cnn":
        title += f"(lip {config['lipschitz']})"
    
    plt.title(title)
    plt.xlabel("Environment steps")
    plt.ylabel("Reward")
    plt.tight_layout()
    
    dirpath = Path(__file__).resolve().parent
    plt.savefig(dirpath / fname)
    plt.close()


def frames_to_video(frames, output_video_name="testing.mp4", fps=60):
    """Take a collection of Atari frames and stitch them togehter
    into a gameplay video."""
    
    # Check inputs and scale up image
    if not len(frames[0].shape) == 2:
        raise ValueError("Each frame should be a grayscale image.")
    new_frames = [cv2.resize(f, (256,256)) for f in frames]
    
    # Write video
    media.write_video(output_video_name, new_frames, fps=fps)
    print("Video generated successfully!")
    

def make_gameplay_video(agent, config, save_frames=False):
    """Generate a gameplay video for a trained agent and save it."""
    
    # Load a single environment
    env = make_env(config["env_id"])
    obs, _ = env.reset()
    
    # Play the game
    done = False
    frames = []
    while not done:
        
        # Store a frame (only 1 env, get 1st of 4 frames)
        frames.append(obs[0,0])
        obs = torch.Tensor(obs).to("cuda")
        
        # Evaluate the agent and env
        action = agent.get_action(obs)
        obs, _, dones, _, _ = env.step(action.cpu().numpy())
        done = dones[0]
        
    # Save path
    dirpath = Path(__file__).resolve().parent
    fname = str(dirpath / f"../results/videos/{config['fname']}")
    
    # Save all the frames if required
    if save_frames:
        data = {'config': config, 'frames': frames}
        with open(fname + ".pickle", 'wb') as fp:
            pickle.dump(data, fp)
    
    # Create a video
    frames_to_video(frames, fname + ".mp4")


def startup_plotting(font_size=14, line_width=1.5, output_dpi=600, tex_backend=True):
    """Edited from https://github.com/nackjaylor/formatting_tips-tricks/"""

    if tex_backend:
        try:
            plt.rcParams.update({
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                        })
        except:
            print("WARNING: LaTeX backend not configured properly. Not using.")
            plt.rcParams.update({"font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                        })
    
    # Default settings
    plt.rcParams.update({
        "lines.linewidth": line_width,
        
        "axes.grid" : True, 
        "axes.grid.which": "major",
        "axes.linewidth": 0.5,
        "axes.prop_cycle": cycler("color", [
            "#0072B2", "#E69F00", "#009E73", "#CC79A7", 
            "#56B4E9", "#D55E00", "#F0E442", "#000000"]),

        "errorbar.capsize": 2.5,
        
        "grid.linewidth": 0.25,
        "grid.alpha": 0.5,
        
        "legend.framealpha": 0.7,
        "legend.edgecolor": [1,1,1],
        
        "savefig.dpi": output_dpi,
        "savefig.format": 'pdf'
    })

    # Change default font sizes.
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=0.8*font_size)
    plt.rc('ytick', labelsize=0.8*font_size)
    plt.rc('legend', fontsize=0.8*font_size)
    