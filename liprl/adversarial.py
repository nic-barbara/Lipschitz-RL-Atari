import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn

from advertorch.attacks import PGDAttack
from pathlib import Path
from liprl import utils
from liprl.ppo_atari_envpool import RecordEpisodeStatistics


def uniform_attack(x, attack_size=0):
    new_x = x.cpu().numpy()
    noise = attack_size * (2*np.random.rand(*new_x.shape) - 1)
    new_x = np.clip(new_x + noise, a_min=0, a_max=255)
    return torch.Tensor(new_x).to("cuda")


def get_attacker_func(agent, attack_size=0, attacker=None, n_iter=40):
    """Create a function to perform attacks on image observations.
    
    Options are as follows:
    - None:         no attacks, returns the un-altered image.
    - "uniform":    uniform random noise with restricted l_infty norm.
    - "l2_pgd":     PGD attack with attack restricted in l2 norm.
    - "l1_pgd":     PGD attack with attack restricted in l1 norm.
    - "linf_pgd":   PGD attack with attack restricted in l_infty norm.
    
    The attack function should take in a single input (a batch of images) and
    return the attacked inputs. The input should be a PyTorch Tensor, and the
    output will be too. Images are expected to have values in [0, 255].
    
    `n_iter` controlls the number of iterations in the PGD attacks (default 40).
    Make it larger for more effective attacks, at the cost of slower compute.
    Note that 40 is EXTREMELY slow, so change this if you need speedier results.
    """
    
    if (not attacker) or (attack_size == 0):
        
        return lambda x: x
    
    elif attacker == "uniform":
        
        return lambda x: uniform_attack(x, attack_size)
    
    elif attacker in ["l2_pgd", "l1_pgd", "linf_pgd"]:
        
        if attacker == "l2_pgd":
            norm = 2
        elif attacker == "l1_pgd":
            norm = 1
        else:
            norm = np.inf
        
        # Autoattack assumes all image values are in [0,1] already
        forward_pass = lambda x: agent.get_logits(x * 255)
        adversary = PGDAttack(forward_pass, 
                              loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                              eps=attack_size / 255, 
                              nb_iter=n_iter, 
                              ord=norm)
        
        def _attack(x):
            x_adv = adversary.perturb(x / 255)
            return x_adv * 255
        
        return _attack
    
    else:        
        raise ValueError(f"Unrecognised attacker '{attacker}'")


def attacked_gameplay_video(
    agent, 
    config,
    attacker=None,
    attack_size=0,
    save_frames=False
):
    """Generate a gameplay video for a trained agent with an 
    attack on the observations and save it."""
    
    # Load a single environment
    env = utils.make_env(config["env_id"])
    obs, _ = env.reset()
    
    # Get an attacker
    attack_fn = get_attacker_func(agent, attack_size, attacker)
    
    # Play the game
    done = False
    frames = []
    original = []
    while not done:
        
        # Store a frame (only 1 env, get 1st of 4 frames)
        obs = torch.Tensor(obs).to("cuda")
        obs_attacked = attack_fn(obs)
        original.append(obs[0,0].cpu().numpy().astype(np.uint8))
        frames.append(obs_attacked[0,0].cpu().numpy().astype(np.uint8))
        
        # Evaluate the agent and env
        action = agent.get_action(obs_attacked)
        obs, _, dones, _, _ = env.step(action.cpu().numpy())
        done = dones[0]
        
    # Save path
    dirpath = Path(__file__).resolve().parent
    attacker = attacker if attacker else "unperturbed"
    fname = str(dirpath / f"../results/videos/{config['fname']}_{attacker}_{attack_size}")
    
    # Save all the frames if required
    if save_frames:
        data = {'config': config, 
                'frames': frames, 
                'attacker': attacker,
                'attack_size': attack_size,
                'unperturbed': original}
        with open(fname + ".pickle", 'wb') as fp:
            pickle.dump(data, fp)
            
    # Create a video
    utils.frames_to_video(frames, fname + ".mp4")


def get_batch_reward(
    agent, 
    config, 
    num_envs=20, 
    num_games=20,
    attacker=None,
    attack_size=0, 
    verbose=True
):
    """Evaluate reward of a trained agent on a batch of envs
    using an attacker."""
    
    # Load a batch of environments
    envs = utils.make_env(config["env_id"], num_envs=num_envs)
    envs = RecordEpisodeStatistics(envs)
    obs = envs.reset()
    
    # Get an attacker
    attack_fn = get_attacker_func(agent, attack_size, attacker)

    # Play the game
    completed_games = 0
    game_rewards = np.zeros(num_games)
    while completed_games < num_games:
        
        # Perturb the observation with an attacker
        obs = torch.Tensor(obs).to("cuda")
        obs = attack_fn(obs)
        
        # Evaluate the agent and env
        action = agent.get_action(obs)
        obs, _, dones, info = envs.step(action.cpu().numpy())
        
        # Store rewards for completed games
        for idx, d in enumerate(dones):
            if d and info["lives"][idx] == 0:
                game_rewards[completed_games] = info["r"][idx]
                completed_games += 1
                current_reward = np.mean(game_rewards[game_rewards != 0])
                if verbose:
                    print("Completed games: {},\t reward: {:.2f}".format(
                        completed_games, current_reward
                    ))
            if completed_games >= num_games:
                break
                
    return np.mean(game_rewards), np.std(game_rewards)


def get_attacked_rewards(
    agent, 
    config, 
    attacker=None,
    attack_sizes=[0],
    verbose=True,
):
    """Get an array of attacked rewards for a given set of attacks and attacker."""
    
    metrics = {
        "attacks": np.array(attack_sizes),
        "rewards": np.zeros(len(attack_sizes)),
        "stdev": np.zeros(len(attack_sizes))
    }
    for i in range(len(attack_sizes)):
        out = get_batch_reward(agent, 
                               config, 
                               attack_size=attack_sizes[i],
                               attacker=attacker,
                               verbose=False)
        metrics["rewards"][i] = out[0]
        metrics["stdev"][i] = out[1]
        
        if verbose:
            print("{}: attack size {:.3g} \t reward: {:.2f}".format(
                attacker, attack_sizes[i], metrics["rewards"][i]
            ))

    return metrics


def save_attack_results(savepath, config: dict, results: dict):
    data = {"config": config, "attack_metrics": results}
    with open(savepath / f"{config['fname']}.pickle", 'wb') as fp:
        pickle.dump(data, fp)
        

def load_attack_results(load_path):
    with open(load_path, 'rb') as fp:
        data = pickle.load(fp)
    return data


def plot_attacked_rewards(config, metrics):
    """Plot environment reward vs attack sizes for a given attacker.
    Run `metrics = get_attacked_rewards(...) first."""
    
    # Grab results
    x = metrics["attacks"]
    y = metrics["rewards"]
    y_err = metrics["stdev"]
    attacker = metrics["attacker"]
    
    # Make a plot and make it pretty
    plt.plot(x, y)
    plt.fill_between(x, y - y_err, y + y_err, alpha=0.2)
    
    fname = f"../results/plots/{config['fname']}_attacked_rewards_{attacker}.pdf"
    title = f"Attacked rewards of {config['network'].upper()} on {config['env_id']} ({attacker})"
    if not config["network"] == "cnn":
        title += f" (lip {config['lipschitz']})"
    
    plt.title(title)
    plt.xlabel("Attack size")
    plt.ylabel("Reward")
    plt.tight_layout()
    
    dirpath = Path(__file__).resolve().parent
    plt.savefig(dirpath / fname)
    plt.close()
