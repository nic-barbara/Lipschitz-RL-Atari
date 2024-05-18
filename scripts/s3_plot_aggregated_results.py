import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from pathlib import Path

import utils
import liprl.utils

# What to plot for
plot_config = {
    "Pong-v5": {
        "uniform": 25.0,
        "l2_pgd": 70.0,
        "linf_pgd": 1.05,
    }
}
reward_networks = ["lbdn", "orthogonal", "aol", "spectral"]
attack_networks = ["lbdn"]
table_networks = ["cnn", "lbdn", "orthogonal", "aol", "spectral"]

labels = {"cnn": "CNN", "lbdn": "Sandwich", "orthogonal": "Cayley",
          "aol": "AOL", "spectral": "SN"}

plot_lipschitz = [5.0, 10.0, 20.0, 60.0, 100.0] # Choose a subset if we want to

# Plotting options
do_titles = False
fsize = 20
colours = cycler("color", ["#E69F00", "#009E73", 
                           "#CC79A7", "#56B4E9", 
                           "#D55E00", "#F0E442", "#000000"])

# Setup
liprl.utils.startup_plotting(font_size=fsize)
dirpath = Path(__file__).resolve().parent
fpath = dirpath / "../results/params/"
apath = dirpath / "../results/attack-results/"

# Get file names and network info
train_files = [utils.get_network_info(str(f)) for f in fpath.iterdir() if f.is_file()]
attack_files = [utils.get_network_info(str(f)) for f in apath.iterdir() if f.is_file()]

# Aggregate results according to network type + lipschitz bound
network_types = utils.lunique([d["network"] for d in train_files])
lipschitz_bounds = utils.lunique([d["lipschitz"] for d in train_files])

# Order the networks and Lipschitz bounds nicely for plotting
network_types = utils.order_networks(network_types)
lipschitz_bounds = utils.order_lipschitz(lipschitz_bounds)

# Aggregate reward and perturbed reward data
reward_data = utils.aggregate_rewards(train_files, network_types, lipschitz_bounds)
attack_data = utils.aggregate_perturbed_rewards(attack_files, 
                                                network_types,
                                                lipschitz_bounds, 
                                                plot_config["Pong-v5"].keys())


def _plot_rewards(env_id, network):
    """Plot reward vs. environment steps during training."""
    
    # Make a plot of all reward curves
    fig, ax = plt.subplots()
    for n in reward_data:
        
        # Find the networks of interest
        data = reward_data[n]
        if not data["network"] in [network, "cnn"]:
            continue
        if not data["network"] == "cnn" and not data["lipschitz"] in plot_lipschitz:
            continue
        
        metrics = data["metrics"]
        
        # Extract data
        x = metrics["steps"] / 1e6
        y = metrics["rewards"]
        y_err = metrics["stdev"]
        
        # Plot formatting
        name_ = labels[data["network"]]
        if not data["network"] == "cnn":
            name_ += " ($\gamma = {:.0f}$)".format(data["lipschitz"])
            
        if data["network"] == "cnn":
            lstyle = "dotted"
        else:
            lstyle = "solid"
        
        # Plot data
        ax.plot(x, y, label=name_, linestyle=lstyle)
        ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)
    
    # Format the plot
    ax.set_xlim(0, x.max())
    ax.set_ylim(-22, 24)
    
    if do_titles: ax.set_title(f"Reward curves on {env_id}")
    ax.set_xlabel("Environment steps ($\\times 10^6$)")
    ax.set_ylabel("Reward (game win margin)")
    ax.legend()
    fig.tight_layout()
    
    plt.savefig(dirpath / f"../results/plots/{env_id}_{network}_reward_curves.pdf")
    plt.close()


def plot_rewards(env_id):
    """Separate figure for each network so it's not cluttered"""
    for network in reward_networks:
        _plot_rewards(env_id, network)


def plot_attacked_rewards(env_id, attacker):
    """Plot reward vs. attack size during testing."""
    
    new_network = "cnn"
    fig, ax = plt.subplots(figsize=(6.2,6.2))
    for n in attack_data:
        
        # Load data for some networks
        data = attack_data[n]
        if not data["network"] in attack_networks + ["cnn"]:
            continue
        if not data["network"] == "cnn" and not data["lipschitz"] in plot_lipschitz:
            continue
        metrics = data[attacker]
        
        # Extract data
        x = metrics["attacks"]
        y = metrics["rewards"]
        y_err = metrics["stdev"]
        
        # Formatting choices
        name_ = labels[data["network"]]
        if not data["network"] == "cnn":
            name_ += " ($\gamma = {:.0f}$)".format(data["lipschitz"])
            
        if data["network"] == "cnn":
            lstyle = "dotted"
        elif data["network"] == "lbdn":
            lstyle = "solid"
        else:
            lstyle = "dashdot"
            
        # Decide whether to reset the colour cycle
        # Useful if comparing CNN with multiple network architectures
        # (uses the same colours for each gamma)
        if not data["network"] == new_network:
            ax.set_prop_cycle(colours)
            new_network = data["network"]
        
        # Plot data
        ax.plot(x, y, label=name_, linewidth=1.5, linestyle=lstyle)
        ax.fill_between(x, y - y_err, y + y_err, alpha=0.2)
    
    # Format the plot
    ax.set_xlim(0, x.max())
    ax.set_ylim(-24, 24)
    
    if do_titles: ax.set_title(f"Attacked reward curves on {env_id} ({attacker})")
    ax.set_xlabel("Attack size $\epsilon$")
    ax.set_ylabel("Reward (game win margin)")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    
    plt.savefig(dirpath / f"../results/plots/{env_id}_attack_reward_curves_{attacker}.pdf")
    plt.close()


def plot_attacked_rewards_lip(env_id, attacker, attack):
    """Plot reward vs Lipschitz bound for a particular attack."""
    
    # Formatting
    ecolor = "grey"
    elinewidth = 0.8
    attack_label = "{:.3g}".format(attack)
    
    new_network = "cnn"
    fig, ax = plt.subplots(figsize=(6.2,6.2))
    for n in attack_data:
        
        # Load data for some networks
        data = attack_data[n]
        if not data["network"] in attack_networks + ["cnn"]:
            continue
        if not data["network"] == "cnn" and not data["lipschitz"] in plot_lipschitz:
            continue
        metrics = data[attacker]
        
        index = np.isclose(metrics["attacks"], attack)
        x = np.array(metrics["lip_mean"])
        y = np.array(metrics["rewards"][index])
        x_err = np.array(metrics["lip_stdev"])
        y_err = np.array(metrics["stdev"][index])
        
        if not index.any():
            print(attack)
            print(metrics["attacks"])
            raise ValueError("Choose an attack that has been simulated.")
              
        # Formatting choices
        name_ = labels[data["network"]]
        if not data["network"] == "cnn":
            name_ += " ($\gamma = {:.0f}$)".format(data["lipschitz"])
            
        if data["network"] == "cnn":
            marker = "o"
        elif data["network"] == "lbdn":
            marker = "s"
        else:
            marker = "D"
            
        # Decide whether to reset the colour cycle
        # Useful if comparing CNN with multiple network architectures
        # (uses the same colours for each gamma)
        if not data["network"] == new_network:
            ax.set_prop_cycle(colours)
            new_network = data["network"]
            
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt=marker, label=name_,
                    ecolor=ecolor, elinewidth=elinewidth,
                    markersize=8)
        
    # Format the plot
    ax.set_xlim(3,1000)
    ax.set_ylim(-24,24)
    
    if do_titles: ax.set_title(f"Reward at attack = {attack_label} on {env_id} ({attacker})")
    ax.set_xlabel("Lipschitz lower bound $\\underline\\gamma$")
    ax.set_ylabel("Reward (game win margin)")
    ax.set_xscale('log')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    
    plt.savefig(dirpath / f"../results/plots/{env_id}_attack_reward_lip_{attacker}.pdf")
    plt.close()


def summarise_results(env_id, attackers, attack_data):
    """Summarise all the attack results in a big fat table."""
    
    # Only look at the main results (might want to remove this later)
    new_attack_data = {}
    for n in attack_data:
        if ((attack_data[n]["lipschitz"] in plot_lipschitz) and
            (attack_data[n]["network"] in table_networks)) or attack_data[n]["network"] == "cnn":
            new_attack_data[n] = attack_data[n]
    attack_data = new_attack_data
    
    # Column and row labels
    col_labels = ["Model", "$\gamma$", "$\\bar{\gamma}$", "Reward"]
    for a in attackers:
        col_labels += [a]
        
    # Create the dataframe
    n_cols = len(col_labels)
    n_models = len(attack_data)
    df = pd.DataFrame(np.zeros((n_models, n_cols)), columns=col_labels)
    df.index = list(attack_data.keys())
    
    # Store the model information
    models = []
    lips = []
    lip_tightness = []
    rewards = []
    for n in attack_data:
        
        # Nicely format the model name
        data = attack_data[n]
        name_ = labels[data["network"]]
        models.append(name_)
            
        # Get Lipschitz information
        lips.append(data["lipschitz"])
        lip_tightness.append(data["lip_estimate"])
        
        # Also get data for max reward
        if np.isnan(data["uniform"]["rewards"]).any():
            data_r = reward_data[n]["metrics"]["rewards"][-1]
        else:
            data_r = data["uniform"]["rewards"][0]
        rewards.append(data_r)
        
    df["Model"] = models
    df.iloc[:,1] = lips
    df.iloc[:,2] = np.array(lip_tightness).round(2)
    df.iloc[:,3] = np.array(rewards).round(2)

    # Get the closest attack size to losing the game for each attacker
    for j in range(len(attackers)):
        
        attacker = attackers[j]
        best_row = [-np.inf, -1]
        never_loses = False
        fspec = "{:.2f}" if "linf" in attacker else "{:.1f}"
        
        for i in range(n_models):
            
            model = df.index[i]
            data = attack_data[model][attacker]
            rewards = data["rewards"]
            attacks = data["attacks"]
            
            # Fill with dummy info if no useful information
            if np.isscalar(rewards):
                df.iloc[i, j+4] = "-"
                continue
            
            # Check if it never loses the game
            if (rewards > 0).all():
                df.iloc[i, j+4] = "$\mathbf{>" + fspec.format(attacks.max()) + "}$"
                never_loses = True
                continue
            
            # Check if it always loses the game
            if (rewards < 0).all():
                df.iloc[i, j+4] = "-"
                continue
            
            # Store attack size with reward closest to 0 (interp to find it)
            winning_attack = utils.get_winning_attack(attacks, rewards)
            df.iloc[i, j+4] = fspec.format(winning_attack)
            if winning_attack > best_row[0]:
                best_row = [winning_attack, i]
            
        # Make bold numbers
        if best_row[1] > 0 and not never_loses:
            df.iloc[best_row[1], j+4] = "\\textbf{" + fspec.format(best_row[0]) + "}"
            
    # Change the attack names
    new_names = {"uniform": "Uniform", 
                 "l2_pgd": "$\ell_2$ PGD", 
                 "linf_pgd": "$\ell_\infty$ PGD"}
    df = df.rename(columns=new_names)
    
    # Format the Lipschitz bounds
    df.iloc[0,1] = 0 # not defined for cnn
    df["$\gamma$"] = df["$\gamma$"].astype(int)
    
    # Write to file      
    fname = dirpath / f"../results/plots/{env_id}_robustness_table.txt"
    results_file = open(fname, "w")
    results_file.write(df.to_latex(index=False, 
                                   float_format="%.3g"))
    results_file.close()


# Run all the plotting functions    
env = "Pong-v5"
plot_rewards(env)
for attacker in plot_config[env]:
    plot_attacked_rewards(env, attacker)
    plot_attacked_rewards_lip(env, attacker, plot_config[env][attacker])

# Tables of results
attackers = list(plot_config[env].keys())
summarise_results(env, attackers, attack_data)
