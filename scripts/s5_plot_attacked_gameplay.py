import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns

from pathlib import Path
from liprl import utils

utils.startup_plotting(font_size=20)

# Set up
dirpath = Path(__file__).resolve().parent
fpath = dirpath / "../results/videos/"
frame_files = [str(f) for f in fpath.iterdir() if f.is_file() and (f.suffix == ".pickle")]

networks = ["cnn", "lbdn"]
labels = {networks[0]: "CNN", networks[1]: "Sandwich"}
attackers = ["None", "uniform", "l2_pgd", "linf_pgd"]
attack_labels = ["None", "Uniform", "$\ell_2$ PGD", "$\ell_\infty$ PGD"]

# Choose which frames to show (indexed from the end)
# Hard-coded for aesthetics
indices = np.array([
    [122, 137],
    [107, 922],
    [97, 418],
    [985, 689],
])

# Custom colour map
my_cmap = sns.color_palette("icefire", as_cmap=True)


def read_files(frame_files, networks=networks):
    """Read in frame files for networks of interest."""
    
    files = []
    for f in frame_files:
        for network in networks:
            if network in f:
                files.append(f)
                break
    
    frame_data = []
    for f in files:
        with open(f, 'rb') as fp:
            data = pickle.load(fp)
        frame_data.append({
            'network': data['config']['network'],
            'frames': data['frames'],
            'attacker': data['attacker'] if 'attacker' in data else 'None',
            'unperturbed': data['unperturbed'] if 'attacker' in data else None
        })
    return frame_data


def get_frames(frame_data: dict, network: str, attacker: str):
    """
    Pick out frames for a particular network and attacker. 
    
    Return the perturbed frames and the original (unperturbed)
    measurements (if applicable).
    """
    
    for data in frame_data:
        if data["network"] == network and data["attacker"] == attacker:
            frames = data["frames"]
            originals = data["unperturbed"]
            return frames, originals
    raise ValueError(f"Couldn't find data for {network} with {attacker}.")


def get_image_frame(frames, originals, end_index=100):
    index = len(frames) - end_index
    new_frame = frames[index]
    if not originals is None:
        new_original = originals[index]
        if not len(frames) == len(originals): 
            raise ValueError("Should have the same number of frames")
    else:
        new_original = None
    return new_frame, new_original


def plot_pong_image(ax, image, title="", ylabel=""):
    ax.imshow(image, cmap="grey")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    

def plot_pong_attack(ax, image, original, title="", clim=None):
    
    # Plot difference to show attack
    if not original is None:
        diff = image.astype(np.int64) - original.astype(np.int64)
        im = ax.imshow(diff, cmap=my_cmap)
        fig.colorbar(im, ax=ax, shrink=0.95) # This is hardcoded for aesthetics
        if clim: 
            im.set_clim(*clim)
    
    # Format
    ax.set_axis_off()
    ax.set_title(title)
    

def _get_clim(attacker):
    """Hard-code colour bar limits for aesthetics in paper plots."""
    
    if attacker == "None":
        return None
    if attacker == "uniform":
        return (-35, 35)
    if attacker == "l2_pgd":
        return (-5.2, 5.2)
    if attacker == "linf_pgd":
        return (-2,2)


# Choose figure and axes sizes
figsize = (11,10)
width_ratios = [1, 1.25, 1, 1.25] # This is hardcoded for aesthetics

# Make the plot
frame_data = read_files(frame_files)
fig, axs = plt.subplots(len(attackers), 2*len(networks), figsize=figsize,
                        gridspec_kw={'width_ratios': width_ratios})

for i in range(len(attackers)):
    for j in range(len(networks)):
        
        # Get images to plot
        attacker = attackers[i]
        network = networks[j % 2]
        index = indices[i, j]
        frames, originals = get_frames(frame_data, network, attacker)
        image, image0 = get_image_frame(frames, originals, index)
        
        # Formatting and labelling
        title = labels[network] if i == 0 else ""
        a_title = f"Attacks ({title})" if i == 0 else ""
        ylabel = attack_labels[i] if j == 0 else ""
        
        # Add to the figure
        clim = _get_clim(attacker)
        plot_pong_image(axs[i,j*len(networks)], image, title, ylabel)
        plot_pong_attack(axs[i,j*len(networks)+1], image, image0, a_title, clim=clim)
    
plt.savefig(dirpath / "../results/plots/Pong-v5_attack_examples.pdf")
plt.close()
