import numpy as np
import liprl.utils
import liprl.adversarial as adv

def order_files(files: list):
    """Sort files with CNN at the top for plotting convenience."""
    files.sort()
    index = ["cnn" in f for f in files]
    cnn_index = np.arange(len(index))[index][0]
    files.insert(0, files.pop(cnn_index))
    return files


def get_network_info(f):
    """Get information about a network form its filename only."""
    
    fsplit = f.split("_")
    network = fsplit[1]
    version = fsplit[-1].split(".")[0]
    lipschitz = fsplit[2][1:] if len(fsplit) > 3 else "inf"

    return {"network": network, "version": version, "lipschitz": lipschitz, "fname": f}


def lunique(x:list) -> list:
    """Return list of unique elements in a list."""
    y = np.unique(np.array(x))
    return list(y)


def order_networks(x: list):
    """Sort a list of network names alphabetically but with 'cnn' at the front."""
    x.sort()
    index = ["cnn" in f for f in x]
    cnn_index = np.arange(len(index))[index][0]
    x.insert(0, x.pop(cnn_index))
    return x


def order_lipschitz(x: list):
    """Sort a list of strings of floats into ascending order and remove 'inf'"""
    lips = np.array([float(xi) for xi in x if not xi == 'inf'])
    x = np.array(x)[lips.argsort()]
    return list(x)


def combine_stdev(s):
    """Approximate the total stdev with sqrt(pooled variance)."""
    return np.sqrt((s**2).mean(axis=0))


def _resample(xs, ys):

    # Get common x-scale
    xmin = np.max([x[0] for x in xs])
    xmax = np.min([x[-1] for x in xs])
    xlen = np.min([len(x) for x in xs])
    x0 = np.linspace(xmin, xmax, xlen)
    
    # Re-scale the data
    new_x, new_y = [], []
    for i in range(len(ys)):
        new_x.append(x0)
        new_y.append(np.interp(x0, xs[i], ys[i]))
        
    return new_x, new_y


def _get_reward_data(file_data):
    """
    Aggregate reward data for a collection of files. Intended 
    for use inside aggregate_rewards(). All files should have
    a common network architecture.
    """
    
    rewards, stdevs, steps = [], [], []
    for d in file_data:
        data = liprl.utils.load_params_configs(d["fname"])
        metrics = data["metrics"]
        rewards.append(metrics["rewards"])
        stdevs.append(metrics["stdev"])
        steps.append(metrics["steps"])
        
    _, rewards = _resample(steps, rewards)
    steps, stdevs = _resample(steps, stdevs)
        
    r = np.array(rewards)
    s = np.array(stdevs)
    metrics = {
        "steps": steps[0],
        "rewards": r.mean(axis=0),
        "stdev": combine_stdev(s)
    }
    return {"network": file_data[0]["network"], 
            "lipschitz": float(file_data[0]["lipschitz"]),
            "metrics": metrics}
    

def aggregate_rewards(files, network_types, lipschitz_bounds):
    """
    Aggregate all reward information for a set of networks 
    and store important plotting information.
    """
    
    results = {}
    for network in network_types:
        
        if network == "cnn":
            name_ = network
            network_data = [d for d in files if d["network"] == network]
            if network_data == []:
                continue
            results[name_] = _get_reward_data(network_data)
            continue
        
        for g in lipschitz_bounds:
            name_ = network + f"_{g}"
            network_data = [d for d in files if (d["network"] == network) 
                            and (d["lipschitz"] == g)]
            if network_data == []:
                continue
            results[name_] = _get_reward_data(network_data)

    return results


def _get_perturbed_reward_data(file_data, attackers):
    """
    Aggregate perturbed reward data for a collection of files. 
    Intended for use inside aggregate_perturbed_rewards(). All files
    should be for a common network architecture.
    """
    
    # Aggregate results for each attack separately
    results = {}
    no_attack_data = False
    for a in attackers:
        rewards, stdevs, steps, lips = [], [], [], []
        for d in file_data:
            
            # Load data for this model
            data = adv.load_attack_results(d["fname"])
            lips.append(data["lip_estimate"])
            
            # Fill and track metrics if available
            if not "attack_metrics" in data:
                no_attack_data = True
                print("No attack data for: ", d["network"], d["lipschitz"], a)
            else:
                metrics = data["attack_metrics"][a]
                rewards.append(metrics["rewards"])
                stdevs.append(metrics["stdev"])
                steps.append(metrics["attacks"])
        
        # Store relevant data
        r = np.array(rewards)
        s = np.array(stdevs)
        l = np.array(lips)
        results[a] = {
            "attacks": np.zeros(1) if no_attack_data else steps[0],
            "rewards": r.mean(axis=0),
            "stdev": combine_stdev(s),
            "lip_mean": np.array([l.mean()]),
            "lip_stdev": np.array([l.std()])
        }
        
    # Add extra info to results dict
    keys = list(results.keys())
    results["network"] = file_data[0]["network"]
    results["lipschitz"] = float(file_data[0]["lipschitz"])
    results["lip_estimate"] = results[keys[0]]["lip_mean"][0]
    results["lip_stdev"] = results[keys[0]]["lip_stdev"][0]
    return results


def aggregate_perturbed_rewards(files, network_types, lipschitz_bounds, attackers):
    """
    Aggregate all reward information for a set of networks 
    and store important plotting information.
    """
    
    results = {}
    for network in network_types:
        
        if network == "cnn":
            name_ = network
            network_data = [d for d in files if d["network"] == network]
            results[name_] = _get_perturbed_reward_data(network_data, attackers)
            continue
        
        for g in lipschitz_bounds:
            name_ = network + f"_{g}"
            network_data = [d for d in files if (d["network"] == network) 
                            and (d["lipschitz"] == g)]
            if not network_data:
                print(f"Skipping attack aggregation for {name_}")
                continue
            results[name_] = _get_perturbed_reward_data(network_data, attackers)

    return results


def get_winning_attack(xs, ys):
    """Find value of x for which y is 0 by linear interpolation between points.
    Assumes y has a negative gradient."""
    
    # Find x and y values either side of zero
    index = np.where(ys < 0)[0][0]
    if index == 0:
        return xs[index]
    x0, x1 = xs[index-1], xs[index]
    y0, y1 = ys[index-1], ys[index]
    
    # Use linear interpolation to estimate value for x when y = 0
    x_at_0 = x0 - y0 * (x1 - x0) / (y1 - y0)
    return x_at_0
