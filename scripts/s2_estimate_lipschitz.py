import torch
import pickle

from pathlib import Path
from liprl import utils
from liprl import adversarial as adv

# Setup
dirpath = Path(__file__).resolve().parent
fpath = dirpath / "../results/params/"
apath = dirpath / "../results/attack-results/"

train_files = [str(f) for f in fpath.iterdir() if f.is_file()]
train_files.sort()


def _get_lipschitz(params, config, verbose=True):
    
    # Load a trained model
    utils.seed_everything(config["seed"])
    env = utils.make_env(config["env_id"])
    agent = utils.load_agent(env, params, config)

    # Set up observation to perturb
    obs, _ = env.reset()
    obs = torch.Tensor(obs).to("cuda") / 255
    forward_pass = lambda x: agent.actor(agent.network(x))
    
    # Estimate Lipschitz bound
    keep_graph = True if config["network"] == "aol" else False
    print(f"\nEstimating Lipschitz bound for {config['fname']}:")
    lip = utils.empirical_lipschitz(forward_pass, obs, 
                                    verbose=verbose, 
                                    keep_graph=keep_graph)
    
    return lip


def _write_lipschitz(f):
    
    # Load data for a trained model
    data = utils.load_params_configs(f)
    params, config = data["params"], data["config"]
    
    # Estimate Lipschitz bound
    lip = _get_lipschitz(params, config)
    
    # Read the attack results file (should have same name!!)
    fname = apath / f"{config['fname']}.pickle"
    if fname.exists():
        data = adv.load_attack_results(fname)
    else:
        data = {"config": config}
        print(f"Creating new results file for {config['fname']}")
        
    # Store the Lipschitz bound and save again
    data["lip_estimate"] = lip
    with open(fname, 'wb') as fp:
            pickle.dump(data, fp)
        
    # Read to test
    data_new = adv.load_attack_results(fname)
    print("Saved estimated gamma as {:.3g}".format(data_new["lip_estimate"]))
    

for f in train_files:
    _write_lipschitz(f)
