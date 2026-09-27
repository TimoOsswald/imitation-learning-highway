import numpy as np
import gymnasium as gym
import highway_env 
from highway_bc.expert_policy import expert_policy

MAX_STEPS = 40


def run_episode(env, seed):
    """
    Drives one episode with the expert policy and records every (state, action) pair.

    Args:
        env: highway-env environment (created once outside)
        seed: random seed for env.reset()

    Returns:
        tuple: (obs_list, act_list, crashed)
            obs_list: list of kinematics arrays, each of shape [V, F], state before each action
            act_list: list of ints, the expert's action index for the matching state
            crashed: True if the ego-vehicle collided during the episode
    """
    action_indexes = env.unwrapped.action_type.actions_indexes
    obs, info = env.reset(seed=seed)
    crashed = False
    obs_list, act_list = [], []

    for step in range(MAX_STEPS):
        action = expert_policy(
            obs, action_indexes, env.unwrapped.action_type.get_available_actions()
        )
        obs_list.append(obs)      # state before the step
        act_list.append(action)   # expert action chosen for that state
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("crashed", False):
            crashed = True
            break
        if terminated or truncated:
            break

    return obs_list, act_list, crashed


def collect(n_seeds, seed_offset=1000):
    """
    Collects expert demonstrations over many seeds and discards crashed episodes.

    Args:
        n_seeds: number of episodes to run 
        seed_offset: first seed; keep it separate from the seeds used for evaluation later

    Returns:
        tuple: (obs, act)
            obs: np.ndarray of shape [N, V, F], all recorded states
            act: np.ndarray of shape [N], the expert action index for each state
    """
    env = gym.make("highway-v0", render_mode="rgb_array")
    all_obs, all_act = [], []
    n_crashed = 0

    for i in range(n_seeds):
        obs_list, act_list, crashed = run_episode(env, seed_offset + i)
        if crashed:
            n_crashed += 1
            continue                  # skip the whole episode
        all_obs.extend(obs_list)      # extend = append elements flat, not as nested list
        all_act.extend(act_list)

    env.close()
    print(f"{n_crashed}/{n_seeds} episodes discarded")
    return np.array(all_obs), np.array(all_act)


from multiprocessing import Pool

_env = None  # one environment per worker process


def _init_worker():
    """Creates the environment once per worker process."""
    global _env
    _env = gym.make("highway-v0")


def _run_seed(seed):
    """Worker task: runs one episode on the process-local environment."""
    return run_episode(_env, seed)


def collect_parallel(n_seeds, seed_offset=1000, n_workers=12):
    """
    Same as collect(), but distributes the seeds over several processes.

    Args:
        n_seeds: number of episodes to run
        seed_offset: first seed
        n_workers: number of processes (leave a few of the 16 threads free)

    Returns:
        tuple: (obs, act) as in collect()
    """
    seeds = range(seed_offset, seed_offset + n_seeds)
    all_obs, all_act = [], []
    n_crashed = 0

    with Pool(n_workers, initializer=_init_worker) as pool:
        # imap keeps the seed order, so results stay reproducible
        for i, (obs_list, act_list, crashed) in enumerate(
            pool.imap(_run_seed, seeds, chunksize=5)
        ):
            if crashed:
                n_crashed += 1
            else:
                all_obs.extend(obs_list)
                all_act.extend(act_list)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{n_seeds} episodes done")

    print(f"{n_crashed}/{n_seeds} episodes discarded")
    return np.array(all_obs), np.array(all_act)


if __name__ == "__main__":
    obs, act = collect_parallel(1000)
    print("obs:", obs.shape, "act:", act.shape)
    print("action distribution:", np.bincount(act))
    np.savez_compressed("data/demos_1000.npz", obs=obs, actions=act)