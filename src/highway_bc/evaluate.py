import torch
import gymnasium as gym
import highway_env
from highway_bc.policy import BCPolicy
from highway_bc.expert_policy import expert_policy

MAX_STEPS = 40
EVAL_SEED_OFFSET = 5000
N_EPISODES = 200


def run_episode(env, model, seed):
    obs, info = env.reset(seed=seed)
    crashed = False
    total_reward, steps = 0, 0

    with torch.no_grad():

        for step in range(MAX_STEPS):
            obs = torch.from_numpy(obs).float()
            obs = obs.flatten()
            obs = obs.unsqueeze(0)
            pred = model(obs)
            action = (pred.argmax(1)).item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if info.get("crashed", False):
                crashed = True
                break
            if terminated or truncated:
                break

    return crashed, total_reward, steps

def run_expert_episode(env, model, seed):
    action_indexes = env.unwrapped.action_type.actions_indexes
    obs, info = env.reset(seed=seed)
    crashed = False
    total_reward, steps = 0, 0

    for step in range(MAX_STEPS):
        action = expert_policy(
                obs, action_indexes, env.unwrapped.action_type.get_available_actions()
            )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if info.get("crashed", False):
            crashed = True
            break
        if terminated or truncated:
            break

    return crashed, total_reward, steps

def evaluate(episode_fn, model, n_episodes):
    env = gym.make("highway-v0")    # , render_mode="rgb_array"
    n_crashed, total_rewards, total_steps = 0, 0, 0

    for i in range(n_episodes):
        crashed, reward, steps = episode_fn(env, model, EVAL_SEED_OFFSET + i)
        total_rewards += reward
        total_steps += steps
        if crashed:
            n_crashed += 1
        print(f"episode {i+1}/{n_episodes}", end="\r")

    env.close()
    crash_rate = n_crashed / n_episodes
    avg_reward = total_rewards / n_episodes
    avg_steps = total_steps / n_episodes

    return crash_rate, avg_reward, avg_steps


def main():
    model = BCPolicy()
    model.load_state_dict(torch.load('checkpoints/bc_policy.pth', weights_only=True))
    model.eval()

    crash_rate, avg_reward, avg_steps = evaluate(run_episode, model, N_EPISODES)
    expert_crash_rate, expert_avg_reward, expert_avg_steps = evaluate(run_expert_episode, model, N_EPISODES)

    print(f"\nevaluated on {N_EPISODES} episodes, seeds {EVAL_SEED_OFFSET}-{EVAL_SEED_OFFSET + N_EPISODES - 1}\n")
    print(f"{'policy':<12}{'crash rate':>12}{'avg return':>12}{'avg steps':>11}")
    print(f"{'BC policy':<12}{100*crash_rate:>11.1f}%{avg_reward:>12.2f}{avg_steps:>11.1f}")
    print(f"{'expert':<12}{100*expert_crash_rate:>11.1f}%{expert_avg_reward:>12.2f}{expert_avg_steps:>11.1f}")


if __name__ == "__main__":
    main()