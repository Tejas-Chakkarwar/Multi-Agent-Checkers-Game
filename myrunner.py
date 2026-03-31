from typing import Dict, List, Tuple

import numpy as np
import torch

from myagent import AgentTransition, SelfPlayActorCriticAgent
from mycheckersenv import env as create_env


def run_training(
    episode_count: int = 400,
    print_every: int = 20,
    seed: int = 42,
) -> Tuple[SelfPlayActorCriticAgent, List[float]]:
    training_env = create_env(render_mode=None)
    training_env.reset(seed=seed)
    sample_observation = training_env.observe("player_0")
    observation_size = int(np.prod(sample_observation["board"].shape))
    action_size = training_env.action_space("player_0").n
    actor_critic_agent = SelfPlayActorCriticAgent(
        observation_size=observation_size,
        action_size=action_size,
        learning_rate=0.0005,
        discount_factor=0.99,
        hidden_size=256,
        value_loss_weight=0.5,
        entropy_weight=0.01,
        device="cpu",
    )

    reward_history = []

    episode_index = 0
    while episode_index < episode_count:
        training_env.reset(seed=seed + episode_index)
        transitions_per_agent: Dict[str, List[AgentTransition]] = {"player_0": [], "player_1": []}
        cumulative_rewards = {"player_0": 0.0, "player_1": 0.0}

        for acting_agent in training_env.agent_iter():
            observation, reward, termination, truncation, _ = training_env.last()
            cumulative_rewards[acting_agent] += float(reward)
            if termination or truncation:
                training_env.step(None)
                continue

            action_index, log_prob, state_value, state_tensor = actor_critic_agent.select_action(observation)
            training_env.step(action_index)
            immediate_reward = float(training_env.rewards[acting_agent])
            done_flag = bool(training_env.terminations[acting_agent] or training_env.truncations[acting_agent])
            transition = AgentTransition(
                state_tensor=state_tensor,
                action_index=action_index,
                log_probability=log_prob,
                state_value=state_value,
                reward=immediate_reward,
                done=done_flag,
            )
            transitions_per_agent[acting_agent].append(transition)

        all_transitions = transitions_per_agent["player_0"] + transitions_per_agent["player_1"]
        update_metrics = actor_critic_agent.update_policy(all_transitions)
        reward_history.append(cumulative_rewards["player_0"])

        if (episode_index + 1) % print_every == 0:
            recent_rewards = reward_history[-print_every:]
            mean_recent_reward = float(np.mean(recent_rewards))
            print(
                "Episode",
                episode_index + 1,
                "mean_player_0_reward",
                round(mean_recent_reward, 4),
                "loss",
                round(update_metrics["total_loss"], 4),
            )
        episode_index += 1

    training_env.close()
    return actor_critic_agent, reward_history


def run_sample_game(trained_agent: SelfPlayActorCriticAgent, seed: int = 999) -> Dict[str, float]:
    evaluation_env = create_env(render_mode="human")
    evaluation_env.reset(seed=seed)
    cumulative_rewards = {"player_0": 0.0, "player_1": 0.0}

    print("\nSample run with board states:\n")
    evaluation_env.render()
    for acting_agent in evaluation_env.agent_iter():
        observation, reward, termination, truncation, _ = evaluation_env.last()
        cumulative_rewards[acting_agent] += float(reward)

        if termination or truncation:
            evaluation_env.step(None)
            continue

        action_index, _, _, _ = trained_agent.select_action(observation)
        evaluation_env.step(action_index)
        evaluation_env.render()

    evaluation_env.close()
    return cumulative_rewards


if __name__ == "__main__":
    torch.manual_seed(123)
    np.random.seed(123)

    trained_model, training_rewards = run_training(episode_count=300, print_every=25, seed=21)
    final_rewards = run_sample_game(trained_model, seed=1001)

    print("\nFinal cumulative reward from sample run:")
    print("player_0:", round(final_rewards["player_0"], 4))
    print("player_1:", round(final_rewards["player_1"], 4))
