import random
import imageio

# from matplotlib import pyplot as plt
import numpy as np
from termcolor import cprint
from ledwm.embodied.envs.LWMSent import LWMSent
from messenger.envs.config import (
    ENTITY_IDS,
    MOVEMENT_IDS,
    ROLE_IDS,
    WITH_MESSAGE,
    ALL_ENTITIES,
    STATE_HEIGHT,
    STATE_WIDTH,
)


def get_avatar_id(obs):
    return obs["avatar_ids"][0]


def get_avatar_pos(obs):
    return obs["avatar_pos"][0][:2]


def get_entity_id_by_role(parsed_manual, role):
    for e in parsed_manual:
        if e[2] == role:
            return ENTITY_IDS[e[0]]
    return None


def get_position_by_id(obs, id):
    index_in_entity_ids = np.where(obs["entity_ids"] == id)[0][0]
    return obs["entity_pos"][index_in_entity_ids][:2]


def out_of_bounds(x):
    return x[0] < 0 or x[0] >= 10 or x[1] < 0 or x[1] >= 10


def get_distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


INTENTIONS = [
    "random",
    "suicide",
    "survive",
    "get_message",
    "go_to_goal",
    "suicide_go_to_goal",
]


class OracleAgent:
    ACTIONS = [
        (0, 0, -1),
        (1, 0, 1),
        (2, -1, 0),
        (3, 1, 0),
        (4, 0, 0),
    ]
    ACTION_NAMES = ["left", "right", "up", "down", "stay"]

    def __init__(self, seed=1000):
        self.random = random.Random(seed + 52398)

    def get_best_action_for_surviving(self, a_pos, e_pos, g_pos):
        distance_to_enemy = get_distance(a_pos, e_pos)
        if g_pos is not None:
            distance_to_goal = get_distance(a_pos, g_pos)
        else:
            distance_to_goal = 1e9

        # print(
        #     f"distance_to_enemy: {distance_to_enemy}, distance_to_goal: {distance_to_goal}"
        # )
        # if far enough from enemy and goal just act randomly
        SAFE_DISTANCE = 6
        if distance_to_enemy >= SAFE_DISTANCE and distance_to_goal >= SAFE_DISTANCE:
            return self.random.choice(range(len(self.ACTIONS)))

        # otherwise, stay further from both
        best_d = -1e9
        best_a = None

        # shuffle action order to randomize choice
        for a, dr, dc in self.random.sample(self.ACTIONS, len(self.ACTIONS)):
            na_pos = (a_pos[0] + dr, a_pos[1] + dc)
            if out_of_bounds(na_pos):
                continue

            d = get_distance(na_pos, e_pos)
            if g_pos is not None:
                d = min(d, get_distance(na_pos, g_pos))

            if d >= SAFE_DISTANCE / 2 or d > best_d:
                best_d = d
                best_a = a

        # print(f"best_d: {best_d}, best_a: {best_a}")
        assert best_a is not None, (a_pos, e_pos)
        return best_a

    def get_best_action_for_chasing(self, a_pos, t_pos):
        best_d = 1e9
        best_a = None
        # shuffle action order to randomize choice
        for a, dr, dc in self.random.sample(self.ACTIONS, len(self.ACTIONS)):
            na_pos = (a_pos[0] + dr, a_pos[1] + dc)
            if out_of_bounds(na_pos):
                continue
            d = get_distance(na_pos, t_pos)
            if d < best_d:
                best_d = d
                best_a = a

        assert best_a is not None, (a_pos, t_pos)
        return best_a

    def act(self, obs, parsed_manual, intention, state=None, mode="train", step=None):
        if intention == "random":
            return self.random.choice(range(len(self.ACTIONS)))

        elif intention == "survive":
            avatar_id = get_avatar_id(obs)
            assert avatar_id in [15, 16], "avatar_id must be 15 or 16"
            a_pos = get_avatar_pos(obs)

            enemy_id = get_entity_id_by_role(parsed_manual, "enemy")
            e_pos = get_position_by_id(obs, enemy_id)
            goal_id = get_entity_id_by_role(parsed_manual, "goal")
            g_pos = get_position_by_id(obs, goal_id)
            # if message has been obtained, don't care about hitting goal
            if avatar_id == WITH_MESSAGE.id:
                g_pos = None
            # choose action that takes avatar furthest from the enemy
            return self.get_best_action_for_surviving(a_pos, e_pos, g_pos)

        elif intention == "suicide":
            avatar_id = get_avatar_id(obs)
            a_pos = get_avatar_pos(obs)
            enemy_id = get_entity_id_by_role(parsed_manual, "enemy")
            e_pos = get_position_by_id(obs, enemy_id)
            return self.get_best_action_for_chasing(a_pos, e_pos)

        elif intention == "suicide_go_to_goal":
            avatar_id = get_avatar_id(obs)
            a_pos = get_avatar_pos(obs)
            goal_id = get_entity_id_by_role(parsed_manual, "goal")
            g_pos = get_position_by_id(obs, goal_id)
            return self.get_best_action_for_chasing(a_pos, g_pos)

        elif intention == "get_message":
            avatar_id = get_avatar_id(obs)
            # if message has been obtained, act randomly
            if avatar_id == WITH_MESSAGE.id:
                return self.random.choice(range(len(self.ACTIONS)))

            a_pos = get_avatar_pos(obs)
            message_id = get_entity_id_by_role(parsed_manual, "message")
            t_pos = get_position_by_id(obs, message_id)
            # choose action that takes avatar closest to the goal
            return self.get_best_action_for_chasing(a_pos, t_pos)

        elif intention == "go_to_goal":
            avatar_id = get_avatar_id(obs)
            a_pos = get_avatar_pos(obs)
            # if message has been obtained, go to goal
            if avatar_id == WITH_MESSAGE.id:
                goal_id = get_entity_id_by_role(parsed_manual, "goal")
                t_pos = get_position_by_id(obs, goal_id)

            # else go to message
            else:
                message_id = get_entity_id_by_role(parsed_manual, "message")
                t_pos = get_position_by_id(obs, message_id)

            # choose action that takes avatar closest to the goal
            return self.get_best_action_for_chasing(a_pos, t_pos)

        else:
            raise ValueError(f"Invalid intention: {intention}")


if __name__ == "__main__":
    env = LWMSent(task="hard", length=32, mode="train", disappear=False)
    policy = OracleAgent(seed=1000)
    obs = env.reset()
    # random choice between "go_to_goal", "random", "suicide", "get_message", "survive"
    # intention = random.choice(policy.INTENTIONS)
    intention = "go_to_goal"
    cprint(f"intention: {intention}", "red")

    # List to store observations for visualization
    obs_history = [obs]
    actions = []
    time_step = 0

    while True:
        print(f"time_step: {time_step}")
        action = policy.act(obs, env.true_parsed_manual, intention)
        time_step += 1
        print(f"action: {policy.ACTION_NAMES[action]}")
        print("")

        obs, reward, done, info = env.step(action)
        obs_history.append(obs)
        actions.append(action)
        if done:
            break

    # Visualize observations over time and create a video
    def visualize_obs(obs, step, action=None, parsed_manual=None, intention="Unknown"):
        plt.figure(figsize=(10, 10))
        # Assuming obs['image'] or some grid representation exists
        if "image" in obs:
            plt.imshow(obs["image"])
        else:
            # Placeholder for grid visualization with role-based coloring
            grid_rgb = np.zeros((STATE_HEIGHT, STATE_WIDTH, 3))  # RGB grid
            avatar_pos = get_avatar_pos(obs)
            avatar_id = get_avatar_id(obs)
            if avatar_id != 0:
                grid_rgb[avatar_pos[0], avatar_pos[1]] = [1, 1, 1]  # White for avatar
                plt.text(
                    avatar_pos[1],
                    avatar_pos[0],
                    "Avatar",
                    color="black",
                    ha="center",
                    va="center",
                )
            if parsed_manual is not None:
                for entity_id in obs["entity_ids"]:
                    if entity_id > 0:  # Assuming positive IDs are entities
                        pos = get_position_by_id(obs, entity_id)
                        entity_name = next(
                            (e.name for e in ALL_ENTITIES if e.id == entity_id),
                            "Unknown",
                        )
                        role = next(
                            (
                                e[2]
                                for e in parsed_manual
                                if ENTITY_IDS.get(e[0]) == entity_id
                            ),
                            "unknown",
                        )
                        if role == "enemy":
                            grid_rgb[pos[0], pos[1]] = [1, 0, 0]  # Red for enemy
                        elif role == "message":
                            grid_rgb[pos[0], pos[1]] = [0, 0, 1]  # Blue for messenger
                        elif role == "goal":
                            grid_rgb[pos[0], pos[1]] = [0, 1, 0]  # Green for goal
                        else:
                            grid_rgb[pos[0], pos[1]] = [
                                0.5,
                                0.5,
                                0.5,
                            ]  # Gray for unknown
                        plt.text(
                            pos[1],
                            pos[0],
                            entity_name,
                            color="white" if role != "unknown" else "black",
                            ha="center",
                            va="center",
                        )
            plt.imshow(grid_rgb)
            # Add grid lines to show cells
            plt.grid(True, which="both", color="black", linestyle="-", linewidth=0.5)
            plt.xticks(np.arange(-0.5, STATE_WIDTH, 1), labels=[])
            plt.yticks(np.arange(-0.5, STATE_HEIGHT, 1), labels=[])
        plt.title(
            f"Step {step}, Action: {action if action is not None else 'Start'}, Intention: {intention}"
        )
        plt.colorbar()
        plt.savefig(f"frame_{step:03d}.png")
        plt.close()

    # Create frames for each step
    for step, (obs, action) in enumerate(zip(obs_history, [None] + actions)):
        visualize_obs(obs, step, action, env.true_parsed_manual, intention)

    # Compile frames into a video
    frames = [
        imageio.imread(f"frame_{step:03d}.png") for step in range(len(obs_history))
    ]
    imageio.mimwrite(f"game_progression_{intention}.mp4", frames, fps=2)

    # Clean up temporary files
    import os

    for step in range(len(obs_history)):
        os.remove(f"frame_{step:03d}.png")

    cprint(f"Video created as 'game_progression_{intention}.mp4'", "green")
    print(env.true_parsed_manual)
