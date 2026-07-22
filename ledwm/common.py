import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time


class Timing:
    """
    A context manager to measure execution time of code blocks.

    Parameters:
        description (str): A description of the timed code block.
        debug (bool): If True, prints the timing output. If False, suppresses it.
    """

    def __init__(self, description="Execution", debug=True):
        self.description = description
        self.debug = debug

    def __enter__(self):
        self.start_time = time.perf_counter()  # Start timing
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()  # End timing
        execution_time = self.end_time - self.start_time
        if self.debug:
            print(f"[{self.description}] completed in {execution_time:.6f} seconds")


def get_multihot_image_from_pos(
    entity_pos,  # (Ne, 3)
    avatar_pos,  # (Ne)
    W=10,
    H=10,
    D=17,
):
    assert entity_pos.ndim == avatar_pos.ndim, (entity_pos.ndim, avatar_pos.ndim)
    image = np.zeros((W, H, D), dtype=np.float32)

    def fill_positions(image, pos):
        x, y, depth_index = pos.T
        for i in range(len(x)):
            # Skip filling if x == 10, y == 10, and depth_index == 0
            if depth_index[i] == 0:
                assert x[i] == 10 and y[i] == 10, (x[i], y[i], depth_index[i])
                continue
            image[x[i], y[i], depth_index[i]] = 1

        return image

    image = fill_positions(image, entity_pos)
    image = fill_positions(image, avatar_pos)
    return image


def log_image(
    img, step, ac=None, reward=None, done=None, reward_pred=None, cont_pred=None
):
    assert len(img.shape) == 3
    assert img.shape[2] == 17
    img = img[:10, :10]  # Remove padding

    idx_to_letter = {
        2: "A",
        3: "M",
        4: "D",
        5: "B",
        6: "F",
        7: "C",
        8: "T",
        9: "H",
        10: "B",
        11: "R",
        12: "Q",
        13: "S",
        14: "W",
        15: "a",
        16: "m",
    }
    actions = ["up", "down", "left", "right", "stay", "reset"]

    text_height_per_line = 20  # Reduced height per line of text
    # Three lines of text: action, reward, done
    total_text_height = text_height_per_line * 6

    scale = 256 / 10
    fontpath = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"
    font = ImageFont.truetype(fontpath, 12) if os.path.exists(fontpath) else None
    text_font = (
        ImageFont.truetype(fontpath, 18) if os.path.exists(fontpath) else None
    )  # Larger font for action, reward, done texts

    new_img = Image.new(
        size=(256, 256 + total_text_height), mode="RGB", color=(31, 33, 50)
    )  # Adjusted for action, reward, done texts
    draw = ImageDraw.Draw(new_img)

    # Adjust text positions to reduce the gap
    cur_y = action_y = 0
    reward_y = action_y + text_height_per_line
    cur_y += text_height_per_line
    done_y = reward_y + text_height_per_line
    cur_y += text_height_per_line

    # Draw action text if action code (ac) is provided
    # if ac is np.array then ac = argmax
    if isinstance(ac, np.ndarray):
        ac = np.argmax(ac)
    if ac is not None and 0 <= ac < len(actions):
        action_text = f"Action = {actions[ac]}"
        draw.text((10, action_y), action_text, fill=(255, 255, 255), font=text_font)

    # Draw reward text if reward is provided
    if reward is not None:
        reward_text = f"Reward = {reward}"
        draw.text((10, reward_y), reward_text, fill=(255, 255, 255), font=text_font)

    # Draw done status if done is provided
    if done is not None:
        done_text = f"Done = {done}"
        draw.text((10, done_y), done_text, fill=(255, 255, 255), font=text_font)

    # add timestep
    # if step is not None:
    step_text = f"Step = {step}"
    step_y = done_y + text_height_per_line
    cur_y += text_height_per_line
    draw.text((10, step_y), step_text, fill=(255, 255, 255), font=text_font)

    if reward is not None:
        if reward == 1:
            win_text = "You Win!"
        elif reward == -1:
            win_text = "You Lose!"
        else:
            win_text = ""
        cur_y += text_height_per_line

        draw.text(
            (10, cur_y),
            win_text,
            fill=(255, 255, 255),
            font=text_font,
        )

    if reward_pred is not None:
        reward_pred_text = f"Reward Pred = {reward_pred}"
        cur_y += text_height_per_line
        draw.text(
            (10, cur_y),
            reward_pred_text,
            fill=(255, 255, 255),
            font=text_font,
        )

    if cont_pred is not None:
        cont_pred_text = f"Cont Pred = {cont_pred}"
        cur_y += text_height_per_line
        draw.text(
            (10, cur_y),
            cont_pred_text,
            fill=(255, 255, 255),
            font=text_font,
        )

    idxs = img.argmax(-1)
    for i, row in enumerate(img):
        for j, col in enumerate(row):
            if idxs[i][j] in (0, 1):  # Skip if index is 0 or 1
                continue

            letter = idx_to_letter[idxs[i][j]]
            color = (247, 193, 119) if letter in ("a", "m") else (238, 108, 133)
            # Move the image content up closer to the text
            draw.text(
                (
                    int(j * scale),
                    int(i * scale) + total_text_height - (2 * text_height_per_line),
                ),
                letter,
                fill=color,
                font=font,
            )

    new_img = np.asarray(new_img)
    return new_img


def print_image_error_file(data, pred, loss):
    # pred = pred.reshape((-1, *pred.shape[2:])).astype(np.int32)
    # data = data.reshape((-1, *data.shape[2:])).astype(np.int32)
    file = open("error.txt", "w")
    file.write(f"loss={loss}\n")
    for t in range(pred.shape[0]):
        file.write(f"Image {t}\n")
        nonzeros = np.nonzero(pred[t] - data[t])
        for i, j, k in zip(*nonzeros):
            file.write(
                f"Pixel {i}, {j}, {k}: {pred[t][i][j][k]} vs {data[t][i][j][k]}\n"
            )


def symbolic_to_multihot(layers):
    n_entities = 17
    layers = layers.astype(int)
    new_ob = np.maximum.reduce(
        [np.eye(n_entities)[layers[..., i]] for i in range(layers.shape[-1])]
    )
    new_ob[:, :, 0] = 0
    return new_ob


def image_error2str(data, pred, action, is_first):
    res = ""
    assert data.shape == pred.shape, f"{data.shape=}, {pred.shape=}"
    for t in range(min(pred.shape[0], NUM_TABLE_STEPS)):
        # gt = symbolic_to_multihot(data[t])
        gt = data[t]
        nonzeros = np.nonzero(pred[t] - gt)
        if action is not None:
            # argmax of action
            argmax_action = np.argmax(action[t])
        else:
            argmax_action = None
        res += f"Image {t} - is_first {is_first[t]} - nonzero = {len(nonzeros[0])} - action {argmax_action}\n"
        for i, j, k in list(zip(*nonzeros))[:N_PIXELS]:
            res += (
                f"Pixel {i}, {j}, {k}: pred={pred[t][i][j][k]} vs data={gt[i][j][k]}\n"
            )
        res += "\n"
    return res


N_PIXELS = 20
NUM_TABLE_STEPS = 40
ACTIONS = ["up", "down", "left", "right", "stay", "reset"]

IDX_TO_ENTITY_NAME = {
    2: "airplane",
    3: "mage",
    4: "dog",
    5: "bird",
    6: "fish",
    7: "scientist",
    8: "thief",
    9: "ship",
    10: "ball",
    11: "robot",
    12: "queen",
    13: "sword",
    14: "wall",
    15: "player",
    16: "player",
}


def atten2str(
    atten,  # bs, nhead, ne, ne
    entity_ids,  # bs, ne
    sent_ids,  # bs, ne
    id2sent,  # sent id to sentj
    is_first,
    rewards,
    entity_pos,  # bs, ne, 3
    avatar_pos,  # bs, 3
):
    res = ""
    for t in range(min(atten.shape[0], NUM_TABLE_STEPS)):
        res += f"Time {t}, {is_first[t]=}, rewards = {rewards[t]} \n"
        res += f"Avatar: {avatar_pos[t]}\n"

        for h in range(atten.shape[1]):
            res += f"Head {h}\n"
            for i in range(atten.shape[2]):
                entity_id = entity_ids[t][i]
                if entity_id == 0:
                    res += f"Entity 0: {entity_pos[t][i]} \n"
                else:
                    res += f"Entity {IDX_TO_ENTITY_NAME[entity_ids[t][i]]}: {entity_pos[t][i]} \n"
                    for j in range(atten.shape[3]):
                        res += (
                            f"Attens: {atten[t][h][i][j]} - {id2sent[sent_ids[t][j]]}\n"
                        )
                    res += "\n"
            res += "\n"
    return res


# def avatar_entity2str(
#     entity_ids,  # bs, ne
#     entity_pos,  # bs, ne, 3
#     avatar_pos,  # bs, 3,
#     rewards=None,  # bs
#     is_first=None,  # bs
#     cont=None,  # bs
#     action=None,  # bs, 6
# ):
#     res = ""
#     for t in range(min(entity_pos.shape[0], NUM_TABLE_STEPS)):
#         res += f"Time {t}"
#         if rewards is not None:
#             res += f", Reward: {rewards[t]}"
#         if is_first is not None:
#             res += f", is_first: {is_first[t]}"
#         if cont is not None:
#             res += f", Cont: {cont[t]}"
#         if action is not None:
#             argmax_action = np.argmax(action[t])
#             res += f", Action: {ACTIONS[argmax_action]}"
#         res += "\n"

#         for i in range(entity_pos.shape[1]):
#             if entity_ids[t][i] == 0:
#                 res += f"Entity 0: \n"
#             else:
#                 entity_id = entity_ids[t][i]
#                 res += f"Entity {IDX_TO_ENTITY_NAME[entity_id]}: {entity_pos[t][i]}\n"

#         res += f"Avatar: {avatar_pos[t]}\n"

#         res += "\n"
#     return res


# def avatar2str(
#     avatar_pos,  # bs, 3
# ):
#     res = ""
#     for t in range(min(avatar_pos.shape[0], 30)):
#         res += f"Time {t}: {avatar_pos[t]}\n"
#     return res


def image2str(image, is_first=None, action=None, reward=None, cont=None, step=None):
    res = ""
    if is_first is not None:
        assert image.shape[0] == is_first.shape[0]
    if action is not None:
        assert image.shape[0] == action.shape[0]
    if reward is not None:
        assert image.shape[0] == reward.shape[0]
    if cont is not None:
        assert image.shape[0] == cont.shape[0]
    if step is not None:
        assert image.shape[0] == step.shape[0]

    for t in range(min(image.shape[0], NUM_TABLE_STEPS)):
        # print all nonzero pixels
        # format: (array([i, i, ...]), array([j, j, ...]), array([k, k, ...]))
        nonzeros = np.nonzero(image[t])
        nonzeros_count = len(nonzeros[0])
        res += f"Image {t},nonzero={nonzeros_count}"
        if reward is not None:
            res += f", reward={reward[t]:.2f}"
        if cont is not None:
            res += f", cont={cont[t]}"
        if is_first is not None:
            res += f", is_first={is_first[t]}"
        if step is not None:
            res += f", step={step[t]}"
        if action is not None:
            # argmax of action
            argmax_action = np.argmax(action[t])
            res += f", action={ACTIONS[argmax_action]}\n"
        else:
            res += "\n"
        # zip the three arrays to get the coordinates of nonzero pixels
        for i, j, k in list(zip(*nonzeros))[:N_PIXELS]:
            res += f"Pixel {i}, {j}, {k}, {image[t][i, j, k]}\n"
        res += "\n"
    return res


def print_image_file(image, tag, loss=0):
    # image: (bs, bl, h, w, 17)
    # merge the first two dimensions, then for each pixel, print all channels in a row
    image = image.reshape((-1, *image.shape[2:])).astype(np.int32)
    # image: (bs*bl, h, w, 17)
    file = open(f"image_{tag}.txt", "w")
    file.write(f"loss={loss}\n")
    for t in range(image.shape[0]):
        file.write(f"Image {t}\n")
        # print all nonzero pixels
        nonzeros = np.nonzero(
            image[t]
        )  # format: (array([i, i, ...]), array([j, j, ...]), array([k, k, ...]))
        # zip the three arrays to get the coordinates of nonzero pixels
        for i, j, k in zip(*nonzeros):
            file.write(f"Pixel {i}, {j}, {k}\n")

        # for each pixel, print all channels in a row
        # print to file instead of console
        for i in range(image.shape[1]):
            for j in range(image.shape[2]):
                if sum(image[t][i][j]) > 0:
                    file.write(f"Pixel {i}, {j}: {image[t][i][j]}\n")
