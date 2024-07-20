import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

import robosuite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from termcolor import colored

# required to register the environments defined in robocasa
import robocasa


@torch.inference_mode()
def main() -> None:
    MODEL_PATH = "openvla/openvla-7b"
    SYSTEM_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )

    def get_openvla_prompt(instruction: str) -> str:
        if "v01" in MODEL_PATH:
            return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
        else:
            return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    print(f"[*] Verifying OpenVLAForActionPrediction using Model `{MODEL_PATH}`")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    proxies = {'http': 'http://127.0.0.1:7890',
               'https': 'http://127.0.0.1:7890'}

    # Load Processor & VLA
    print("[*] Instantiating Processor and Pretrained OpenVLA")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, proxies=proxies)

    # === 4-BIT QUANTIZATION MODE (`pip install bitsandbytes`) :: [~6GB of VRAM Passive || 7GB of VRAM Active] ===
    print("[*] Loading in 4-Bit Quantization Mode")
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        proxies=proxies,
    )


    camera_names =[
            "robot0_robotview",
            "robot0_agentview_center",
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_frontview",
            "robot0_eye_in_hand",
    ]

    env = robosuite.make(
        env_name="PnPCounterToSink",
        robots="PandaMobile",
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        layout_ids=0,
        style_ids=8,
        translucent_robot=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_eye_in_hand",
        ignore_done=True,
        use_camera_obs=True,
        # simulation in sync mode. this means every step takes 0.1s simulation time to complete, actually it depends
        # on openvla inference time, currently is like 0.7s wall time per step.
        # in other word, the simulation time in the real world is 10 times slower than the simulation time in the
        # simulation world. 0.1s simulation time takes 0.7s wall time.
        control_freq=10,
        renderer="mjviewer",
        camera_names=camera_names,
    )

    env = VisualizationWrapper(env)

    ep_meta = env.get_ep_meta()
    lang = ep_meta.get("lang", None)
    prompt = get_openvla_prompt(lang)
    # prompt = prompt.split('pick the')
    # prompt = f'{prompt[0]}pick the knife from{prompt[1].split("from")[1]}'
    print(colored(f"Instruction prompt: \n{prompt}", "green"))

    obs = env.reset()

    zero_action = np.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)
    env.step(zero_action)

    while True:
        # image = obs["robot0_agentview_center_image"]
        # image = obs["robot0_agentview_left_image"]
        image = obs["robot0_eye_in_hand_image"]
        # plt.figure(1)
        # plt.imshow(image)
        # plt.axis("off")
        # plt.show()

        # openGL to openCV conversion
        image = np.flipud(image)
        # plt.figure(2)
        # plt.imshow(image)
        # plt.axis("off")
        # plt.show()

        # cv2.imshow("robot view", image)

        # np.save("image.npy", image)
        # break

        # already in RGB format
        image = Image.fromarray(image, mode='RGB')

        # === 8-BIT/4-BIT QUANTIZATION MODE ===
        inputs = processor(prompt, image).to(device, dtype=torch.float16)

        # Run OpenVLA Inference
        start_time = time.time()
        vla_action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        print(f"\t=>> Time: {time.time() - start_time:.4f} || Action: {vla_action}")

        # to clear the motion of the base and the torso
        action = zero_action.copy()

        arm_actions = vla_action[:6]
        # TODO(kun): check if need to flip some actions
        # arm_actions[0], arm_actions[1] = arm_actions[1], -arm_actions[0]
        # arm_actions[3], arm_actions[4] = arm_actions[4], -arm_actions[3]

        action[:6] = arm_actions
        # gripper action
        action[6] = vla_action[-1]

        obs, _, _, _ = env.step(action)

        if env._check_success():
            print(colored("Task completed successfully!", "green"))
            break

    env.close()


if __name__ == "__main__":
    main()
