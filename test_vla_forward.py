import time

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

import matplotlib.pyplot as plt


@torch.inference_mode()
def main() -> None:
    torch.cuda.empty_cache()

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

    lang = "pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button"
    prompt = get_openvla_prompt(lang)
    print(colored(f"Instruction prompt: \n{prompt}", "green"))


    zero_action = np.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float64)

    # already in RGB format
    image = np.load("image.npy")
    plt.figure(1)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    image = Image.fromarray(image, mode='RGB')

    # === 8-BIT/4-BIT QUANTIZATION MODE ===
    inputs = processor(prompt, image).to(device, dtype=torch.float16)

    # Run OpenVLA Inference
    start_time = time.time()
    vla_action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    # vla_action = zero_action.copy()
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

    print(f"Action: {action}")


if __name__ == "__main__":
    main()
