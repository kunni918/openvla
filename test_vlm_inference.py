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
    # MODEL_PATH = "prismatic-vlms/prism-dinosiglip+7b"

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

    prompt = "what objects are in the image? and where they are? "
    prompt = f"In: {prompt}\nOut:"
    print(colored(f"Instruction prompt: \n{prompt}", "green"))

    # already in RGB format
    image = np.load("image.npy")
    plt.figure(1)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

    image = Image.fromarray(image, mode='RGB')
    image.save("saved_image.png", "PNG")

    # === 8-BIT/4-BIT QUANTIZATION MODE ===
    inputs = processor(prompt, image).to(device, dtype=torch.float16)

    # We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
    # in order for the predictions to match the training configuration and be accurate.
    inputs['input_ids'] = torch.cat(
        (inputs['input_ids'], torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(inputs['input_ids'].device)), dim=1
    )

    generated_ids = vla.generate(**inputs, max_new_tokens=512)

    generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    print(colored(f"Generated text: \n{generated_text}", "green"))


if __name__ == "__main__":
    main()
