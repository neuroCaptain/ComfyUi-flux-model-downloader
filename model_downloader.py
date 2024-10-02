# Credits: https://comfyanonymous.github.io/ComfyUI_examples/flux/
# Script made by: Neuro Captain

import os
import logging
import asyncio
import subprocess

logger = logging.getLogger(__name__)

def check_and_install_packages():
    import importlib
    required_packages = ['aiohttp', 'tqdm']
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            logger.info(f"Installing package: {package}")
            subprocess.run(["pip", "install", package])

check_and_install_packages()

import aiohttp
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COMFYUI_DIR = os.path.join(BASE_DIR, "ComfyUI")
MODEL_CHECKPOINTS_DIR = os.path.join(COMFYUI_DIR, "models", "checkpoints")

FLUX_DEV_URL = "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
FLUX_DEV_NAME = "flux1-dev-fp8.safetensors"
FLUX_SCHNELL_URL = "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors"
FLUX_SCHNELL_NAME = "flux1-schnell-fp8.safetensors"


async def download_model(url: str, destination_path: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                total_size = int(response.headers.get('Content-Length', 0))
                chunk_size = 1024
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(destination_path)) as bar:
                    with open(destination_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            bar.update(len(chunk))
            else:
                logger.error(f"Failed to download {url}. Status code: {response.status}")


async def download_flux_dev():
    dest_path = os.path.join(MODEL_CHECKPOINTS_DIR, FLUX_DEV_NAME)
    if os.path.exists(dest_path):
        logger.info(f"Flux Dev already downloaded.")
        return
    logger.info(f"Downloading Flux Dev...")
    await download_model(FLUX_DEV_URL, dest_path)


async def download_flux_schnell():
    dest_path = os.path.join(MODEL_CHECKPOINTS_DIR, FLUX_SCHNELL_NAME)
    if os.path.exists(dest_path):
        logger.info(f"Flux Schnell already downloaded.")
        return
    logger.info(f"Downloading Flux Schnell...")
    await download_model(FLUX_SCHNELL_URL, dest_path)


async def main():
    if not os.path.exists(COMFYUI_DIR):
        logger.error(f"ComfyUI directory not found.")
        return

    while True:
        print("Select models to download:")
        print("1. Flux Dev (17.2G)")
        print("2. Flux Schnell (17.2G)")
        print("3. All (34.4G)")
        choice = input("Enter your choice (1/2/3): ")
        if choice in ["1", "2", "3"]:
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    if choice == '1':
        await download_flux_dev()
    elif choice == '2':
        await download_flux_schnell()
    elif choice == '3':
        await asyncio.gather(download_flux_dev(), download_flux_schnell())


if __name__ == '__main__':
    asyncio.run(main())