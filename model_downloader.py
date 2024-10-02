import logging
import asyncio
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def check_and_install_packages():
    import importlib
    required_packages = ["aiohttp", "tqdm"]
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            logger.info(f"Installing package: {package}")
            subprocess.run(["pip", "install", package])


check_and_install_packages()


import aiohttp  # noqa: E402
from tqdm import tqdm  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent

COMFYUI_DIR = BASE_DIR / "ComfyUI"
MODEL_CHECKPOINTS_DIR = COMFYUI_DIR / "models" / "checkpoints"

FLUX_DEV_URL = "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"
FLUX_DEV_NAME = "flux1-dev-fp8.safetensors"
FLUX_SCHNELL_URL = "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors"
FLUX_SCHNELL_NAME = "flux1-schnell-fp8.safetensors"


async def download_model(url: str, destination_path: Path):
    try:
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(
            trust_env=True,
            connector=aiohttp.TCPConnector(ssl=False),
            timeout=timeout,
        ) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    total_size = int(response.headers.get("Content-Length", 0))
                    chunk_size = 1024
                    with tqdm(
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        desc=destination_path.name,
                    ) as bar:
                        with destination_path.open("wb") as f:
                            async for chunk in response.content.iter_chunked(
                                chunk_size
                            ):
                                f.write(chunk)
                                bar.update(len(chunk))
                else:
                    logger.error(
                        f"Failed to download {url}. "
                        f"Status code: {response.status}"
                    )
    except aiohttp.ClientError as e:
        logger.error(f"Failed to download {url}. Error: {e}")
    except asyncio.TimeoutError:
        logger.error(
            "Download timed out. Please check your internet "
            "connection and try again."
        )


async def download_flux_dev():
    dest_path = MODEL_CHECKPOINTS_DIR / FLUX_DEV_NAME
    if dest_path.exists():
        logger.info("Flux Dev already downloaded.")
        return
    logger.info("Downloading Flux Dev...")
    await download_model(FLUX_DEV_URL, dest_path)


async def download_flux_schnell():
    dest_path = MODEL_CHECKPOINTS_DIR / FLUX_SCHNELL_NAME
    if dest_path.exists():
        logger.info("Flux Schnell already downloaded.")
        return
    logger.info("Downloading Flux Schnell...")
    await download_model(FLUX_SCHNELL_URL, dest_path)


async def main():
    if not COMFYUI_DIR.exists():
        logger.error("ComfyUI directory not found.")
        return

    if not MODEL_CHECKPOINTS_DIR.exists():
        logger.error("Model checkpoints directory not found.")
        return

    flux_dev_exists = (MODEL_CHECKPOINTS_DIR / FLUX_DEV_NAME).exists()
    flux_schnell_exists = (MODEL_CHECKPOINTS_DIR / FLUX_SCHNELL_NAME).exists()

    while True:
        menu = (
            "Select models to download:\n"
            "1. Flux Dev (17.2G)\n"
            "2. Flux Schnell (17.2G)\n"
            "3. All (34.4G)\n"
        )
        if flux_dev_exists or flux_schnell_exists:
            menu += "4. Reinstall existing models\n"

        print(menu)
        if flux_dev_exists or flux_schnell_exists:
            valid_choices = ["1", "2", "3", "4"]
        else:
            valid_choices = ["1", "2", "3"]

        choice = input(f"Enter your choice ({'/'.join(valid_choices)}): ")
        if choice in valid_choices:
            break
        else:
            print(f"Invalid choice. Please enter {'/'.join(valid_choices)}.")

    if choice == "1":
        await download_flux_dev()
    elif choice == "2":
        await download_flux_schnell()
    elif choice == "3":
        await asyncio.gather(download_flux_dev(), download_flux_schnell())
    elif choice == "4":
        tasks = []
        if flux_dev_exists:
            logger.info("Reinstalling Flux Dev...")
            (MODEL_CHECKPOINTS_DIR / FLUX_DEV_NAME).unlink()
            tasks.append(download_flux_dev())
        if flux_schnell_exists:
            logger.info("Reinstalling Flux Schnell...")
            (MODEL_CHECKPOINTS_DIR / FLUX_SCHNELL_NAME).unlink()
            tasks.append(download_flux_schnell())
        await asyncio.gather(*tasks)

    logger.info("Download completed!")

if __name__ == "__main__":
    asyncio.run(main())
