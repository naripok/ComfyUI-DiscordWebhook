import asyncio
import os
import shutil
import tempfile

import numpy as np
from discord_webhook import AsyncDiscordWebhook
from PIL import Image, ImageDraw


def create_default_image():
    """Create a simple TV test pattern image."""
    image = Image.new("RGB", (128, 128), "black")
    colors = ["white", "yellow", "cyan", "green", "magenta", "red", "blue", "black"]
    bar_width = 128 // len(colors)
    draw = ImageDraw.Draw(image)

    for i, color in enumerate(colors):
        draw.rectangle([i * bar_width, 0, (i + 1) * bar_width, 128], fill=color)

    return image


class DiscordPostViaWebhook:
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Discord"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "subtitle": ("STRING", {"default": "", "multiline": True}),
            },
        }

    async def send_webhook(self, url, message, files=None):
        """Send the webhook with the given message and up to 4 files."""
        webhook = AsyncDiscordWebhook(url=url, content=message[:2000], timeout=30.0)
        if files:
            for file in files:
                webhook.add_file(file=file["data"], filename=file["name"])
        await webhook.execute()

    def process_image(self, image):
        """Process the image (or batch of images) and return them in a format suitable for Discord."""
        if image is None:
            image = create_default_image()

        images_to_send = []

        # Check if it's a batched image (4D array: [batch_size, height, width, channels])
        if isinstance(image, np.ndarray):
            if image.ndim == 4:
                # Batch of images, process each image separately
                for i in range(image.shape[0]):
                    img = image[i]  # Don't squeeze unless we know the axis has size 1
                    img = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
                    images_to_send.append(img)
            elif image.ndim == 3:
                # Single image (3D array: [height, width, channels])
                image = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))
                images_to_send.append(image)
            else:
                raise ValueError(
                    "Input image array must be 3D or 4D (batch of images)."
                )

        elif hasattr(image, "cpu"):
            array = image.cpu().numpy()

            # Handle batched images
            if array.ndim == 4:  # Batch of images
                for i in range(array.shape[0]):
                    img_array = array[
                        i
                    ]  # Only access the i-th image, no squeeze needed
                    img = Image.fromarray(
                        np.clip(img_array * 255, 0, 255).astype(np.uint8)
                    )
                    images_to_send.append(img)
            elif array.ndim == 3:
                # Single image
                array = np.clip(array * 255, 0, 255).astype(np.uint8)
                images_to_send.append(Image.fromarray(array))
            else:
                raise ValueError("Input tensor must be 3D or 4D (batch of images).")

        # Save each image to a temporary file and collect file data
        files = []
        temp_dir = tempfile.mkdtemp()

        for idx, img in enumerate(images_to_send):
            file_path = os.path.join(temp_dir, f"image_{idx}.png")
            img.save(file_path, format="PNG", compress_level=1)

            # If the file size exceeds 20MB, resize and save again
            if os.path.getsize(file_path) > 20 * 1024 * 1024:
                img = img.resize((img.width // 2, img.height // 2))
                img.save(file_path, format="PNG", compress_level=9)

            with open(file_path, "rb") as f:
                files.append({"data": f.read(), "name": f"image_{idx}.png"})

        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

        return files

    def execute(self, image, subtitle=""):
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            raise ValueError("DISCORD_WEBHOOK_URL environment variable is not set.")

        files = self.process_image(image)

        if files:
            # Split files into batches of 4 (Discord limit)
            batches = [files[i : i + 4] for i in range(0, len(files), 4)]

            # Send multiple webhooks if necessary
            for batch in batches:
                asyncio.run(self.send_webhook(webhook_url, subtitle, batch))
        else:
            # No images to send, just send the message
            asyncio.run(self.send_webhook(webhook_url, subtitle))

        return (image,)


NODE_CLASS_MAPPINGS = {"DiscordPostViaWebhook": DiscordPostViaWebhook}

NODE_DISPLAY_NAME_MAPPINGS = {"DiscordPostViaWebhook": "Use Discord Webhook"}
