# STEP 2: Load and display the three original images

from pathlib import Path

import cv2
import matplotlib.pyplot as plt


# Find the project folder and input folder
PROJECT_FOLDER = Path(__file__).parent
INPUT_FOLDER = PROJECT_FOLDER / "input"


def load_image(image_name):
    """
    Load one color image using OpenCV.

    cv2.IMREAD_COLOR loads the image with 3 channels:
    Blue, Green, and Red (BGR).
    """
    image_path = INPUT_FOLDER / image_name

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    # If OpenCV cannot read the image, image will be None.
    if image is None:
        raise ValueError(
            f"Could not load image: {image_path}\n"
            "The file may be corrupted or may not be a valid image."
        )

    return image


def print_image_information(image_name, image):
    """
    Print image shape, height, width, and number of channels.
    """
    height, width, channels = image.shape

    print(f"\nImage: {image_name}")
    print(f"Shape: {image.shape}")
    print(f"Width: {width} pixels")
    print(f"Height: {height} pixels")
    print(f"Number of channels: {channels}")


def display_images(images):
    """
    Display the three original images in one figure.
    """
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    for axis, (image_name, image) in zip(axes, images.items()):
        # Convert OpenCV BGR image to RGB for correct Matplotlib colors.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axis.imshow(image_rgb)
        axis.set_title(f"Original: {image_name}")
        axis.axis("off")

    plt.tight_layout()
    plt.show()


# Program starts here
if __name__ == "__main__":
    # Load all original color images
    images = {
        "medical.jpg": load_image("medical.jpg"),
        "dark.jpg": load_image("dark.jpg"),
        "bright.jpg": load_image("bright.jpg"),
    }

    # Print information for each image
    for image_name, image in images.items():
        print_image_information(image_name, image)

    # Display all original images
    display_images(images)