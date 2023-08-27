# Basic Image Style Transfer with TensorFlow

I've written a simple script to help me get more comfortable with TensorFlow by dabbling in this image manipulation technique.

## Requirements

Before we get into the details, make sure you have the following installed:

- TensorFlow (version 2.0 or newer)
- TensorFlow Hub
- Matplotlib
- NumPy

You can install them using the following command:

```bash
pip install tensorflow tensorflow_hub matplotlib numpy
```

## How to Use

Here's a step-by-step guide on how to use the script:

1. Download the `style_transfer.py` script or clone this repository to your local machine.

2. Open the script and locate the `content_image_url` and `style_image_url` variables. Replace them with the URLs of the content and style images you want to use.

3. Adjust the `output_image_size`, `content_img_size`, and `style_img_size` variables to customize the dimensions of the output, content, and style images.

4. Run the script using the following command:

```bash
python style_transfer.py
```

5. The script will take the content and style images, apply the style transfer process using TensorFlow Hub, and display the original content image, the style image, and the stylized output image.

## What's Happening Behind the Scenes

Let me break down the script's functions and operations for you:

- `crop_center(image)`: This function crops an image to make it square by focusing on the center.

- `load_image(image_url, image_size, preserve_aspect_ratio)`: Here, we load and prepare an image from a given URL. The image is cropped and resized as needed.

- `show_n(images, titles)`: A utility function that uses Matplotlib to display multiple images side by side.

The script loads the content and style images and prepares them for processing. It uses a pre-trained style transfer model from TensorFlow Hub to apply the style of the style image to the content image. The result is a stylized image that combines both styles.

## Why Did I Write This?

Just a little exercise to get more familiar with TensorFlow. It's a simple script for a simple task.

![image](https://user-images.githubusercontent.com/79209089/155845770-27c9e765-ab1f-4817-96c5-f10fc474a33e.png)
