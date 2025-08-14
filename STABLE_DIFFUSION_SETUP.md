# Stable Diffusion WebUI Setup Guide

This guide will help you set up the Stable Diffusion WebUI, which is required for the AI image generation feature in this application.

## Prerequisites
- Python 3.8 or later
- Git
- A machine with a CUDA-capable GPU (recommended for faster generation)

## Installation Steps

1. **Clone the Stable Diffusion WebUI repository**:
   ```bash
   git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
   cd stable-diffusion-webui
   ```

2. **Install the required dependencies**:
   - On Windows, run `webui-user.bat`
   - On Linux/macOS, run `./webui.sh`
   
   The first time you run this, it will download the required models and dependencies automatically.

3. **Start the WebUI**:
   - The script will automatically start a local web server (usually at `http://127.0.0.1:7860`)
   - Keep this terminal window open while using the application

4. **Verify the API is working**:
   - Open your web browser and go to `http://127.0.0.1:7860`
   - You should see the Stable Diffusion WebUI interface

## Integration with Doodle Recognizer

1. The Doodle Recognizer application is already configured to connect to the Stable Diffusion WebUI at `http://127.0.0.1:7860`
2. No additional configuration is needed as long as both applications are running on the same machine

## Troubleshooting

- **Out of Memory Errors**: If you encounter CUDA out of memory errors, try reducing the image size in the `app.py` file (look for `width` and `height` parameters in the `/generate_cartoon` endpoint)

- **Slow Generation**: Image generation can be slow on CPU. Using a CUDA-capable GPU is highly recommended.

- **Connection Issues**: Make sure the WebUI is running and accessible at `http://127.0.0.1:7860` before using the AI image generation feature.

## Alternative Setup (Google Colab)

If you don't have a powerful enough GPU, you can use Google Colab to run the WebUI:

1. Open this Colab notebook: [https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb](https://colab.research.google.com/github/TheLastBen/fast-stable-diffusion/blob/main/fast_stable_diffusion_AUTOMATIC1111.ipynb)
2. Follow the instructions in the notebook to start the WebUI
3. Once running, click on the public URL provided by Colab (it will look like `https://xxxx.gradio.app`)
4. Update the `STABLE_DIFFUSION_API_URL` in `app.py` to use this URL instead of `http://127.0.0.1:7860`
