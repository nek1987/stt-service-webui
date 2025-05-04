
# STT Service with VLLM and Web UI

This repository contains all the necessary files to deploy an Automatic Speech Recognition (ASR) service based on the `islomov/navaistt_v1_medium` model using VLLM, along with a simple web interface built with Gradio.

The deployment is designed to be performed using Docker Compose and is optimized for use with Coolify v4.

**Repository Structure:**

STT-VLLM-WEBUI/
├── docker-compose.yml          # Main Docker Compose file
├── vllm-service/               # Files for the VLLM service
│   ├── Dockerfile              # Custom Dockerfile for VLLM with audio support
│   └── .dockerignore
├── webui-service/              # Files for the web UI service
│   ├── Dockerfile
│   ├── app.py                  # Gradio UI Python script
│   ├── requirements.txt        # UI dependencies
│   └── .dockerignore
└── README.md                   # This README file


**Prerequisites:**

1.  A server with Docker and Docker Compose installed.
2.  NVIDIA drivers and NVIDIA Container Toolkit installed on the server.
3.  Coolify v4 installed and running on your server.
4.  An NVIDIA GPU with sufficient VRAM for the Whisper medium model (approx. 10-12GB VRAM or more recommended).
5.  A GitHub account and a clone of this repository pushed to your account.

**Deployment using Coolify v4:**

1.  **Create Repository:** Clone this repository and push it to your GitHub account.
2.  **Log in to Coolify:** Open the Coolify web interface.
3.  **Create New Resource:** Click on "Add new Resource".
4.  **Select Application Type:** Choose "Compose".
5.  **Configure Source:** Select "By pulling a public repository".
6.  **Enter Repository Details:**
    * **Repository:** Enter the name of your GitHub repository (e.g., `your_github_username/your-stt-repo`).
    * **Branch:** Select the branch (e.g., `main`).
    * Coolify might require integrating with your GitHub account if you haven't done so already.
7.  **Configure Environment Variables (Optional, Recommended):**
    Navigate to the "Environment Variables" section for your Compose resource in Coolify. Although variables for the UI are set in `docker-compose.yml`, you can override them here or add the API key for VLLM if required.
    * `VLLM_API_BASE`: The address of the VLLM service. In our `docker-compose.yml`, this is set to `http://stt-service:8000/v1`, which works within the Docker network.
    * `API_KEY`: API key for VLLM (if configured on the VLLM service).
    * `MODEL_NAME`: The model name (defaults to `islomov/navaistt_v1_medium`).
8.  **Configure Ports:** Coolify will scan your `docker-compose.yml`. Ensure that for the `stt-webui` service, the internal Container Port is set to `7860`. Configure the Public Port and/or domain for accessing the web interface externally. The `stt-service` (VLLM) does not expose ports to the host by default in this Compose file; communication happens within the Docker network.
9.  **Deploy:** Click the "Deploy" button. Coolify will pull the repository, build the Docker images for both services (VLLM with audio support and the UI), and then run them. This might take some time on the first deployment (especially downloading the VLLM model weights).

**Using the Web Interface:**

After successful deployment, access the address provided by Coolify for your `stt-webui` service in your web browser. You will see a simple interface where you can upload audio files and get their text transcription.

**Troubleshooting:**

* **Check Coolify Logs:** In the Coolify interface for your Compose resource, you can view the logs for each service (`stt-service` and `stt-webui`). This is the first step for debugging.
* **GPU Errors:** If the `stt-service` fails to start, check its logs for GPU-related errors. Ensure that NVIDIA drivers and the NVIDIA Container Toolkit are correctly installed on the Coolify server.
* **UI Connection Errors:** If the `stt-webui` service starts but transcription doesn't work, check its logs for errors related to connecting to the `VLLM_API_BASE` address. Ensure the `VLLM_API_BASE` environment variable is set correctly. If you changed the VLLM service name in `docker-compose.yml`, update it in the `VLLM_API_BASE` for the UI as well.

Good luck with your deployment!
```