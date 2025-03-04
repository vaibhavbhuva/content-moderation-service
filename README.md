## FastAPI Moderation APIs

This project provides a FastAPI application that integrates with Google Cloud Natural Language API and Google Cloud Vision API for text and image moderation.

### Prerequisites

* Python 3.12+
* Google Cloud Platform (GCP) account with the necessary APIs enabled (Natural Language API and Vision API).
* A service account key file (JSON) with appropriate permissions.
* Docker (optional, for containerization).
* Virtual environment (recommended).

### Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/KB-iGOT/profanity-moderation.git
    cd profanity-moderation
    ```

2.  **Create a virtual environment:**

    ```bashs
    python3.12 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install uv && uv pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**

    * Create a `.env` file in the root directory of the project.
    * Add the following line, replacing the path with your service account key file path:

        ```text
        LOG_LEVEL=INFO
        GOOGLE_APPLICATION_CREDENTIALS=<YOUR_SERVICE_ACCOUNT_KEY_FILE_PATH>
        ```

5.  **Running the Application:**

    ```bash
    uvicorn src.main:app --reload
    ```

6.  **Access the API:**

    * The API will be available at `http://127.0.0.1:8000/docs`.

### API Endpoints

* **`/moderate_text/` (POST):** Moderates text.
    * Request body: `{"text": "Your text here"}`
    * Response: JSON array of category and confidence scores.
* **`/moderate_image/` (POST):** Detects unsafe features in an image.
    * Request: Upload an image file.
    * Response: JSON object with safe search results.

### Docker

1.  **Build the Docker image:**

    ```bash
    docker build -t moderation-api .
    ```

2.  **Run the Docker container:**

    ```bash
    docker run -p 8000:8000 -e LOG_LEVEL=INFO -e GOOGLE_APPLICATION_CREDENTIALS=/app/prj-demo-fe2334234.json -v $(pwd)/prj-demo-fe2334234.json:/app/prj-demo-fe2334234.json moderation-api
    ```

    * **Important:** Replace `prj-demo-fe2334234.json` with your actual service account key file name.
    * The `-v` flag mounts the service account key file into the container.
    * Or, use a docker secret or other method to securely provide your credentials.

### Error Handling

* The API returns HTTP 500 errors with detailed error messages in case of exceptions.
* Vision API errors are also handled and returned as part of the response.

### Using UV

This project now utilizes the `uv` package for faster dependency installation. `uv` is significantly faster than `pip` and is recommended for production deployments.

### .env file

The .env file is used to store environment variables, such as the path to your service account key file. This allows you to keep sensitive information out of your code.