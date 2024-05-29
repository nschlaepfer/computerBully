
# Compute Bully 

## Tech Stack
- Python
- OpenCV
- LangChain
- Sentence Transformers
- Groq API
- Play.ht API
- Pydub
- Pyaudio
- dotenv

## Setup

1. **Clone the repository:**

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up the environment variables:**
    - Create a `.env` file in the root directory of the project.
    - Add the following environment variables to the `.env` file:
        ```
        GROQ_API_KEY=<your-groq-api-key>
        PLAYHT_API_KEY=<your-playht-api-key>
        PLAYHT_USER_ID=<your-playht-user-id>
        ```

5. **Install the Ollama model for VLLM:**
    ```sh
    ollama pull 0ssamaak0/xtuner-llava:phi3-mini-int4
    ```

6. **Get API keys:**
    - **Groq API:** (FREE)
    - **Play.ht API:** (5 dollars)
        - Clone the voice of your choosing.

## Usage

1. **Run the main script:**
    ```sh
    python bullyAI.py
    ```

2. **Interact with the application:**
    - The application will capture images from your webcam periodically.
    - It will generate captions for the images using the Ollama model.
    - The captions will be processed by the Groq API to generate a narrative response.
    - The narrative response will be converted to speech using the Play.ht API and played back.

3. **Stop the application:**
    - Press `q` in the webcam preview window to stop the application.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## Acknowledgements

- [LangChain](https://github.com/hwchase17/langchain)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq](https://www.groq.com/)
- [Play.ht](https://play.ht/)
- [Pydub](https://github.com/jiaaro/pydub)
- [OpenCV](https://opencv.org/)
```
