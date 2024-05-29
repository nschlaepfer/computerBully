from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
import random
import logging
import time
import math
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from PIL import Image
from io import BytesIO
import base64
import cairosvg
import ollama
import asyncio
from ollama import AsyncClient
import cv2
import threading
import subprocess
import textwrap
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import pyaudio
import io
import os
from groq import Groq
import dotenv
from openai import OpenAI
import curses
import pyautogui

dotenv.load_dotenv()

def prompt_func(data):
    text = data["text"]
    image = data["image"]
    image_part = {
        "type": "image_url",
        "image_url": f"data:image/png;base64,{image}",
    }
    content_parts = [image_part, {"type": "text", "text": text}]
    return [HumanMessage(content=content_parts)]

def generate_image_caption(image_b64, use_openai=False):
    if not image_b64:
        return "Failed to process image for captioning."
    
    if use_openai:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        },
                        {
                            "type": "text",
                            "text": "Please describe the individual or the main subject in this image in detail."
                        }
                    ]
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content
    else:
        llm = ChatOllama(model="0ssamaak0/xtuner-llava:phi3-mini-int4", temperature=0.1)
        chain = prompt_func | llm | StrOutputParser()
        query_chain = chain.invoke(
            {"text": "Describe the individual or the main subject in this image in detail.", "image": image_b64}
        )
        return query_chain

def capture_webcam_image(cap):
    success, frame = cap.read()
    if success:
        # Convert the captured frame to PNG and then to base64
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return image_base64
    else:
        return None

def capture_screen_image():
    screenshot = pyautogui.screenshot()
    buffered = BytesIO()
    screenshot.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return image_base64

def process_image(cap, caption_text, use_openai=False):
    image_b64 = capture_webcam_image(cap)
    screen_b64 = capture_screen_image()

    if image_b64:
        caption = generate_image_caption(image_b64, use_openai)
        logging.info(f"Caption: {caption}")
        caption_text.append(caption)

        if screen_b64:
            screen_caption = generate_image_caption(screen_b64, use_openai)
            logging.info(f"Screen Caption: {screen_caption}")
            caption_text.append(screen_caption)
            combined_caption = f"Webcam: {caption} | Screen: {screen_caption}"
        else:
            combined_caption = caption

        # Print the combined caption being sent to Groq
        print(f"Sending to Groq: {combined_caption}")

        # Initialize Groq client
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # Create a chat completion with the generated caption
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"You are David Attenborough. Describe the importance of what you see in this image in max 2 sentences be very mean and very rude. This is what you see through web cam and computer screen: {combined_caption}"
                }
            ],
            model="llama3-8b-8192",
        )

        # Extract the response from Groq API
        narrative_response = chat_completion.choices[0].message.content

        # Print the Groq response
        print(f"Groq Response: {narrative_response}")
        logging.info(f"Narrative Response: {narrative_response}")

        # PlayHT API integration for TTS
        url = "https://api.play.ht/api/v2/tts/stream"
        payload = {
            "text": narrative_response,
            "voice": "s3://voice-cloning-zero-shot/92ae9ddb-c01e-44ca-b01e-7a9066d175b7/alfred/manifest.json",
            "output_format": "mp3",
            "voice_engine": "PlayHT2.0"
        }
        headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
            "AUTHORIZATION": os.getenv("PLAYHT_AUTHORIZATION"),
            "X-USER-ID": os.getenv("PLAYHT_USER_ID")
        }
        response = requests.post(url, json=payload, headers=headers, stream=True)
        
        if response.ok:
            # Setup audio stream
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(2),
                            channels=2,
                            rate=12000,  # Adjusted to match the sample rate of the audio file
                            output=True)
            
            # Buffer to store audio data
            audio_buffer = io.BytesIO()
            
            # Stream audio
            for chunk in response.iter_content(chunk_size=1024):
                audio_buffer.write(chunk)
            
            audio_buffer.seek(0)  # Rewind the buffer
            audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")
            stream.write(audio_segment.raw_data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
        else:
            logging.error("Failed to generate speech from text.")
    else:
        logging.warning("Failed to capture image from webcam.")

def periodic_image_capture(cap, caption_text, use_openai=False):
    while True:
        process_image(cap, caption_text, use_openai)
        if use_openai:
            time.sleep(0.5)  # Take screenshots every 0.5 seconds when using OpenAI
        else:
            time.sleep(1)  # Default interval for other models

def select_model(stdscr):
    curses.curs_set(0)
    stdscr.clear()
    stdscr.addstr(0, 0, "Select the model to use:")
    options = ["OpenAI", "Local VLLM"]
    current_option = 0

    while True:
        for idx, option in enumerate(options):
            if idx == current_option:
                stdscr.addstr(idx + 1, 0, f"> {option}", curses.A_REVERSE)
            else:
                stdscr.addstr(idx + 1, 0, f"  {option}")

        key = stdscr.getch()

        if key == curses.KEY_UP and current_option > 0:
            current_option -= 1
        elif key == curses.KEY_DOWN and current_option < len(options) - 1:
            current_option += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            return current_option

def main():
    cap = cv2.VideoCapture(0)  # 0 is typically the default camera
    caption_text = []

    # Initialize curses and get user selection
    selected_option = curses.wrapper(select_model)
    use_openai = (selected_option == 0)

    # Start the periodic capture in a separate thread
    thread = threading.Thread(target=periodic_image_capture, args=(cap, caption_text, use_openai))
    thread.start()

    # Open a window to display the camera preview
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a black background for the caption text
        caption_background = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)

        # Display the caption text on the black background
        if caption_text:
            caption = caption_text[-1]
            wrapped_caption = textwrap.wrap(caption, width=300)
            for i, line in enumerate(wrapped_caption):
                cv2.putText(caption_background, line, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Combine the camera preview and caption background vertically
        combined_frame = np.vstack((frame, caption_background))

        cv2.imshow('Webcam Preview', combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
