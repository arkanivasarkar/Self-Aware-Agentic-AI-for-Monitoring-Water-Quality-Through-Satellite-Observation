import os
import re
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk, ImageSequence, ImageDraw
import pyttsx3
import pygame
from agno.agent import RunResponse
from evaluate_current_satellite_data import monitor_water_quality
from speech_interactions_openai import text_to_speech, speech_to_text 
from ai_agents import multi_ai_agent


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
PROFILE_PICTURE_PATH = os.path.join(BASE_DIR, 'assets', 'NEMORRA.png')
CALLING_GIF_PATH = os.path.join(BASE_DIR, 'assets', 'calling.gif')
SPEECH_GIF_PATH = os.path.join(BASE_DIR, 'assets', 'speech.gif')
RINGTONE_PATH = os.path.join(BASE_DIR, 'assets', 'ringtone_for_nemorra.mp3')


LOCATION = 'location1'  # Location of a fish farm


class NEMORRA:
    def __init__(self, root):
        self.root = root
        self.root.title("NEMORRA")
        self.root.geometry("360x640")  # Set window size to match Android phone aspect ratio
        self.root.configure(bg="black")  # Set background to black
        self.root.iconphoto(True, tk.PhotoImage(file=PROFILE_PICTURE_PATH))

        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Adjust speech speed

        # Add Agent's Profile Picture
        self.circular_image = self.make_profile_picture_circular(self.load_image(PROFILE_PICTURE_PATH, (250, 250)))
        self.static_image = ImageTk.PhotoImage(self.circular_image)
        self.image_label = Label(root, image=self.static_image, bg="black")
        self.image_label.pack(pady=10)

        # Initialize calling GIF and ringtone
        self.calling_gif_frames = self.load_gif(CALLING_GIF_PATH)
        self.calling_ringtone = RINGTONE_PATH
        self.playRingtone = True
        self.calling_gif_uiLabel = Label(root, image=self.calling_gif_frames[0], bg="black", fg='red')
        self.calling_gif_uiLabel.pack(padx=0, pady=100, side=tk.BOTTOM)
        self.calling_gif_uiLabel.bind("<Button-1>", lambda e: self.calling_gif_onpress_event())

        self.calling_gif_current_frame = 0
        self.animate_calling_gif()  # Animate the gif

        # Initialize speech GIF
        self.speech_frames = self.load_gif(SPEECH_GIF_PATH)
        self.is_playing_speech_gif = False
        self.start_converse = True

    def load_image(self, path, size):
        img = Image.open(path)
        return img.resize(size, Image.Resampling.LANCZOS)

    def load_gif(self, path):
        gif = Image.open(path)
        return [ImageTk.PhotoImage(frame.convert("RGBA").copy()) for frame in ImageSequence.Iterator(gif)]

    def make_profile_picture_circular(self, img):
        """Crop an image into a circular shape."""
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, img.size[0], img.size[1]), fill=255)
        circular = Image.new("RGBA", img.size)
        circular.paste(img, (0, 0), mask)
        return circular

    # GIF Handling
    def animate_calling_gif(self):
        """Animate the calling GIF."""
        if not pygame.mixer.music.get_busy() and self.playRingtone:
            pygame.mixer.music.play(-1)  # Play ringtone if not already playing
        self.calling_gif_uiLabel.configure(image=self.calling_gif_frames[self.calling_gif_current_frame])
        self.calling_gif_current_frame = (self.calling_gif_current_frame + 1) % len(self.calling_gif_frames)
        self.root.after(100, self.animate_calling_gif)  # Adjust delay as needed

    def calling_gif_onpress_event(self):
        """Stop calling GIF and ringtone, then start speech GIF."""
        pygame.mixer.music.stop()
        self.playRingtone = False
        self.calling_gif_uiLabel.pack_forget()
        self.calling_gif_uiLabel.unbind("<Button-1>")

        self.speech_gif_uiLabel = Label(self.root, image=self.speech_frames[0], bg="black")
        self.speech_gif_uiLabel.pack(padx=60, pady=30, side=tk.TOP)
        self.speech_gif_current_frame = 0
        self.start_conversation()

    def animate_speech_gif(self):
        """Animate the speech GIF."""
        if self.is_playing_speech_gif:
            self.speech_gif_uiLabel.configure(image=self.speech_frames[self.speech_gif_current_frame])
            self.speech_gif_current_frame = (self.speech_gif_current_frame + 1) % len(self.speech_frames)
            self.root.after(100, self.animate_speech_gif)  # Adjust delay as needed

    def stop_speech_gif(self):
        """Stop speech GIF animation."""
        self.is_playing_speech_gif = False
        self.speech_gif_uiLabel.configure(image=self.speech_frames[0])  # Reset to first frame

    # Speech Methods
    def speak_text(self, text):
        """Speak the text aloud."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()  # Block execution until speaking is done


    # agentic AI output generation
    def process_agentic_ai_output(self, prompt):
        """Process agentic AI output."""
        response: RunResponse = multi_ai_agent.run(prompt)
        return re.sub(r'[#<>/]', '', response.content)  # Clean response


    def start_conversation(self):
        """Start the conversation with NEMORRA."""
        self.is_playing_speech_gif = True
        self.animate_speech_gif()

        # Greet user
        text_to_speech("HI, I am Nemora. I am your Agentic AI friend helping in sustainability.")

        # Monitor water quality
        output = monitor_water_quality(LOCATION)
        text_to_speech(output)

        while True:
            speech_output = speech_to_text()

            # Run model command
            if 'run model' in speech_output.lower():
                output = monitor_water_quality(LOCATION)
                text_to_speech(output)
                continue

            # Break loop
            if 'stop' in speech_output.lower() or 'all' in speech_output.lower():
                break

            # Process AI response
            response = self.process_agentic_ai_output(speech_output)
            text_to_speech(response)

        # End conversation
        text_to_speech("It was nice to talk to you.")
        self.root.after(2000, self.root.destroy)  # Close after 2 seconds


if __name__ == "__main__":
    root = tk.Tk()
    app = NEMORRA(root)
    root.mainloop()
