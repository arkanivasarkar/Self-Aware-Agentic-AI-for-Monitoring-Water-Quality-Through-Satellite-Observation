import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk, ImageSequence, ImageDraw
import pyttsx3
import pygame
from agno.agent import RunResponse
import re
import requests
# Get the parent directory to import all children
import sys
import os
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AIAgents.speech_interactions import text_to_speech, speech_to_text  # You can remove this if not used
from AIAgents.ai_agent import web_search_agent, rewriter_agent



# openai.api_key=os.getenv("OPENAI_API_KEY")
os.environ['API_KEY'] = 'ag-58BhPxCx8IoiNEX0Yy-hWtKgaYyoh66MszhvhZRM9_g'
os.environ['GROQ_API_KEY'] = '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'


class AgenticAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NEMORRA")
        self.root.geometry("360x640")  # Set window size to match an Android phone aspect ratio
        self.root.configure(bg="black")  # Set background to black
        root.iconphoto(True, tk.PhotoImage(file="C:\\Users\\arkaniva\\Downloads\\NEMORRA.png"))


        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Adjust speech speed


        # Add Agent's Profile Picture
        img = Image.open("C:\\Users\\arkaniva\\Downloads\\NEMORRA.png")  # Replace with your image path
        img = img.resize((250, 250), Image.Resampling.LANCZOS)  # Resize image to fit aspect ratio
        self.circular_image = self.make_profile_picture_circular(img)
        self.static_image = ImageTk.PhotoImage(self.circular_image)
        self.image_label = Label(root, image=self.static_image, bg="black")
        self.image_label.pack(pady=10)

        # Add Calling Icon Gif
        gif_path = "C:\\Users\\arkaniva\\Downloads\\firstrequestrecords-first-request.gif"  # Replace with your GIF path
        self.calling_gif = Image.open(gif_path)
        self.calling_gif_frames = [ImageTk.PhotoImage(frame.convert("RGBA").copy()) for frame in ImageSequence.Iterator(self.calling_gif)]


        # Add ringtone and animate calling gif
        pygame.mixer.init()
        self.calling_ringtone = "C:\\Users\\arkaniva\\Downloads\\7120-download-iphone-6-original-ringtone-42676.mp3"  # Replace with your MP3 file
        pygame.mixer.music.load(self.calling_ringtone)
        self.playRingtone = True
        self.calling_gif_uiLabel = Label(root, image=self.calling_gif_frames[0], bg="black", fg='red')  # Static first frame
        self.calling_gif_uiLabel.pack(padx=0, pady=100, side=tk.BOTTOM)
        self.calling_gif_uiLabel.bind("<Button-1>", lambda e: self.calling_gif_onpress_event())

        self.calling_gif_current_frame = 0
        self.animate_calling_gif()  # Animate the gif

        # Add speech GIF for when AI is speaking
        gif_path = "C:\\Users\\arkaniva\\Downloads\\657857332480a11898e8759599f591bf.gif"  # Replace with your GIF path
        self.speech_gif = Image.open(gif_path)
        self.speech_frames = [ImageTk.PhotoImage(frame.copy()) for frame in ImageSequence.Iterator(self.speech_gif)]
        self.is_playing_speech_gif = False

        self.start_converse = True


    def animate_calling_gif(self):
        """Animate the GIF if playing."""
        if not pygame.mixer.music.get_busy() and self.playRingtone:  # Play only if not already playing
            pygame.mixer.music.play(-1)
        self.calling_gif_uiLabel.configure(image=self.calling_gif_frames[self.calling_gif_current_frame])
        self.calling_gif_current_frame = (self.calling_gif_current_frame + 1) % len(self.calling_gif_frames)
        self.root.after(100, self.animate_calling_gif)  # Adjust delay as needed


    def calling_gif_onpress_event(self):
        pygame.mixer.music.stop()
        self.playRingtone = False
        self.calling_gif_uiLabel.pack_forget()
        self.calling_gif_uiLabel.unbind("<Button-1>") 

        self.speech_gif_uiLabel = Label(self.root, image=self.speech_frames[0], bg="black")  # Static first frame
        self.speech_gif_uiLabel.pack(padx=60, pady=30, side=tk.TOP)
        self.speech_gif_current_frame = 0

        self.calling_gif_uiLabel.update_idletasks()
        self.speech_gif_uiLabel.update_idletasks()
        self.start_conversation()


    def make_profile_picture_circular(self, img):
        """Crop an image into a circular shape."""
        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, img.size[0], img.size[1]), fill=255)
        circular = Image.new("RGBA", img.size)
        circular.paste(img, (0, 0), mask)
        return circular

    

    def start_gif(self):
        """Start GIF animation."""
        # if not self.is_playing_speech_gif:
        self.is_playing_speech_gif = True
        self.animate_speech_gif()


    def animate_speech_gif(self):
        """Animate the GIF if playing."""
        if self.is_playing_speech_gif:
            self.speech_gif_uiLabel.configure(image=self.speech_frames[self.speech_gif_current_frame])
            self.speech_gif_current_frame = (self.speech_gif_current_frame + 1) % len(self.speech_frames)
            self.root.after(100, self.animate_speech_gif)  # Adjust delay as needed


    def speak_text(self, text):
        """Speak the text and stop the GIF after speaking."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()  # This blocks execution until speaking is done


    def stop_gif(self):
        """Stop GIF animation."""
        self.is_playing_speech_gif = False
        self.speech_gif_uiLabel.configure(image=self.speech_frames[self.speech_gif_current_frame])  # Reset to first frame


    def process_agentic_ai_output(self, agentType, promt):
        response: RunResponse = agentType.run(promt)
        response2: RunResponse = rewriter_agent.run(response.content)
        response2 = re.sub(r'[#<>/]', '', response2.content)
        return response2
    

    def start_conversation(self):
        self.start_gif()
        text_to_speech('HI I am Nemora. I am your Agentic AI friend helping in sustainability.')
        text_to_speech('I was going through my daily satellite data survey, and I noticed that you have degrading water quality in your fish farm.')
      
        output = self.process_agentic_ai_output(web_search_agent, "My satellite image analysis from SENTINEL dataset tell me that the water qaulity in my fish farm is going bad.Identify the reasons. Answer in a paragraph in brief.")
        
        #print(output)
        root.after(20, text_to_speech(output))
        while True:
            speech_output = speech_to_text()#
            print(f'You Spkoe: {speech_output}')
            if 'stop' in speech_output.lower() or 'all' in speech_output.lower():
                break

            if 'show' in speech_output.lower():
                    text_to_speech('Sure, I am fetching and compiling the information.')
                    url = 'https://behappyfishstorage.blob.core.windows.net/behappyfish/NEMORRA_FarmerX_Report.pdf'
                    
                    file_name = url.split("/")[-1]
                    file_path = os.path.join("C:\\Users\\arkaniva\\Downloads", file_name)

                    # Download the file and save it to the Downloads folder
                    response = requests.get(url)

                    # Check if the request was successful
                    if response.status_code == 200:
                        with open(file_path, 'wb') as file:
                            file.write(response.content)
                        print(f"File downloaded successfully to {file_path}")
                    else:
                        print(f"Failed to download file. Status code: {response.status_code}")

                    continue

            agentic_ai_output = self.process_agentic_ai_output(web_search_agent, speech_output)
            root.after(20, text_to_speech(agentic_ai_output))

        text_to_speech(f'It was nice to talk to you.')
        root.after(2000, root.destroy)
        


# Main code to run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = AgenticAIApp(root)
    root.mainloop()
