import speech_recognition as sr

class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)

    def get_voice_input(self):
        try:
            with self.microphone as source:
                print("\nListening... (Speak now)")
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=10)

            print("Processing speech...")
            return self.recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            print("No speech detected within timeout period.")
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        return None

# if __name__ == "__main__":
#     voice = VoiceHandler()
#     print(voice.get_voice_input())