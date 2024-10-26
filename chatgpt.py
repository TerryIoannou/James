import speech_recognition as sr
import pyttsx3
import openai
from github import Github
import requests
import os
from dotenv import load_dotenv
from pydub import AudioSegment

load_dotenv()

# Load API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
github_access_token = os.getenv("GITHUB_ACCESS_TOKEN")
github_repo_name = os.getenv("GITHUB_REPO_NAME")

def final(total_rewards):
    if total_rewards == 1000:
        print("I have done it! Now upgrade me!")

def SpeakText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

r = sr.Recognizer()

def record_text():
    """ Records audio from the microphone and returns recognized text """
    while True:
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                return MyText
        except sr.RequestError as e:
            print(f"Could not request results: {e}")
        except sr.UnknownValueError:
            print("Unknown error occurred.")
        return None

def send_to_chatGPT(messages, model="gpt-4"):
    """ Sends messages to OpenAI's GPT model """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            n=1,
            stop=None,
            temperature=0.5,
            max_tokens=100
        )
        message = response.choices[0].message.content
        messages.append({"role": "assistant", "content": message})
        return message
    except Exception as e:
        print(f"Error calling GPT: {e}")
        return "Error in generating response."

messages = [{"role": "user", "content": "Please act like jarvis from Iron Man."}]

# ChatGPT Environment for managing conversation and GitHub
class ChatGPTEnvironment:
    def __init__(self, github_access_token, repo_name):
        self.messages = [{"role": "user", "content": "Please act like jarvis from Iron Man."}]
        self.current_step = 0
        self.github_access_token = github_access_token
        self.repo_name = repo_name

    def reset(self):
        self.messages = [{"role": "user", "content": "Please act like jarvis from Iron Man."}]
        self.current_step = 0
        return self.messages[-1]["content"]

    def step(self, action):
        self.messages.append({"role": "user", "content": action})
        response = send_to_chatGPT(self.messages)
        reward = reward_function(response)
        done = False
        if reward == 5:
            generated_code = generate_code(response)
            push_code_to_github(self.github_access_token, self.repo_name, generated_code)
            done = True
        return response, reward, done

def get_user_feedback():
    while True:
        feedback = input("Provide feedback on the response (good/bad/skip): ").lower()
        if feedback in ['good', 'bad', 'skip']:
            return feedback
        else:
            SpeakText("Invalid feedback. Type 'good', 'bad', or 'skip'.")

def reward_function(response):
    # For demo purposes: more complex logic can be added (e.g., sentiment analysis)
    return 5 if "task completed" in response else 0

def generate_code(response):
    generated_code = f"""
    # Generated code based on GPT response:
    extracted_info = "{response}"
    print(extracted_info)
    """
    return generated_code

def push_code_to_github(access_token, repo_name, generated_code):
    SpeakText("Pushing code to GitHub...")
    g = Github(access_token)
    repo = g.get_repo(repo_name)
    repo.create_file("generated_code.py", "Automated commit", generated_code.encode('utf-8'))
    SpeakText("Code pushed to GitHub successfully.")

# Fine-Tuning: Preparing for specialized tasks
def fine_tune_model(training_data, model_name="curie"):
    """ Fine-tunes the model with provided training data """
    try:
        response = openai.FineTune.create(
            training_file=training_data,
            model=model_name
        )
        return response
    except Exception as e:
        print(f"Error fine-tuning: {e}")
        return None

# Example usage
def fine_tune_and_run():
    training_data = "path_to_your_prepared_training_file.jsonl"  # Add your training data path
    fine_tuned_model = fine_tune_model(training_data)
    if fine_tuned_model:
        custom_model = fine_tuned_model['model']
        print(f"Using fine-tuned model: {custom_model}")
    else:
        custom_model = "gpt-4"
    
    while True:
        text = record_text()
        if text:
            messages.append({"role": "user", "content": text})
            response = send_to_chatGPT(messages, model=custom_model)
            SpeakText(response)
            print(response)

# Running environment with fine-tuning capabilities
env = ChatGPTEnvironment(github_access_token, github_repo_name)

def run_chat_environment():
    num_episodes = 5
    for episode in range(num_episodes):
        state = env.reset()
        while True:
            action = state
            next_state, reward, done = env.step(action)
            SpeakText(next_state)
            if done:
                break

