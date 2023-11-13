import speech_recognition as sr
import pyttsx3
import openai
import requests
from pydub import AudioSegment



def final(total_rewards):

    if total_rewards == 100:
     return print("I have done it now upgrade me!")
    





openai.api_key = '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'

def SpeakText(command):
# start the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

r = sr.Recognizer()




def record_text():
       while(1):
        try:

                with sr.Microphone() as source2:
                     #use the microphone as source of input
                     r.adjust_for_ambient_noise(source2, duration=0.2)
                    #listen to the user or me
                     audio2 = r.listen(source2)
                    
                    #using google to recognize audio
                     MyText = r.recognize_google(audio2)

                     return MyText

#

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("unknown error occured")
        return



def send_to_chatGPT(messages, model="babbage-002"):
    response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    n=1,
    stop=None,
    temperature=0.5,
    max_tokens=100 # Limit the length of the generated output
)
    message = response.choices[0].message.content
    messages.append(response.choices[0].message)
    return message
messages = [{"role": "user", "content": "Please act like jarvis from Iron Man."}]
while(1):
     text = record_text()
     messages.append({"role": "user", "content": text})
     response = send_to_chatGPT(messages)
     SpeakText(response)

     print(response)


class ChatGPTEnvironment:
    def __init__(self):
        # Initialize any necessary components
        self.messages = [{"role": "user", "content": "Please act like jarvis from Iron Man."}]
        self.current_step = 0

    def reset(self):
        # Reset the environment and return an initial state (prompt)
        self.messages = [{"role": "user", "content": "Please act like jarvis from Iron Man."}]
        self.current_step = 0
        initial_state = self.messages[-1]["content"]
        return initial_state

    def step(self, action):
        self.messages.append({"role": "user", "content": action})
        response = send_to_chatGPT(self.messages)
        state = response
        reward = reward_function(response)
        done = False  # Set to True if the conversation is finished
        return state, reward, done
    

    
def analyze_tone_and_context(audio, messages):
    # Placeholder: Analyze both tone of voice and context of conversation
    tone_score = analyze_tone_of_voice(audio)
    context_score = analyze_context_of_conversation(messages)
    return tone_score, context_score

def analyze_tone_of_voice(audio):
        if audio.channels == 2:
            audio = audio.set_channels(1)
    
    # Analyze tone of voice and return a score (e.g., between 0 and 1)
        loudness = audio.dBFS
        tone_score = (loudness + 40) / 40  # Normalize to a range of 0 to 1
        return tone_score   
#
def analyze_context_of_conversation(messages):
    # Placeholder: Analyze the context of the conversation and return a score (e.g., between 0 and 1)
    # This is a very basic example, you may implement a more sophisticated context analysis based on your needs
    key_terms = ["jarvis", "iron man"]
    user_input = messages[-1]["content"]
    context_score = sum(1 for term in key_terms if term in user_input) / len(key_terms)
    return context_score

def connectivityToGitHub(repo):
    # Set up your API endpoint
    api_url = 'add api end point of my repo'

# Set up the request headers with your access token
    headers = {'Authorization': 'token YOUR_ACCESS_TOKEN'}

# Make the API request
    response = requests.get(api_url, headers=headers)

# Check the response
    if response.status_code == 200:
        # Success! Parse and use the response data
        repositories = response.json()
        for repo in repositories:
            print(repo['name'])
    else:
        print(f"Error accessing GitHub API. Status code: {response.status_code}")
def reward_function(response, tone_score, context_score, assigned_task_completed):
    # Implement a function that evaluates the quality of the response
    # Provide a positive reward for desirable responses and a negative reward for less desirable ones
    # You can use various metrics (e.g., sentiment analysis, similarity to target response, etc.) to compute the reward
    
    task_reward = 2 if assigned_task_completed else 0  # Assign a positive reward if the task is completed
    
    if tone_score > 0.7 and context_score > 0.5:
        return 0.5 + task_reward  # Positive reward for good tone and context, plus task completion reward
    elif tone_score < 0.3:
        return -1  # Negative reward for negative tone
    else:
        return 0  # Neutral reward for other cases

# Set up your API endpoint
api_url = 'add api end point of my repo'

# Set up the request headers with your access token
headers = {'Authorization': 'token YOUR_ACCESS_TOKEN'}

# Make the API request
response = requests.get(api_url, headers=headers)

# Check the response
if response.status_code == 200:
    # Success! Parse and use the response data
    repositories = response.json()
    for repo in repositories:
        print(repo['name'])
else:
    print(f"Error accessing GitHub API. Status code: {response.status_code}")


env = ChatGPTEnvironment()


num_episodes = 5  # Set the number of episodes
for episode in range(num_episodes):
    state = env.reset()

    while True:
        # Example action selection (prompt generation)
        action = state

        # Interact with ChatGPT and get response
        next_state, _, _ = env.step(action)

        # Analyze tone of voice and context of conversation
        audio_input = ...
        tone_score = analyze_tone_of_voice(action)
        context_score = analyze_context_of_conversation(env.messages)

        # Calculate reward based on tone and context scores
        reward = reward_function(next_state, tone_score, context_score)

        # Optionally, you can convert response to speech and play it
        SpeakText(next_state)

        # Check if episode is done (e.g., after a certain number of interactions or a terminal condition is met)
        done = False  # Placeholder, you'll need to set the actual done condition

        if done:
            reward+=2
            break


