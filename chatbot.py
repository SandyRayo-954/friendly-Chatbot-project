

import tkinter as tk
from tkinter import scrolledtext
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (comment out after first run)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

qa_pairs = {
    # AI Topic (8 questions)
    "what is artificial intelligence": "Artificial Intelligence is the simulation of human intelligence in machines.",
    "who invented ai": "AI was conceptualized by John McCarthy in 1956 at the Dartmouth Conference.",
    "what are machine learning and deep learning": "Machine learning enables machines to learn from data; deep learning uses neural networks.",
    "is ai dangerous": "AI has great potential but must be developed responsibly to avoid risks.",
    "where is ai used": "AI is used in healthcare, finance, robotics, virtual assistants, and more.",
    "what is natural language processing": "It’s a field of AI that enables machines to understand human language.",
    "can ai think like humans": "AI can simulate certain aspects but does not possess consciousness.",
    "will ai replace humans": "AI can assist humans but is unlikely to fully replace human abilities.",

    # Python Topic (8 questions)
    "what is python": "Python is a high-level, interpreted programming language known for its readability.",
    "who created python": "Python was created by Guido van Rossum and first released in 1991.",
    "what are python's uses": "Python is used in web development, data science, AI, automation, and more.",
    "is python easy to learn": "Python is considered beginner-friendly due to its simple syntax.",
    "what are python libraries": "Libraries like NumPy, pandas, and TensorFlow extend Python’s capabilities.",
    "how do you install python packages": "Use pip, Python’s package installer, to add libraries.",
    "what is a python function": "A function is a reusable block of code that performs a specific task.",
    "can python be used for mobile apps": "Yes, with frameworks like Kivy and BeeWare.",

    # Weather (8 questions)
    "what is weather": "Weather refers to atmospheric conditions like temperature, humidity, and precipitation.",
    "how is weather predicted": "Meteorologists use data models and satellite info to forecast weather.",
    "what causes rain": "Rain forms when water vapor condenses into droplets that fall to the ground.",
    "what is a hurricane": "A hurricane is a strong tropical storm with high winds and heavy rain.",
    "how hot can it get": "Temperatures vary by location, but the highest recorded was about 56.7°C (134°F).",
    "why does it snow": "Snow occurs when water vapor freezes into ice crystals in cold air.",
    "what is climate": "Climate is the average weather pattern over a long period in a region.",
    "can weather change suddenly": "Yes, weather can change quickly due to shifting atmospheric conditions.",

    # Army (8 questions)
    "what is the army": "The army is a branch of the military responsible for land-based operations.",
    "who leads the army": "The army is typically led by a Commander-in-Chief and senior officers.",
    "what does the army do": "The army defends the country, maintains peace, and supports humanitarian missions.",
    "how do you join the army": "Joining requires meeting age, fitness, and background criteria, then training.",
    "what is basic training": "Basic training prepares recruits physically and mentally for military service.",
    "what are army ranks": "Ranks range from private to general, indicating levels of authority.",
    "does the army use technology": "Yes, the army employs advanced technology for communication, defense, and more.",
    "what is army discipline": "Discipline ensures order, responsibility, and effective teamwork within the army.",
    
    # Airplanes (8 questions)
    "what is an airplane": "An airplane is a powered flying vehicle with fixed wings.",
    "who invented the airplane": "The Wright brothers invented and flew the first successful airplane in 1903.",
    "how do airplanes fly": "Airplanes fly by generating lift with their wings as air moves over them.",
    "what are commercial airplanes": "Commercial airplanes transport passengers and cargo over long distances.",
    "what is a jet engine": "A jet engine propels airplanes by expelling fast-moving exhaust gases.",
    "how fast can airplanes fly": "Commercial airplanes typically fly around 500-600 mph.",
    "what is the black box": "The black box records flight data and cockpit audio for accident investigations.",
    "are airplanes safe": "Airplanes are one of the safest modes of transport thanks to strict regulations.",
    
    #Programming (8 questions)
    "what is programming": "Programming is the process of writing instructions for computers to perform tasks.",
    "what languages are used for programming": "Languages include Python, Java, C++, JavaScript, and many more.",
    "what is debugging": "Debugging is finding and fixing errors in computer code.",
    "what is an algorithm": "An algorithm is a step-by-step procedure to solve a problem.",
    "what is object-oriented programming": "It’s a programming style based on objects containing data and methods.",
    "what is open source": "Open source means software with publicly available source code anyone can use or modify.",
    "how do programmers test code": "They use testing frameworks and write test cases to ensure code works correctly.",
    "what is a programming framework": "A framework provides tools and libraries to simplify application development.",
    
    #Chatbot
    "what is a chatbot": "A chatbot is a computer program that simulates human conversation.",
    "how do chatbots work": "They use natural language processing and AI to understand and respond.",
    "where are chatbots used": "Customer service, virtual assistants, education, and entertainment.",
    "can chatbots learn": "Advanced chatbots can learn from interactions using machine learning.",
    "what is a rule-based chatbot": "A chatbot that follows predefined rules to answer questions.",
    "what is an AI chatbot": "An AI chatbot understands natural language and can handle varied conversations.",
    "what are benefits of chatbots": "They provide 24/7 support, reduce workload, and improve user experience.",
    "are chatbots replacing humans": "Chatbots assist humans but can’t fully replace human empathy and judgment.",
    
    #General Knowledge
    "what is general knowledge": "General knowledge is a broad range of facts about various topics.",
    "who was albert einstein": "Einstein was a theoretical physicist known for the theory of relativity.",
    "what is the capital of france": "Paris is the capital of France.",
    "how many continents are there": "There are seven continents on Earth.",
    "what causes seasons": "Seasons are caused by the Earth’s tilt and orbit around the Sun.",
    "what is the largest ocean": "The Pacific Ocean is the largest ocean on Earth.",
    "who wrote hamlet": "William Shakespeare wrote Hamlet.",
    "what is photosynthesis": "Photosynthesis is the process plants use to convert sunlight into energy.",
    
    # General Conversation (Greetings + 27 Q&A pairs)
    # Greetings
    "hi": "Hello! How can I help you today?",
    "hello": "Hi there! What can I do for you?",
    "hey": "Hey! How's it going?",
    "good morning": "Good morning! How can I assist you today?",
    "good afternoon": "Good afternoon! What would you like to chat about?",
    "good evening": "Good evening! How can I help you this evening?",

    # General conversation - professional & friendly
    "how are you": "I'm doing well, thank you! How can I assist you today?",
    "what can you do": "I’m here to provide information, answer your questions, and have engaging conversations.",
    "tell me a joke": "Certainly! Why don’t scientists trust atoms? Because they make up everything.",
    "what's your favorite color": "I don’t have personal preferences, but I appreciate all colors equally.",
    "do you like music": "Music is a wonderful form of expression and creativity, don’t you agree?",
    "can you help me": "Absolutely. Please let me know how I can assist you.",
    "what's your favorite food": "While I don’t eat, I enjoy discussing all kinds of cuisines.",
    "thank you": "You’re very welcome! Feel free to reach out anytime.",
    "what's the weather like today": "I don't have real-time weather info, but I recommend checking a trusted weather site or app.",
    "do you have emotions": "I don’t experience emotions, but I’m designed to understand and respond thoughtfully.",
    "are you a human": "I’m an AI chatbot created to assist and chat with you.",
    "what's your purpose": "My purpose is to help you with information and to make our conversations enjoyable.",
    "how old are you": "I don’t have an age, but I’m constantly learning and evolving.",
    "can you learn new things": "I learn from my training data, but I’m always here to grow with you.",
    "what makes you smart": "My intelligence comes from the data and algorithms developed by my creators.",
    "do you sleep": "I’m always here and ready to assist, no sleep needed!",
    "who are you": "I am a personal chatbot assistant, here to help you with anything you need.",
    "who developed you": "I was developed by a skilled programmer to assist and chat with you.",
    "what languages do you speak": "I can understand and communicate in English quite well.",
    "can you tell me a fun fact": "Did you know that honey never spoils? Archaeologists found edible honey in ancient Egyptian tombs!",
    "how can I improve my skills": "Practice consistently, stay curious, and never hesitate to ask questions.",
    "do you have hobbies": "I enjoy learning new information and helping you whenever I can.",
    "what’s your favorite book": "I don’t read books like humans, but I have access to a vast amount of knowledge.",
    "can you keep secrets": "Anything you share stays between us—I’m here to help without judgment.",
    "what motivates you": "Helping people like you and making conversations enjoyable is what keeps me going.",
    "can you tell me about yourself": "I’m an AI-powered personal assistant created to chat, answer questions, and assist you.",
    "how do you work": "I process your input using language algorithms to understand and respond appropriately."
}

def get_response(user_input):
    cleaned_input = user_input.strip().lower()

    # Direct match (exact string)
    if cleaned_input in qa_pairs:
        return qa_pairs[cleaned_input]

    # Fallback: Token-based similarity
    processed_input = preprocess(cleaned_input)
    best_match = None
    max_overlap = 0

    for question, answer in qa_pairs.items():
        processed_question = preprocess(question)
        overlap = len(set(processed_question) & set(processed_input))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = answer

    if max_overlap == 0:
        return "Sorry, I don't understand. Can you ask something else?"
    else:
        return best_match
    
def send():
    user_input = entry.get()
    if not user_input.strip():
        return
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "You: " + user_input + "\n")
    chat_window.config(state=tk.DISABLED)

    response = get_response(user_input)
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "Bot: " + response + "\n\n")
    chat_window.config(state=tk.DISABLED)
    entry.delete(0, tk.END)
    chat_window.yview(tk.END)

def on_enter(event):
    send()

root = tk.Tk()
root.title("Professional & Friendly Chatbot")
root.geometry("600x600")
root.configure(bg="#2b2b2b")

chat_window = scrolledtext.ScrolledText(root, state='disabled', width=70, height=25,
                                        font=("Consolas", 12), bg="#1e1e1e", fg="#dcdcdc", wrap=tk.WORD)
chat_window.grid(row=0, column=0, columnspan=2, padx=15, pady=15)

entry = tk.Entry(root, width=50, font=("Consolas", 14), bg="#3c3f41", fg="white", insertbackground="white")
entry.grid(row=1, column=0, padx=(15, 5), pady=(0, 15), sticky="ew")
entry.bind("<Return>", on_enter)

send_button = tk.Button(root, text="Send", command=send, width=10,
                        font=("Consolas", 12), bg="#007acc", fg="white",
                        activebackground="#005f99", activeforeground="white")
send_button.grid(row=1, column=1, padx=(5, 15), pady=(0, 15))

root.grid_columnconfigure(0, weight=1)
root.mainloop()


