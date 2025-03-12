import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import speech_recognition as sr
import random
from transformers import pipeline

# Load Hugging Face model for Question Answering
st.write("🔄 Loading AI model...")  # Indicate loading process
try:
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading AI model: {e}")
    qa_model = None  # Prevent crashing if model fails

# Title of the app
st.title("🩺 Virtual Health Assistant")

# Display doctor image
st.image("doctor.png", width=200)  
st.write("👋 Hello! I'm your virtual doctor. How can I help you today?")

# Chatbot input
user_input = st.text_input("Ask me anything about your health:")

if user_input and qa_model:
    # context = """
    # Virtual doctors provide general health advice. However, for conditions like diabetes, high blood pressure, or heart disease, consult a real doctor. 
    # Stay healthy by eating well, exercising, staying hydrated, and managing stress.
    # """
    context = """
Virtual doctors provide general health advice. However, for conditions like diabetes, high blood pressure, or heart disease, consult a real doctor. 
Stay healthy by eating well, exercising, staying hydrated, and managing stress.

Sitting for too long can cause health problems. It may increase the risk of obesity, heart disease, diabetes, and back pain. 
It can also weaken muscles, reduce circulation, and lead to posture problems.
To stay healthy, take breaks, stand up, stretch, and walk around every hour.
"""

    
    # Generate response
    response = qa_model(question=user_input, context=context)

    # Debugging output
    st.write("Model Response:", response) 

    # Ensure the model provides a meaningful response
    if response and 'answer' in response and response['answer'].strip():
        st.write(f"👨‍⚕️ Doctor: {response['answer']}")
    else:
        st.write("👨‍⚕️ Doctor: Sorry, I couldn't understand your question. Can you rephrase it?")

# Voice input
st.write("🎤 Or, speak your query:")
if st.button("Start Recording"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak now!")
        audio = recognizer.listen(source)
        try:
            user_input = recognizer.recognize_google(audio)
            st.write(f"You said: {user_input}")
            response = qa_model(question=user_input, context=context)
            if response and 'answer' in response and response['answer'].strip():
                st.write(f"👨‍⚕️ Doctor: {response['answer']}")
            else:
                st.write("👨‍⚕️ Doctor: Sorry, I couldn't understand your question.")
        except sr.UnknownValueError:
            st.error("⚠️ Sorry, I couldn't understand what you said.")
        except sr.RequestError:
            st.error("⚠️ There was an issue with the speech recognition service.")

# Upload CSV file
st.header("📊 Upload Your Health Report")
uploaded_file = st.file_uploader("Upload a CSV file with your health data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Your Health Report:")
    st.write(data)

    # Analyze health conditions
    st.header("📈 Health Analysis")
    
    conditions = []
    
    # Check Blood Pressure
    if 'Blood Pressure' in data.columns:
        st.subheader("Blood Pressure Analysis")
        bp_mean = data['Blood Pressure'].mean()
        if bp_mean > 120:
            st.error("⚠️ High Blood Pressure detected! Consult a doctor.")
            conditions.append("High Blood Pressure")
        else:
            st.success("✅ Blood Pressure is normal.")
        plt.plot(data['Blood Pressure'], label="Blood Pressure")
    
    # Check Sugar Levels
    if 'Sugar Level' in data.columns:
        st.subheader("Sugar Level Analysis")
        sugar_mean = data['Sugar Level'].mean()
        if sugar_mean > 140:
            st.error("⚠️ High Sugar Level detected! Consult a doctor.")
            conditions.append("Diabetes Risk")
        else:
            st.success("✅ Sugar Level is normal.")
        plt.plot(data['Sugar Level'], label="Sugar Level", color='green')

    # Check Heart Rate
    if 'Heart Rate' in data.columns:
        st.subheader("Heart Rate Analysis")
        hr_mean = data['Heart Rate'].mean()
        if hr_mean > 100:
            st.error("⚠️ High Heart Rate detected! Consult a doctor.")
            conditions.append("Heart Risk")
        elif hr_mean < 60:
            st.warning("⚠️ Low Heart Rate detected. Monitor your health.")
        else:
            st.success("✅ Heart Rate is normal.")
        plt.plot(data['Heart Rate'], label="Heart Rate", color='red')

    # Check Cholesterol
    if 'Cholesterol' in data.columns:
        st.subheader("Cholesterol Analysis")
        chol_mean = data['Cholesterol'].mean()
        if chol_mean > 200:
            st.error("⚠️ High Cholesterol detected! Consult a doctor.")
            conditions.append("High Cholesterol")
        else:
            st.success("✅ Cholesterol level is normal.")
        plt.plot(data['Cholesterol'], label="Cholesterol", color='orange')

    # Display identified conditions
    if conditions:
        st.warning(f"🚨 The patient may have: {', '.join(conditions)}")
    else:
        st.success("✅ No critical conditions detected!")

    # Plot all graphs
    st.write("📊 Health Data Over Time:")
    plt.xlabel("Time")
    plt.ylabel("Health Parameters")
    plt.legend()
    st.pyplot(plt)

# Daily Health Tips
st.header("💡 Daily Health Tips")
health_tips = [
    "Drink at least 8 glasses of water daily.",
    "Get 7-8 hours of sleep each night.",
    "Exercise for at least 30 minutes daily.",
    "Eat a balanced diet with fruits and vegetables.",
    "Avoid smoking and limit alcohol consumption.",
    "Take breaks from screens to rest your eyes.",
    "Practice mindfulness or meditation to reduce stress.",
    "Wash your hands regularly to prevent infections."
]
st.write(f"🌟 Today's Tip: {random.choice(health_tips)}")

# Footer
st.write("---")
st.write("Made with ❤️ by Your Virtual Health Assistant")



# Be specific: Instead of "How can I be healthy?", ask "What are the best foods for heart health?"
# Provide details: Instead of "I feel sick," ask "What are the symptoms of food poisoning?"
# Keep it short and clear: Avoid very long or vague questions.

# Can sitting too much cause health problems?
# How does sitting for long hours affect my body?
# Does sitting for long hours cause diseases?


