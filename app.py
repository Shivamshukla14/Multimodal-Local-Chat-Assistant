import streamlit as st
import os
from llm_chains import load_normal_chain
from utils import save_chat_history_json, load_chat_history_json, get_timestamp
from audio_handler import transcribe_audio
from images_handler import handle_image
from pdf_chat_handler import pdf_chat_handler
from html_templates import get_bot_template, get_user_template, css
from langchain.memory import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder
import yaml


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

#Functions--------------------------------------------------------------
def load_chain(chat_history):
    return load_normal_chain(chat_history)


def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""


def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "new_session":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.new_session_key)
        else:
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.session_key)


def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key



#Main function---------------------------------------------------------------
def main():
    st.title("Multimodal Local Chat(MLOC) Assistant 🤖")
    st.write(css, unsafe_allow_html = True)
    chat_container = st.container()
    st.sidebar.title("Chat Sessions💬")
    chat_sessions = ["new_session"] + os.listdir(config["chat_history_path"])
    print(chat_sessions)

    
    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.pdf_chat = False
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key = "session_key", index = index, on_change = track_index)

    if st.session_state.session_key != "new_session":
        st.session_state.history = load_chat_history_json(config["chat_history_path"] + st.session_state.session_key)
    else:
        st.session_state.history = []
    

    chat_history = StreamlitChatMessageHistory(key = "history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input("Enter a prompt here:", key = "user_input", on_change = set_send_input)

    send_button_column, voice_recording_column = st.columns([0.11, 1])
    with send_button_column:
        send_button = st.button("Send",  key = "send_button", on_click=clear_input_field)
    with voice_recording_column:
        voice_recording = mic_recorder(start_prompt="🎙️", stop_prompt="🎤", just_once = True)
    

    uploaded_audio = st.sidebar.file_uploader("Upload your audio file🔉:", type = ["wav", "mp3", "ogg"])
    uploaded_image = st.sidebar.file_uploader("Upload your image file📷:", type = ["jpeg", "jpg", "png"])
    uploaded_pdf = st.sidebar.file_uploader("Upload your pdf file📃:", type = ["pdf"], accept_multiple_files = True)

    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        print(transcribed_audio)
        llm_chain.run("Summmarize this text: " + transcribed_audio)

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain.run(transcribed_audio)


    if send_button or st.session_state.send_input:

        if uploaded_image:
            with st.spinner("Processing your image..."):
                user_message = "Describe the image in detail please"
                if st.session_state.user_question != "":
                    user_message = st.session_state.user_question
                    st.session_state.user_question = ""
                llm_response = handle_image(uploaded_image.getvalue(), st.session_state.user_question)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_response)

        if uploaded_pdf:
            with st.spinner("Processing your pdf..."):
                user_message = "Give a short summary on the pdf"
                if st.session_state.user_question != "":
                    user_message = st.session_state.user_question
                    st.session_state.user_question = ""
                llm_response = pdf_chat_handler(uploaded_pdf, user_message)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_response)

        
        if st.session_state.user_question != "":       
            llm_response = llm_chain.run(st.session_state.user_question)
            st.session_state.user_question = ""

    if chat_history.messages != []:
        with chat_container:
            st.write("Chat History: ")
            for message in chat_history.messages:
                if message.type == "human":
                    st.write(get_user_template(message.content), unsafe_allow_html = True)
                else:
                    st.write(get_bot_template(message.content), unsafe_allow_html = True)
                # st.chat_message(message.type).write(message.content)


    save_chat_history()

if __name__ == "__main__":
    main()