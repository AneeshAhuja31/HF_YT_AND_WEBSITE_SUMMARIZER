# import validators
# import streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain.chains.summarize import load_summarize_chain
# from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
# from langchain_community.document_loaders.youtube import TranscriptFormat
# from langchain_huggingface import HuggingFaceEndpoint


# st.title("Summarize Text from YT or Website")
# st.subheader("Summarize URL")


# with st.sidebar:
#     hf_token = st.text_input("Enter HF API Token",value="",type="password")

# url = st.text_input("URL",label_visibility="collapsed")

# repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
# llm = HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temperature=0.7,token=hf_token)

# prompt_template = """
# Provide summary of the following content in 300 words:
# Content:{text}
# """

# prompt = PromptTemplate(template=prompt_template,input_variables=['text'])

# if st.button("Summarize the content from YT or Website"):
#     if not hf_token.strip() or not url.strip():
#         st.error("Please provide the information")
#     elif not validators.url(url):
#         st.error("Please enter a valid a valid url.")
#     else:
#         try:
#             with st.spinner("Waiting..."):
#                 if "youtube.com" in url:
#                     loader = YoutubeLoader.from_youtube_url(url,add_video_info=True)
#                 else:
#                     loader = UnstructuredURLLoader(
#                         urls=[url],
#                         ssl_verify=False,
#                         dd_video_info=True,
#                         transcript_format=TranscriptFormat.CHUNKS,
#                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}
#                     )
#                 docs = loader.load()
#                 # repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
#                 # llm = HuggingFaceEndpoint(repo_id=repo_id,max_length=150,temperature=0.7,token=hf_token)

#                 chain = load_summarize_chain(llm=llm,chain_type="stuff",prompt=prompt)
#                 output_summary = chain.run(docs)

#                 st.success(output_summary)
        
#         except Exception as e:
#             st.exception(e)

import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.documents import Document
import re
from youtube_transcript_api import YouTubeTranscriptApi


st.title("Summarize Text from YT or Website")
st.subheader("Summarize URL")


with st.sidebar:
    hf_token = st.text_input("Enter HF API Token", value="", type="password")

url = st.text_input("URL", label_visibility="collapsed")

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=hf_token)

prompt_template = """
Provide summary of the following content in 300 words:
Content:{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

def extract_video_id(url):
    youtube_regex = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(youtube_regex, url)
    return match.group(1) if match else None

def get_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([t['text'] for t in transcript_list])
        return transcript_text
    except Exception as e:
        st.error(f"Error getting YouTube transcript: {e}")
        return None

if st.button("Summarize the content from YT or Website"):
    if not hf_token.strip() or not url.strip():
        st.error("Please provide the information")
    elif not validators.url(url):
        st.error("Please enter a valid a valid url.")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in url or "youtu.be" in url:
                    video_id = extract_video_id(url)
                    if not video_id:
                        st.error("Could not extract YouTube video ID from URL")
                    else:
                        transcript_text = get_youtube_transcript(video_id)
                        if transcript_text:
                            docs = [Document(page_content=transcript_text)]
                        else:
                            st.error("Failed to get YouTube transcript")
                            st.stop()
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}
                    )
                    docs = loader.load()
                
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        
        except Exception as e:
            st.exception(e)