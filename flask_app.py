import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from urllib.parse import urlparse, parse_qs
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Initialize embeddings and LLM
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual API key

# Initialize embeddings
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize LLM
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama3-70b-8192"
)

# Function to extract video ID from URL
def extract_video_id(link):
    """
    Extract the video id from a YouTube video link.
    """
    parsed_url = urlparse(link)
    
    if parsed_url.netloc == "www.youtube.com":
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            return query_params["v"][0]
        else:
            return None
    elif parsed_url.netloc == "youtu.be":
        path = parsed_url.path
        if path.startswith("/"):
            path = path[1:]
        return path
    else:
        return None

# Function to get subtitles using YouTube Transcript API
def get_subtitles(video_id):
    op = YouTubeTranscriptApi.get_transcript(video_id)
    op_use = TextFormatter.format_transcript(op, op)
    return op_use

# Global variable to store subtitles
subtitles = ""
@app.route('/')
def home():
    return "Welcome to the YouTube Transcript and PDF Generator API!"

# Endpoint to provide the YouTube video link and fetch subtitles
@app.route('/set_video_link', methods=['GET'])
def set_video_link():
    global subtitles
    video_link = request.args.get('link')
    if video_link:
        video_id = extract_video_id(video_link)
        if video_id:
            subtitles = get_subtitles(video_id)
            return jsonify({"message": "Subtitles fetched successfully!"}), 200
        else:
            return jsonify({"error": "Invalid video link!"}), 400
    return jsonify({"error": "Link parameter missing!"}), 400

# Endpoint to ask questions based on the fetched subtitles
@app.route('/ask_question', methods=['GET'])
def ask_question():
    global subtitles
    qinput = request.args.get('qinput')
    if subtitles and qinput:
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question based on the provided context and the information present in LLM.
            <context>
            {context}
            </context>
            Question: {qinput}
            """
        )
        
        formatted_prompt = prompt.format(context=subtitles, qinput=qinput)

        response = llm.invoke(formatted_prompt)
        return jsonify({"response": response.content}), 200
    return jsonify({"error": "Subtitles not fetched or question missing!"}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
