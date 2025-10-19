import streamlit as st
import anthropic
import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from typing import List, Dict, Any
import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Load environment variables
load_dotenv()

# Download NLTK data (only once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.title("Claude Chatbot")

# Sidebar for internet search toggle
with st.sidebar:
    st.header("Settings")
    enable_internet_search = st.checkbox("Enable Internet Search (SerpAPI)", value=False)
    if enable_internet_search:
        st.info("Internet search is enabled. The chatbot will search the web when needed.")

# ----------------- Credibility Scoring System -----------------

class HybridCredibilityFeatures:
    def __init__(self):
        self.credible_domains = {
            'nature.com': 1.0, 'science.org': 1.0, 'edu': 0.9,
            'gov': 0.9, 'org': 0.7, 'com': 0.5,
            'thelancet.com': 1.0, 'sciencedirect.com': 0.9,
            'springer.com': 0.9, 'ieee.org': 0.9, 'acm.org': 0.9,
            'nih.gov': 1.0, 'clinical-journal.com': 0.8,
            'who.int': 1.0, 'nejm.org': 1.0, 'jamanetwork.com': 0.9,
            'webmd.com': 0.6, 'wikipedia.org': 0.4,
            'blogspot.com': 0.2, 'youtube.com': 0.2
        }
        self.academic_sites = {
            'researchgate.net', 'academia.edu', 'scholar.google.com',
            'arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'jstor.org'
        }
        self.sid = SentimentIntensityAnalyzer()

    def extract_urls(self, text):
        url_pattern = r'https?://[^\s]+'
        return re.findall(url_pattern, text)

    def score_url(self, url):
        try:
            domain = urlparse(url).netloc.lower()
            base_domain = '.'.join(domain.split('.')[-2:])

            if domain in self.academic_sites:
                return 1.0
            for credible_domain, score in self.credible_domains.items():
                if credible_domain in domain or credible_domain in base_domain:
                    return score
            return 0.3  # Default for unknown
        except:
            return 0.0

    def get_link_score(self, text):
        urls = self.extract_urls(text)
        if urls:
            return max(self.score_url(url) for url in urls) * 100
        return 0.0

    def get_string_score(self, text):
        clean_sentence = re.sub(r'https?://[^\s]+', '', text)
        sentiment_score = self.sid.polarity_scores(clean_sentence)['compound']
        return round(100.0 - (abs(sentiment_score) * 90.0), 2)

# Train credibility model
@st.cache_resource
def train_credibility_model():
    fe = HybridCredibilityFeatures()

    # Training data
    training_data = {
        'sentence': [
            "A study from NEJM suggests a link between this drug and reduced heart disease. https://www.nejm.org/some-article-id",
            "My doctor says this supplement will boost my immune system. http://healthylifehacks.com/immune-booster",
            "A new research paper on coffee is available at Harvard. https://harvard.edu/research/coffee-health",
            "This new weight loss method is a miracle, as seen in this video. https://www.youtube.com/watch?v=12345",
            "Vaccines for children are safe and effective, per the CDC. https://www.cdc.gov/vaccinesafety/",
            "Detox tea benefits are explored in this blog. https://detox-guru.blogspot.com/2025/08/tea.html"
        ],
        'human_score': [90.0, 30.0, 95.0, 20.0, 100.0, 40.0]
    }
    train_df = pd.DataFrame(training_data)

    # Feature extraction
    train_df['link_score'] = train_df['sentence'].apply(fe.get_link_score)
    train_df['string_score'] = train_df['sentence'].apply(fe.get_string_score)

    # Normalization
    feature_scaler = MinMaxScaler()
    X_train = feature_scaler.fit_transform(train_df[['link_score', 'string_score']])

    target_scaler = MinMaxScaler(feature_range=(0,1))
    y_train = target_scaler.fit_transform(train_df[['human_score']])

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return fe, model, feature_scaler, target_scaler

# Initialize credibility scoring
fe, credibility_model, feature_scaler, target_scaler = train_credibility_model()

def score_search_result(link, title, snippet):
    """Score a search result based on credibility"""
    # Combine title, link, and snippet for scoring
    combined_text = f"{title} {snippet} {link}"

    link_score = fe.get_link_score(link)
    string_score = fe.get_string_score(combined_text)

    # Normalize and predict
    features = feature_scaler.transform([[link_score, string_score]])
    prediction_scaled = credibility_model.predict(features)
    prediction = target_scaler.inverse_transform(prediction_scaled.reshape(-1,1))

    # Clip to 0-100 range
    credibility_score = np.clip(prediction[0][0], 0, 100)

    return round(credibility_score, 1)

# Initialize the Anthropic client
@st.cache_resource
def get_anthropic_client():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("Please set your ANTHROPIC_API_KEY in the .env file")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)

client = get_anthropic_client()

def search_serpapi(query: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Search using SerpAPI for the given query and return the results.

    :param query: The search query string.
    :param api_key: Your SerpAPI key.
    :return: A list of search results.
    :raises Exception: For any errors during the request.
    """
    try:
        search = GoogleSearch({
            "q": query,
            "location": "Austin, Texas, United States",
            "api_key": api_key
        })
        results = search.get_dict()
        return results.get("organic_results", [])
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # If internet search is enabled, perform SerpAPI search first and display results
    search_results_response = []
    if enable_internet_search:
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if serpapi_key:
            try:
                with st.spinner("Searching the web..."):
                    search_results = search_serpapi(prompt, serpapi_key)
                    if search_results:
                        # Display search results in a separate container
                        with st.chat_message("assistant"):
                            st.markdown("**🔍 Web Search Results (with Credibility Scores):**")
                            for idx, result in enumerate(search_results[:5], 1):
                                title = result.get('title', 'No title')
                                link = result.get('link', '#')
                                snippet = result.get('snippet', '')

                                # Calculate credibility score
                                cred_score = score_search_result(link, title, snippet)

                                # Convert score to stars (1-5)
                                star_rating = max(1, min(5, round(cred_score / 20)))
                                stars = "⭐" * star_rating
                                empty_stars = "☆" * (5 - star_rating)

                                # Store for response
                                search_results_response.append({
                                    "title": title,
                                    "link": link,
                                    "snippet": snippet,
                                    "credibility_score": cred_score
                                })

                                # Display formatted search result with score
                                st.markdown(f"**{idx}. {title}**")
                                st.markdown(f"**Credibility:** {stars}{empty_stars} ({cred_score}/100)")
                                st.markdown(f"🔗 Link: {link}")
                                if snippet:
                                    st.markdown(f"📝 {snippet}")
                                st.markdown("---")
            except Exception as e:
                st.warning(f"Search failed: {e}")
        else:
            st.warning("Please set SERPAPI_API_KEY in your .env file to use internet search.")

    # Get response from Claude
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Prepare messages with search context if available
        if search_results_response:
            search_context = "\n\nHere are relevant web search results:\n"
            for idx, result in enumerate(search_results_response, 1):
                search_context += f"{idx}. Title: {result['title']}\n"
                search_context += f"   Link: {result['link']}\n"
                if result.get('snippet'):
                    search_context += f"   Snippet: {result['snippet']}\n"
                search_context += "\n"

            enhanced_messages = st.session_state.messages.copy()
            enhanced_messages[-1] = {
                "role": "user",
                "content": f"{prompt}\n{search_context}"
            }
        else:
            enhanced_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        # Stream the response from Claude
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=enhanced_messages,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5
            }]
        ) as stream:
            for text in stream.text_stream:
                full_response += text
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
