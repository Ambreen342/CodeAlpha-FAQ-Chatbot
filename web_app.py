import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords
nltk.download("stopwords")

# Stop words
stop_words = set(stopwords.words("english"))


# FAQ questions
faq_questions = [
        "How long does delivery take?",
    "How can I contact customer support?",
    "Can I change my delivery address?",
    "What should I do if my order is damaged?",
    "Do you offer discounts?"
    "How can I place an order?",
    "How can I track my order?",
    "What is the return policy?",
    "What payment methods are supported?",
    "How do I cancel my order?"
]

# FAQ answers
faq_answers = [
        "Delivery usually takes 3 to 7 business days.",
    "You can contact customer support through our support page.",
    "You can change your delivery address before the order is shipped.",
    "Please contact customer support and report the damaged order.",
    "Yes, discounts are available during selected promotional periods."
    "Select a product, add it to your cart, and proceed to checkout.",
    "Open your account and select 'My Orders' to see your tracking information.",
    "Products can be returned within 30 days if they meet the return conditions.",
    "We support credit cards, debit cards, and online payment methods.",
    "Go to 'My Orders', select the order, and choose the cancel option."
]


# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Preprocess FAQ questions
processed_questions = [
    preprocess(question)
    for question in faq_questions
]


# Convert FAQ questions into TF-IDF vectors
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(processed_questions)


# Streamlit webpage
st.title("🛒 Online Shopping FAQ Chatbot")
st.sidebar.title("FAQ Topics")

st.sidebar.write("📦 Orders")
st.sidebar.write("🚚 Delivery & Tracking")
st.sidebar.write("↩️ Returns")
st.sidebar.write("💳 Payments")
st.sidebar.write("📞 Customer Support")

st.subheader("Your online shopping assistant")
st.info(
    "This chatbot uses Natural Language Processing (NLP), "
    "TF-IDF, and cosine similarity to find the best answer."
)

st.write(
    "Ask a question about orders, tracking, returns, payments, or cancellation."
)
st.write("### Example questions")

st.write("• How can I track my order?")
st.write("• What is the return policy?")
st.write("• What payment methods are supported?")
st.write("• How long does delivery take?")
st.write("• How do I cancel my order?")


# User input
user_question = st.chat_input("Type your question here...")


# Process the question
if user_question:

    processed_user_question = preprocess(user_question)

    user_vector = vectorizer.transform([processed_user_question])

    similarities = cosine_similarity(user_vector, faq_vectors)

    best_match = similarities.argmax()

    best_score = similarities[0][best_match]


    # Similarity threshold
    if best_score >= 0.3:
        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(faq_answers[best_match])
    else:
        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(
            "Sorry, I don't know the answer to that question."
        )