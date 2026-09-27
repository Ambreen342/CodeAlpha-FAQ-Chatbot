import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

faq_questions = [
    "How can I place an order?",
    "How can I track my order?",
    "What is the return policy?",
    "What payment methods are supported?",
    "How do I cancel my order?"
]

print(faq_questions)
faq_answers = [
    "Select a product, add it to your cart, and proceed to checkout.",
    "Open your account and select 'My Orders' to see your tracking information.",
    "Products can be returned within 30 days if they meet the return conditions.",
    "We support credit cards, debit cards, and online payment methods.",
    "Go to 'My Orders', select the order, and choose the cancel option."
]

print(faq_answers)
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


test_question = "How can I track my order?"

print(preprocess(test_question))
processed_questions = [
    preprocess(question)
    for question in faq_questions
]

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(processed_questions)

print(faq_vectors.toarray())
print("Welcome to the Online Shopping FAQ Chatbot!")
print("Type 'exit' to quit.")
while True:
    user_question = input("Ask your question: ")

    if user_question.lower() == "exit":
        print("Goodbye!")
        break

    processed_user_question = preprocess(user_question)

    user_vector = vectorizer.transform([processed_user_question])

    similarities = cosine_similarity(user_vector, faq_vectors)

    best_match = similarities.argmax()
    best_score = similarities[0][best_match]

    if best_score >= 0.3:
        print("Answer:")
        print(faq_answers[best_match])
    else:
        print("Sorry, I don't know the answer to that question.")