import gradio as gr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax

# Initialize models
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Initialize Transformers sentiment analysis pipeline
transformer_pipeline = pipeline("sentiment-analysis")

def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

def analyze_sentiment(text):
    vader_result = sia.polarity_scores(text)
    roberta_result = polarity_scores_roberta(text)
    transformer_result = transformer_pipeline(text)[0]
    
    combined_results = {
        **vader_result,
        **roberta_result,
        'transformer_label': transformer_result['label'],
        'transformer_score': transformer_result['score']
    }
    
    return combined_results

# Create Gradio interface
iface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter Text"),
    outputs=gr.JSON(label="Sentiment Scores"),
    title="Sentiment Analysis with VADER, RoBERTa, and Transformers",
    description="Enter a text to analyze its sentiment using VADER, RoBERTa, and Transformers models."
)

iface.launch()