import re
import unicodedata

import gradio as gr
import joblib
import numpy as np
import spaces
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HF_REPO = "Thevyne/yoruba-sentiment-models"
NN_FOLDER = "afriberta_diacritics_removed"
LABELS = ["positive", "neutral", "negative"]   # ids 0, 1, 2

# ---- Text preparation: identical to the notebook (align_format, then strip_marks) ----
LINK_RE = re.compile(r"https?://\S+|www\.\S+|\bt\.co/\S+", flags=re.IGNORECASE)
HASHTAG_TOKEN_RE = re.compile(r"#\S+")
MENTION_TOKEN_RE = re.compile(r"@\S+")


def align_format(text, rule="all", repl=""):
    t = unicodedata.normalize("NFC", text)
    t = LINK_RE.sub(" ", t)
    t = re.sub(r"^\s*rt\b", " ", t, flags=re.IGNORECASE)
    t = HASHTAG_TOKEN_RE.sub(" ", t)
    t = MENTION_TOKEN_RE.sub(" ", t)
    t = t.lower()
    kept = []
    for ch in t:
        cat = unicodedata.category(ch)
        if ch.isspace() or cat[0] in "LMN":
            kept.append(ch)
        else:
            kept.append(repl)
    return " ".join("".join(kept).split())


def strip_marks(text):
    decomposed = unicodedata.normalize("NFD", text)
    kept = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", kept)


def prepare(text):
    return strip_marks(align_format(text))


# ---- Models ----
lr_model = joblib.load(hf_hub_download(HF_REPO, "lr_diacritics_removed.joblib"))
tokenizer = AutoTokenizer.from_pretrained(HF_REPO, subfolder=NN_FOLDER)
nn_model = AutoModelForSequenceClassification.from_pretrained(HF_REPO, subfolder=NN_FOLDER).eval()
nn_model.to("cuda")


@spaces.GPU
def predict(text):
    if not text or not text.strip():
        return "Type or paste a Yoruba tweet.", {}, {}
    clean = prepare(text)
    if not clean:
        return "Nothing left after cleaning (only links, mentions or symbols).", {}, {}

    lr_probs = lr_model.predict_proba([clean])[0]
    enc = tokenizer(clean, return_tensors="pt", truncation=True, max_length=128).to("cuda")
    with torch.no_grad():
        nn_probs = torch.softmax(nn_model(**enc).logits, dim=-1)[0].cpu().numpy()

    lr_out = {l: float(p) for l, p in zip(LABELS, lr_probs)}
    nn_out = {l: float(p) for l, p in zip(LABELS, nn_probs)}
    agree = LABELS[int(np.argmax(lr_probs))] == LABELS[int(np.argmax(nn_probs))]
    note = ("Both models agree." if agree else
            "The models disagree. Treat this prediction as uncertain.")
    return f"Text the models saw: {clean}\n{note}", lr_out, nn_out


with gr.Blocks(title="Yoruba Sentiment") as demo:
    gr.Markdown(
        "# Yoruba tweet sentiment\n"
        "Classifies a Yoruba tweet as positive, neutral or negative. Tone marks and underdots are "
        "removed before prediction, so text with or without them gives the same result. "
        "Main model: word + character TF-IDF logistic regression (test macro-F1 0.73 on unseen tweets). "
        "Second opinion: fine-tuned AfriBERTa (0.73). Sarcasm is not handled."
    )
    box = gr.Textbox(label="Yoruba text", lines=3, placeholder="Mo nifẹ ọja yii gan-an")
    btn = gr.Button("Classify")
    info = gr.Textbox(label="Prepared text and agreement", interactive=False)
    with gr.Row():
        lr_label = gr.Label(label="Logistic regression (main)")
        nn_label = gr.Label(label="AfriBERTa (second opinion)")
    btn.click(predict, box, [info, lr_label, nn_label])
    gr.Examples(["Mo nifẹ ọja yii gan-an", "Ó burú jù, mi ò fẹ́ rẹ̀", "Ọ̀la ni ìpàdé wa"], box)

demo.launch(ssr_mode=False)