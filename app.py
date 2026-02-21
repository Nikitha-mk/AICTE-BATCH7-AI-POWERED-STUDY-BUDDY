import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate(prompt, max_len=300):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=max_len)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def explain(topic):
    return generate(f"Explain the topic '{topic}' clearly in simple language for students.")

def summarize(notes):
    return generate("Write a clear bullet-point summary:\n" + notes)

def quiz(topic):
    return generate(
        f"Create 5 multiple choice questions with answers about {topic}. "
        "Each question must have A B C D options."
    )

with gr.Blocks() as app:
    gr.Markdown("# 📚 AI-Powered Study Buddy")

    with gr.Tab("Explain"):
        t = gr.Textbox()
        o = gr.Textbox()
        gr.Button("Explain").click(explain, t, o)

    with gr.Tab("Summarize"):
        n = gr.Textbox(lines=6)
        s = gr.Textbox()
        gr.Button("Summarize").click(summarize, n, s)

    with gr.Tab("Quiz"):
        qin = gr.Textbox()
        qout = gr.Textbox(lines=10)
        gr.Button("Generate Quiz").click(quiz, qin, qout)

app.launch()