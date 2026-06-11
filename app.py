import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate(prompt, max_len=512):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_length=max_len,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------- EXPLAIN ----------------
def explain(topic):
    prompt = f"""
    Explain the topic: {topic}

    Requirements:
    - Simple language
    - Student friendly
    - Include examples
    - Use bullet points
    """
    return generate(prompt)


# ---------------- SUMMARIZE ----------------
def summarize(notes):
    prompt = f"""
    Summarize the following notes.

    Requirements:
    - Bullet points
    - Important concepts only
    - Easy exam revision format

    Notes:
    {notes}
    """
    return generate(prompt)


# ---------------- QUIZ ----------------
def quiz(topic):
    prompt = f"""
    Create 5 MCQ questions about {topic}.

    Format:

    Q1.
    A.
    B.
    C.
    D.

    Correct Answer:

    Repeat for all 5 questions.
    """
    return generate(prompt)


# ---------------- FLASHCARDS ----------------
def flashcards(topic):
    prompt = f"""
    Create 10 study flashcards about {topic}.

    Format:

    Front: Question
    Back: Answer

    Make them useful for exam preparation.
    """
    return generate(prompt)


with gr.Blocks() as app:

    gr.Markdown("# 📚 AI Study Buddy")

    # Explain Tab
    with gr.Tab("Explain"):
        inp = gr.Textbox(label="Topic")
        out = gr.Textbox(label="Explanation", lines=15)
        gr.Button("Explain").click(explain, inp, out)

    # Summary Tab
    with gr.Tab("Summarize"):
        notes = gr.Textbox(lines=10, label="Paste Notes")
        summary = gr.Textbox(lines=15, label="Summary")
        gr.Button("Summarize").click(summarize, notes, summary)

    # Quiz Tab
    with gr.Tab("Quiz"):
        topic = gr.Textbox(label="Topic")
        quiz_out = gr.Textbox(lines=20, label="Quiz")
        gr.Button("Generate Quiz").click(quiz, topic, quiz_out)

    # Flashcard Tab
    with gr.Tab("Flashcards"):
        flash_topic = gr.Textbox(label="Topic")
        flash_out = gr.Textbox(lines=20, label="Flashcards")
        gr.Button("Generate Flashcards").click(
            flashcards,
            flash_topic,
            flash_out
        )

app.launch()
