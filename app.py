import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- MODEL ----------------

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ---------------- GENERATE FUNCTION ----------------

def generate(prompt, max_new_tokens=256):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generate("Explain Machine Learning in 5 bullet points"))
# ---------------- FEATURES ----------------

def explain_topic(topic):
    prompt = f"""
    Explain the topic '{topic}' in simple language.

    Include:
    - Definition
    - Key points
    - Examples
    - Applications

    Use bullet points.
    """
    return generate(prompt)


def summarize_notes(notes):
    prompt = f"""
    Summarize the following text.

    Requirements:
    - 8 bullet points
    - Keep important information only
    - Easy revision notes

    Text:
    {notes}
    """
    return generate(prompt)


def generate_quiz(topic):
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
    return generate(prompt, 400)


def generate_flashcards(topic):
    prompt = f"""
    Create 10 flashcards about {topic}.

    Format:

    Flashcard 1
    Front:
    Back:

    Flashcard 2
    Front:
    Back:
    """
    return generate(prompt, 400)

# ---------------- UI ----------------

css = """
.gradio-container {
    max-width: 1100px !important;
}

.main-title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    margin-bottom:10px;
}

.subtitle {
    text-align:center;
    color:gray;
    margin-bottom:20px;
}

footer {
    display:none;
}
"""

with gr.Blocks(
    title="AI Powered Study Buddy",
    theme=gr.themes.Soft(),
    css=css
) as app:

    gr.HTML("""
    <div class="main-title">📚 AI Powered Study Buddy</div>
    <div class="subtitle">
    Explain • Summarize • Quiz • Flashcards
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("📖 Explain"):
            topic = gr.Textbox(
                label="Topic",
                placeholder="Example: Machine Learning"
            )

            explain_output = gr.Textbox(
                label="Explanation",
                lines=15
            )

            gr.Button("Explain").click(
                explain_topic,
                topic,
                explain_output
            )

        with gr.Tab("📝 Summarize"):
            notes = gr.Textbox(
                label="Paste Notes",
                lines=10
            )

            summary_output = gr.Textbox(
                label="Summary",
                lines=15
            )

            gr.Button("Summarize").click(
                summarize_notes,
                notes,
                summary_output
            )

        with gr.Tab("❓ Quiz"):
            quiz_topic = gr.Textbox(
                label="Enter Topic"
            )

            quiz_output = gr.Textbox(
                label="Quiz with Answers",
                lines=20
            )

            gr.Button("Generate Quiz").click(
                generate_quiz,
                quiz_topic,
                quiz_output
            )

        with gr.Tab("🧠 Flashcards"):
            flash_topic = gr.Textbox(
                label="Enter Topic"
            )

            flash_output = gr.Textbox(
                label="Flashcards",
                lines=20
            )

            gr.Button("Generate Flashcards").click(
                generate_flashcards,
                flash_topic,
                flash_output
            )

    gr.Markdown("""
    ---
    ### Features
    ✅ Explain Concepts  
    ✅ Summarize Notes  
    ✅ Generate MCQs  
    ✅ Show Correct Answers  
    ✅ Create Flashcards  
    """)

app.launch(server_name="0.0.0.0", server_port=7860)
