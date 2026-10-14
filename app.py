"""Gradio demo.  Run:  python app.py   (use share=True in Colab to get a public link)"""
import gradio as gr

from src.data import load_tickets, split
from src.pipeline import TicketAssistant

train, _ = split(load_tickets())
assistant = TicketAssistant(train)

EXAMPLES = [
    ["I was charged twice for my Najm Plus subscription. This is urgent."],
    ["طلبي رقم 48213 ما وصل وتأخر 9 أيام. أحتاج مساعدة الآن لو سمحتم!"],
    ["How can I change the email address on my account? No rush, just asking."],
    ["الـ app يعلق لما أحاول أدفع payment."],
]


def run(text: str):
    if not text or not text.strip():
        return "", "", "", [], "Please enter a ticket / الرجاء إدخال رسالة"
    r = assistant.analyze(text)
    cases = [[c["id"], c["category"], c["score"], c["text"]] for c in r["similar_cases"]]
    return r["category"], r["priority"], r["intent"], cases, r["draft_reply"]


with gr.Blocks(title="Intelligent Support Ticket Assistant") as demo:
    gr.Markdown("# Intelligent Support Ticket Assistant\nEnglish / العربية — classify, prioritize, find similar cases, draft a grounded reply.")
    inp = gr.Textbox(label="Customer message / رسالة العميل", lines=4)
    btn = gr.Button("Analyze / تحليل", variant="primary")
    with gr.Row():
        cat = gr.Textbox(label="Category")
        pri = gr.Textbox(label="Priority")
        intent = gr.Textbox(label="Intent")
    sims = gr.Dataframe(headers=["id", "category", "similarity", "ticket"], label="Similar historical cases")
    reply = gr.Textbox(label="Draft reply (grounded on similar cases)", lines=6)
    btn.click(run, inp, [cat, pri, intent, sims, reply])
    gr.Examples(EXAMPLES, inputs=inp)

if __name__ == "__main__":
    demo.launch()
