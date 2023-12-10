from transformers import pipeline
import gradio as gr
from huggingface_hub import login
login(token='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', add_to_git_credential=True)


pipe = pipeline(model="dacavi/whisper-small-hi")  # change to "your-username/the-name-you-picked"

def transcribe(audio):
    text = pipe(audio)["text"]
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs="text",
    title="Whisper Small Hindi",
    description="Realtime demo for Hindi speech recognition using a fine-tuned Whisper small model.",
)

iface.launch()