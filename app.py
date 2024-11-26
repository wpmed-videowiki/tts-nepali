import os
import io
import scipy
import torch
from random import choice
from string import ascii_uppercase
from flask import Flask, request, send_file
from pydub import AudioSegment

from transformers import AutoTokenizer, AutoModelForTextToWaveform

tokenizer = AutoTokenizer.from_pretrained("tuskbyte/nepali_male_v1")
model = AutoModelForTextToWaveform.from_pretrained("tuskbyte/nepali_male_v1")
app = Flask(__name__)


def generate_random_string():
    return "".join(choice(ascii_uppercase) for i in range(12))


@app.route("/", methods=["POST"])
def convert_wav():
    text = request.json["text"]
    inputs = tokenizer(text, return_tensors="pt")

    filename = f"{generate_random_string()}.wav"
    mp3_filename = f"{generate_random_string()}.mp3"

    with torch.no_grad():
        output = model(**inputs).waveform
    # Random file name using random string

    scipy.io.wavfile.write(
        filename, rate=model.config.sampling_rate, data=output.T.float().numpy()
    )
    AudioSegment.from_wav(filename).export(mp3_filename, format="mp3")
    with open(mp3_filename, "rb") as f:
        audio = io.BytesIO(f.read())

    # Remove the files
    os.remove(filename)
    os.remove(mp3_filename)
    return send_file(audio, download_name=mp3_filename, as_attachment=True)


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)

