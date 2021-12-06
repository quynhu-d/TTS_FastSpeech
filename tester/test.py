import torchaudio
from typing import List
from model import Vocoder, FastSpeechConfig, FastSpeech
import torch
from data import Batch


def test(vocoder: Vocoder, fconfig: FastSpeechConfig, model_path: str, transcript:List[str]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
    model = FastSpeech(1000, fconfig)
    vocoder = vocoder.to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    tokens, token_lengths = tokenizer(transcript)
    # for tokens_ in tokens:
    # tokens = pad_sequence([
    #             tokens_[0] for tokens_ in tokens
    # ]).transpose(0, 1)
    # token_lengths = torch.cat(token_lengths)
    test_batch = Batch(
        waveform=None, waveform_length=None,
        transcript=transcript, tokens=tokens, token_lengths=token_lengths
    )
    output = model(test_batch)
    reconstructed_wav = vocoder.inference(output.to(device))
    return reconstructed_wav


if __name__ == '__main__':
    transcript = [
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
    ]
    fconfig = FastSpeechConfig()
    vocoder = Vocoder()
    model_path = '/content/drive/MyDrive/TTS/saved/overfit/05-12-03-05-48/model.pth'
    wavs_pred = test(vocoder, fconfig, model_path, transcript)
