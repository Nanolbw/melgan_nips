import glob
import os

import numpy as np
from matplotlib import pyplot as plt

from mel2wav.interface import MelVocoder
from pathlib import Path
from tqdm import tqdm
import argparse
import librosa
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    # args = parse_args()
    load_path='/Users/bowenliu/Desktop/melgan-neurips/models/multi_speaker.pt'
    vocoder = MelVocoder(load_path)
    save_path='/Users/bowenliu/Desktop/melgan-neurips/wav'
    # args.save_path.mkdir(exist_ok=True, parents=True)
    folder='/Users/bowenliu/Desktop/mindaudio/examples/deepspeech2/LibriSpeech/test-clean/61/70970'
    txt_files = glob.glob(os.path.join(folder, '*.flac'))
    for i, fname in tqdm(enumerate(txt_files)):
        print(fname)
        wavname = fname
        wav, sr = librosa.core.load(fname)
        print(torch.from_numpy(wav)[None].size())
        mel = vocoder(torch.from_numpy(wav)[None])
        recons = vocoder.inverse(mel).squeeze().cpu().numpy()
        import soundfile as sf

        save_filename = os.path.join(save_path, f'{i}.wav')
        print(save_filename)
        sf.write(save_filename, recons, sr)




if __name__ == "__main__":
    main()
