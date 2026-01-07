import torch
from ModelDecoder import TransformerStackedDecoder
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from PreProcess import ctrl_expressions as ctrl_expressions_list
from AudioDataset import WavSample， pack_exp

import librosa
import matplotlib.pyplot as plt
import numpy as np
SECONDS = 23 
StartTransitionDelta = 10



def MainInferAdvance(wav_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("./wav2vec2-base-960h")

    wave_data, sr = librosa.load(wav_path, sr=16000)
    segment_len = sr*5
    segment_num = (len(wave_data)//segment_len) + 1

    decoderModel = TransformerStackedDecoder(
        input_dim=768,
        output_dim=136,
        num_heads=16,
        num_layers=9
    ).to(device)

    res = []
    for i in range(segment_num):
        start_point = segment_len * i
        wavSample = wave_data[start_point:start_point + segment_len]
        if len(wavSample) < segment_len:
            wavSample = np.pad(wavSample, (0, segment_len - len(wavSample)), mode='constant', constant_values=0)

        inputs = processor(wavSample, sampling_rate=sr, return_tensors="pt", padding=True)
        wav_feat = model(**inputs).last_hidden_state.to(device)

        state_dict = torch.load("./Weights/transformer_decoder_V3.pth", map_location=device)
        decoderModel.load_state_dict(state_dict)
        decoderModel.to(device)
        decoderModel.eval()

        pred = decoderModel(wav_feat)
        pred = pred.squeeze(0).cpu().numpy()
        pred = pred.tolist()

        res.append(pred)

    r = []
    for p in res:
        for v in p:
            r.append(v)

    r = mergeProcess(r)
    print("the length of : {}".format(len(r)))
    pack_exp(r, "pred_test_padding.json")

def main_backup():
    print("___________")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("./wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("./wav2vec2-base-960h")

    wav_path = './Dataset/zhuboshuolianbo/wav/BV1A3H6z9Exx.wav'
    json_path = './Dataset/zhuboshuolianbo/json/CD_BV1A3H6z9Exx_1.json'


    testWavSample = WavSample(json_path=json_path,wav_path=wav_path,processor=processor,model=model)

    decoderModel = TransformerStackedDecoder(
        input_dim=768,
        output_dim=136
    ).to(device)

    state_dict = torch.load("./Weights/transformer_decoder.pth", map_location=device)
    decoderModel.load_state_dict(state_dict)
    decoderModel.to(device)
    decoderModel.eval()

    exp_list = ["CTRL_expressions_browDownL", "CTRL_expressions_browDownR"]
    mouth_ctrls = [name for name in ctrl_expressions_list if "mouth" in name]
    print(mouth_ctrls)
    print(len(mouth_ctrls))
    testWavSample.plot_compare(5,channel=mouth_ctrls,decoder=decoderModel)


if __name__ == '__main__':
    wav_path = "./default.wav"
    MainInferAdvance(wav_path)
