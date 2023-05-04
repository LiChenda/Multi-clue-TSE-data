import os
import hashlib
import sys
sys.path.append('dep/audioset_tagging_cnn/utils/')
sys.path.append('dep/audioset_tagging_cnn/pytorch/')

from dep.audioset_tagging_cnn.pytorch.models import *
from pytorch_utils import move_data_to_device
import config
import glob
import tqdm
import soundfile as sf
import kaldiio
from pathlib import Path
import numpy as np

device = torch.device('cuda') 


def load_model():
    sample_rate = 16000
    window_size = 512
    hop_size = 160
    mel_bins = 64
    fmin = 50
    fmax = 8000
    model_type = "Cnn14_16k"
    checkpoint_path = "./Cnn14_16k_mAP=0.438.pth"
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    return model

def audio_tagging(model, audio_path):
    waveform, _ = sf.read(audio_path)
    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    
    one_hot = np.zeros(clipwise_output.shape, dtype='float32')
    one_hot[np.argmax(clipwise_output)] = 1

    return one_hot

def download_model():
    # download ckpt for audioset_tagging_cnn (https://github.com/qiuqiangkong/audioset_tagging_cnn)
    CHECKPOINT_PATH="Cnn14_16k_mAP=0.438.pth"
    URL = "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
    MD5 = "362fc5ff18f1d6ad2f6d464b45893f2c"
    if not os.path.exists(CHECKPOINT_PATH) or hashlib.md5(open(CHECKPOINT_PATH,'rb').read()).hexdigest() != MD5:
        os.system(f"wget -O {CHECKPOINT_PATH} {URL}")



if __name__ == "__main__":
    download_model()
    model = load_model()

    folders = [Path("output/train"), Path("output/val"),Path("output/test"),Path("output/unseen")]

    for folder in folders:
        target_folder=folder / "tag_clue"
        os.makedirs(target_folder, exist_ok=True)


        utts = {}
        with open(folder/"s1.scp") as f:
            for line in f:
                key, file = line.strip().split()
                utts[key] = file
                
        feats_writer = kaldiio.WriteHelper(f"ark,scp:{target_folder/'feats.ark'},{target_folder/'feats.scp'}")

        for key in tqdm.tqdm(utts.keys()):

            emb = audio_tagging(model, utts[key])
            
            feats_writer[key] = emb

        feats_writer.close()


