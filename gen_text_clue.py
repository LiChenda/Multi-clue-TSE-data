import os
import kaldiio
import transformers
import json
from transformers import pipeline, AutoTokenizer
import numpy as np
from pathlib import Path
import tqdm



def load_caps():
    # this file was created with a pretrained audio caption model for anchor audios:
    # https://github.com/wsntxxn/AudioCaption
    caps = "metadata/caps.json"

    with open(caps, 'r') as f:
        rtv = json.load(f)
    return rtv


feature_extraction = pipeline('feature-extraction', model="distilroberta-base", tokenizer="distilroberta-base", device=0)

def get_feature(pipelie, text):
    feature = pipelie(text)
    feature = np.array(feature)
    
    return feature[0][1:-1]




if __name__ == "__main__":
    caps = load_caps()

    folders = [Path("output/train"), Path("output/val"),Path("output/test"),Path("output/unseen")]

    for folder in folders:
        target_folder=folder / "text_clue"
        os.makedirs(target_folder, exist_ok=True)
        utts = {}
        with open(folder/"s1.scp") as f:
            for line in f:
                key, file = line.strip().split()
                utts[key] = file
        
        feats_writer = kaldiio.WriteHelper(f"ark,scp:{target_folder/'feats.ark'},{target_folder/'feats.scp'}")

        for key in tqdm.tqdm(utts.keys()):
            vid = key.split('_mix_')[0]
            text = caps[vid]
            clue_emb = get_feature(feature_extraction, text)

            feats_writer[key] = clue_emb
        feats_writer.close()


