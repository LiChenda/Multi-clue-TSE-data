import glob
import tqdm
import soundfile as sf
import kaldiio
from pathlib import Path
from PIL import Image
import os

from transformers import AutoFeatureExtractor, SwinForMaskedImageModeling,SwinModel
import decord
decord.bridge.set_bridge('torch')


# Replace the following dirs with your AudioSet and AudioCaps data path here.
# Assume all .mp4 files are included in {audiocaps_dir} and {audioset_dir}
audio_caps = "/mnt/rblack/data/AudioCaps/videos/YouTubeVideoClips/"
audio_set = "/mnt/rblack/data/AudioSet/videos/"
assert os.path.exists(audio_caps) and os.path.exists(audio_set), "Please make sure that you have downloaded Audioset and AudioCaps videos."

videos = glob.glob(f'{audio_caps}/*/*.mp4')
videos = videos + glob.glob(f'{audio_set}/*.mp4')


id2anchors = {}
with open("./anchors/all.txt") as f:
    for line in f:
        line = line.strip().split()
        assert len(line) == 2
        wav_id, anchor = line
        anchor = float(anchor)
        if wav_id in id2anchors:
            assert anchor == id2anchors[wav_id]
        id2anchors[wav_id] = anchor

id2video = {}

for v in videos:
    uid = Path(v).name.replace('.mp4', "")
    id2video[uid] = v



print("Using base model to save GPU memory. In our paper, the model is the large one. swin-large-patch4-window7-224.")
model_name = "microsoft/swin-base-patch4-window7-224"
# model_name = "microsoft/swin-large-patch4-window7-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = SwinModel.from_pretrained(model_name)
model.cuda()



def get_feature(model, uid):
    

    anchor = float(id2anchors[uid])
    
    vr = decord.VideoReader(id2video[uid])
    fps = vr.get_avg_fps()
    
    duration = len(vr) / fps
    
    
    if anchor - 1.0 < 0:
        anchor = 1.01
    if anchor + 1.0 > duration:
        anchor = duration - 1.01
        
    start_t = int((anchor - 1.0) * fps)
    end_t = int((anchor + 1.0) * fps)
    
    
    anchor_frame = int(anchor * fps)
    
    frames = vr.get_batch(list(range(start_t, end_t))).numpy()
    
    rtv = []

    frames = [Image.fromarray(f) for f in frames[-60:]]
    
    inputs = feature_extractor(images=frames, return_tensors="pt")

    inputs = {k:v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)
            
    rtv = outputs.pooler_output.squeeze(0).detach().cpu().numpy()
    
    return rtv, frames[0]


if __name__ == "__main__":
    folders = [Path("output/train"), Path("output/val"),Path("output/test"),Path("output/unseen")]
    for folder in folders:

        target_folder=folder / "visual_clue"
        os.makedirs(target_folder, exist_ok=True)

        utts = {}
        with open(folder/"s1.scp") as f:
            for line in f:
                key, file = line.strip().split()
                utts[key] = file


        feats_writer = kaldiio.WriteHelper(f"ark,scp:{target_folder/'feats.ark'},{target_folder/'feats.scp'}")

        for key in tqdm.tqdm(utts.keys()):

            vid = key.split('_mix_')[0]
            feature, _ = get_feature(model, vid)
            feats_writer[key] = feature


        feats_writer.close()