import json
import pathlib
import librosa
import random
import numpy
import os
import tqdm
import soundfile as sf
random.seed(55)


# Replace the following dirs with your AudioSet and AudioCaps data path here.
# Assume all .mp4 or .wav files are included in {audiocaps_dir} and {audioset_dir}
audiocaps_dir = "/mnt/rblack/data/AudioCaps/videos/audios_16k/all/"
audioset_dir = "/mnt/rblack/data/AudioSet/videos/"
ontology_path = "/mnt/rblack/data/AudioSet/metadata/ontology.json"

assert os.path.exists(audiocaps_dir) and os.path.exists(audioset_dir) and os.path.exists(ontology_path), "Please make sure you have all data downloaded"

# Output path for simulation data
output_dir = "./output/"

SR=16000
CLIP_LEN = 2.0
HCLIP_LEN = CLIP_LEN / 2

anchors_file = "anchors/all.txt"
list_files = {
    "train": "lists/train.txt",
    "val": "lists/val.txt",
    "test": "lists/test.txt",
    "unseen": "lists/unseen.txt"
}


def read_audio(wav_id):
    # Here may need to be modified according to your own directory structure.
    if (pathlib.Path(audiocaps_dir) / f"{wav_id}.wav").exists():
        audio_path = pathlib.Path(audiocaps_dir) / f"{wav_id}.wav"
        pass
    elif (pathlib.Path(audioset_dir) / f"{wav_id}.mp4").exists():
        audio_path = pathlib.Path(audioset_dir) / f"{wav_id}.mp4"
        pass
    else:
        print(f"wav {wav_id} not founded!")
        return None
    
    audio, _ = librosa.load(audio_path, sr=SR)

    return audio


if __name__ == "__main__":

    with open(ontology_path, 'r') as f:
        ontology = json.load(f)
        
    id2tag = {}

    for cls in ontology:
        id2tag[cls['id']] = cls['name']


    anchors = {}

    with open(anchors_file) as f:
        for line in f:
            line = line.strip().split()
            assert len(line) == 2
            wav_id, anchor = line
            anchor = float(anchor)
            if wav_id in anchors:
                assert anchor == anchors[wav_id]
            anchors[wav_id] = anchor



    os.makedirs(output_dir, exist_ok=True)

    for folder in list_files:
        list_file = list_files[folder]
        os.makedirs(f"{output_dir}/{folder}/wavs", exist_ok=True)
        os.makedirs(f"{output_dir}/{folder}/wavs/s1", exist_ok=True)
        os.makedirs(f"{output_dir}/{folder}/wavs/s2", exist_ok=True)
        os.makedirs(f"{output_dir}/{folder}/wavs/mix", exist_ok=True)
        with open(list_file) as list_file, \
            open(f"{output_dir}/{folder}/wav.scp", 'w') as mix_scp, \
            open(f"{output_dir}/{folder}/s1.scp", 'w') as s1_scp, \
            open(f"{output_dir}/{folder}/s2.scp", 'w') as s2_scp:
            for line in tqdm.tqdm(list_file):
                wav_id_1, wav_id_2, gain_in_db = line.strip().split()

                s1, s2 = read_audio(wav_id_1), read_audio(wav_id_2)
                if s1 is None or s2 is None:
                    continue

                anchor_1, anchor_2 = anchors[wav_id_1], anchors[wav_id_2]


                st1 = int((anchor_1 * SR)) - int(SR * HCLIP_LEN)
                ed1 = int((anchor_1 * SR))+ int(SR * HCLIP_LEN)

                if st1 < 0:
                    st1 = 0
                    ed1 = 0 + int(CLIP_LEN * SR)
                elif ed1 > len(s1):
                    ed1 = len(s1)
                    st1 = ed1 - int(CLIP_LEN * SR)


                st2 = int((anchor_2 * SR)) - int(SR * HCLIP_LEN)
                ed2 = int((anchor_2 * SR)) + int(SR * HCLIP_LEN)

                if st2 < 0:
                    st2 = 0
                    ed2 = 0 + int(CLIP_LEN * SR)
                elif ed2 > len(s2):
                    ed2 = len(s2)
                    st2 = ed2 - int(CLIP_LEN * SR)

                
                s1 = s1[st1:ed1]
                s2 = s2[st2:ed2]

                eng1 = (s1 ** 2).sum() + 1e-6
                eng2 = (s2 ** 2).sum() + 1e-6

                s1 = s1 / numpy.sqrt(eng1)
                s2 = s2 / numpy.sqrt(eng2)

                gain_in_db = float(gain_in_db)
                gain = 10 ** (gain_in_db / 20.0)
                s1 = s1 * gain

                mix = s1 + s2

                clip_max = abs(mix).max() + 1e-6
                mix = mix / clip_max * 0.9
                s1 = s1 / clip_max * 0.9
                s2 = s2 / clip_max * 0.9

                uid = wav_id_1 + "_mix_" + wav_id_2

                s1_path = f'{output_dir}/{folder}/wavs/s1/{uid}.wav'
                s2_path = f'{output_dir}/{folder}/wavs/s2/{uid}.wav'
                mix_path = f'{output_dir}/{folder}/wavs/mix/{uid}.wav'

                sf.write(s1_path, s1, 16000)
                sf.write(s2_path, s2, 16000)
                sf.write(mix_path, mix, 16000)

                mix_scp.write(f"{uid} {mix_path}\n")
                s1_scp.write(f"{uid} {s1_path}\n")
                s2_scp.write(f"{uid} {s2_path}\n") 




