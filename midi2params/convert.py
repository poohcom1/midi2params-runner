import os
import warnings
import torch
import scipy.io.wavfile

from midi2params.train_utils import midi2params, load_config
from midi2params.datasets import load_midi_file
from midi2params.util import load_ddsp_model, synthesize_ddsp_audio
from midi2params.models import SeqModel


def convert(model_path, ckpt_path, midi_path, audio_file_name, config_path):
    config = load_config(config_path)

    ddsp_model = load_ddsp_model(ckpt_path)

    midi_files = []

    if os.path.isfile(midi_path):
        midi_files.append(midi_path)
    else:
        for file in os.listdir(midi_path):
            midi_files.append(os.path.join(midi_path, file))

        print("Directory found! Converting %d files!" % len(midi_files))

    for i, file in enumerate(midi_files):
        print("Loading midi filed: " + file)
        pitches, onset_arr, offset_arr = load_midi_file(file)

        batch = {}

        pitches = torch.Tensor([pitches])
        onset_arr = torch.Tensor([onset_arr])
        offset_arr = torch.Tensor([offset_arr])

        if torch.cuda.is_available():
            print("Cuda available")
            pitches = pitches.cuda()
            onset_arr = onset_arr.cuda()
            offset_arr = offset_arr.cuda()

            print("Cuda set")

        batch['pitches'] = pitches
        batch['onset_arr'] = onset_arr
        batch['offset_arr'] = offset_arr

        print("Getting params...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            best_model = SeqModel(config)
            best_model.load_state_dict(torch.load(model_path))

            if torch.cuda.is_available():
                best_model.cuda()

            f0_pred, ld_pred = midi2params(best_model, batch)

        train_params = {
            'f0_hz': f0_pred[0],
            'loudness_db': ld_pred[0]
        }

        # Resynthesize parameters
        print("Resynthesizing...")
        new_model_resynth = synthesize_ddsp_audio(ddsp_model, train_params)

        file_name = audio_file_name
        if not audio_file_name:
            file_name = ''.join(os.path.basename(file).split('.')[:-1]) + ".wav"
        elif len(midi_files) > 1:
            file_name = ''.join(audio_file_name.split('.')[:-1]) + i + ".wav"
        else:
            file_name = audio_file_name

        scipy.io.wavfile.write(
            file_name, 16000, -new_model_resynth.swapaxes(0, 1)[0])
        print("Finished! Audio saved to %s." % file_name)
