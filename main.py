if __name__ == '__main__':
    import sys
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from midi2params.convert import convert

    if len(sys.argv) == 1:
        print("convert_midi [model] [DDSP] [config] [midi] [output_file]")
        sys.exit()

    model_path = sys.argv[1]
    ckpt_path = sys.argv[2]
    config_path = sys.argv[3]
    midi_path = sys.argv[4]
    audio_file_name = sys.argv[5] if len(sys.argv) > 4 else None

    convert(model_path, ckpt_path, midi_path, audio_file_name, config_path)
