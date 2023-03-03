import argparse
import os

import torch
import torchaudio
import time

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Text to speak.', default="The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.")
    parser.add_argument('--voice', type=str, help='Selects the voice to use for generation. See options in voices/ directory (and add your own!) '
                                                 'Use the & character to join two voices together. Use a comma to perform inference on multiple voices.', default='random')
    parser.add_argument('--preset', type=str, help='Which voice preset to use.', default='standard')
    parser.add_argument('--output_path', type=str, help='Where to store outputs.', default='results/')
    parser.add_argument('--model_dir', type=str, help='Where to find pretrained model checkpoints. Tortoise automatically downloads these to .models, so this'
                                                      'should only be specified if you have custom checkpoints.', default=MODELS_DIR)
    parser.add_argument('--candidates', type=int, help='How many output candidates to produce per-voice.', default=3)
    parser.add_argument('--seed', type=int, help='Random seed which can be used to reproduce results.', default=None)
    parser.add_argument('--produce_debug_state', type=bool, help='Whether or not to produce debug_state.pth, which can aid in reproducing problems. Defaults to true.', default=True)
    parser.add_argument('--cvvp_amount', type=float, help='How much the CVVP model should influence the output.'
                                                          'Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)', default=.0)
    parser.add_argument('--temperature', type=float, help='The softmax temperature of the autoregressive model.', default=.8)
    
    parser.add_argument('--autoregressive_samples', type=int, help='umber of samples taken from the autoregressive model, all of which are filtered using CLVP. As Tortoise is a probabilistic model, more samples means a higher probability of creating something "great".')
    parser.add_argument('--diffusion_iterations', type=int, help='Number of diffusion steps to perform. [0,4000]. More steps means the network has more chances to iteratively refine the output, which should theoretically mean a higher quality output. Generally a value above 250 is not noticeably better, however.')

    args = parser.parse_args()

    if (hasattr(args, "autoregressive_samples") and args.autoregressive_samples is not None) or (hasattr(args, "diffusion_iterations") and args.diffusion_iterations is not None):
        del args.preset
    if hasattr(args, "preset"):
        del args.autoregressive_samples
        del args.diffusion_iterations


    os.makedirs(args.output_path, exist_ok=True)

    tts = TextToSpeech(models_dir=args.model_dir)

    selected_voices = args.voice.split(',')
    for k, selected_voice in enumerate(selected_voices):
        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]
        voice_samples, conditioning_latents = load_voices(voice_sel)

        if (hasattr(args, "autoregressive_samples") and args.autoregressive_samples is not None) or (hasattr(args, "diffusion_iterations") and args.diffusion_iterations is not None):
            gen, dbg_state = tts.tts_with_preset(args.text, k=args.candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                      use_deterministic_seed=args.seed, return_deterministic_state=True, cvvp_amount=args.cvvp_amount,
                                      temperature=args.temperature,
                                      num_autoregressive_samples=args.autoregressive_samples, diffusion_iterations=args.diffusion_iterations)
        else:
            gen, dbg_state = tts.tts_with_preset(args.text, k=args.candidates, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                      preset=args.preset, use_deterministic_seed=args.seed, return_deterministic_state=True, cvvp_amount=args.cvvp_amount,
                                      temperature=args.temperature)

        timestamp = int(time.time())
        outdir = f"{args.output_path}/{selected_voice}/{timestamp}/"

        os.makedirs(outdir, exist_ok=True)

        with open(os.path.join(outdir, f'input.txt'), 'w') as f:
            f.write(args.text)

        if isinstance(gen, list):
            for j, g in enumerate(gen):
                torchaudio.save(os.path.join(outdir, f'{k}_{j}.wav'), g.squeeze(0).cpu(), 24000)
        else:
            torchaudio.save(os.path.join(outdir, f'{k}.wav'), gen.squeeze(0).cpu(), 24000)

        if args.produce_debug_state:
            os.makedirs('debug_states', exist_ok=True)
            torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')

