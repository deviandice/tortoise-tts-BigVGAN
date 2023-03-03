import os
import random
import uuid
import gc

from time import time
from urllib import request

import torch
import torch.nn.functional as F
import progressbar
import torchaudio

from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from tortoise.models.diffusion_decoder import DiffusionTts
from tortoise.models.autoregressive import UnifiedVoice
from tqdm import tqdm

from tortoise.models.arch_util import TorchMelSpectrogram
from tortoise.models.clvp import CLVP
from tortoise.models.cvvp import CVVP
from tortoise.models.random_latent_generator import RandomLatentConverter
from tortoise.models.models import BigVGAN
from tortoise.utils.audio import wav_to_univnet_mel, denormalize_tacotron_mel
from tortoise.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from tortoise.utils.tokenizer import VoiceBpeTokenizer
from tortoise.utils.wav2vec_alignment import Wav2VecAlignment

from tortoise.utils.device import get_device, get_device_name, get_device_batch_size

pbar = None
STOP_SIGNAL = False
MODELS_DIR = os.environ.get('TORTOISE_MODELS_DIR', os.path.realpath(os.path.join(os.getcwd(), './models/tortoise/')))
MODELS = {
    'autoregressive.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/autoregressive.pth',
    'classifier.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/classifier.pth',
    'clvp2.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/clvp2.pth',
    'cvvp.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/cvvp.pth',
    'diffusion_decoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/diffusion_decoder.pth',
    'rlg_auto.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_auto.pth',
    'rlg_diffuser.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_diffuser.pth',
    'g_05000000.pt': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/g_05000000.pt',
}


def hash_file(path, algo="md5", buffer_size=0):
    import hashlib

    hash = None
    if algo == "md5":
        hash = hashlib.md5()
    elif algo == "sha1":
        hash = hashlib.sha1()
    else:
        raise Exception(f'Unknown hash algorithm specified: {algo}')

    if not os.path.exists(path):
        raise Exception(f'Path not found: {path}')

    with open(path, 'rb') as f:
        if buffer_size > 0:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                hash.update(data)
        else:
            hash.update(f.read())

    return "{0}".format(hash.hexdigest())


def check_for_kill_signal():
    global STOP_SIGNAL
    if STOP_SIGNAL:
        STOP_SIGNAL = False
        raise Exception("Kill signal detected")


def tqdm_override(arr, verbose=False, progress=None, desc=None):
    check_for_kill_signal()

    if verbose and desc is not None:
        print(desc)

    if progress is None:
        return tqdm(arr, disable=not verbose)
    return progress.tqdm(arr, desc=f'{progress.msg_prefix} {desc}' if hasattr(progress, 'msg_prefix') else desc,
                         track_tqdm=True)


def download_models(specific_models=None):
    """
    Call to download all the models that Tortoise uses.
    """

    os.makedirs(MODELS_DIR, exist_ok=True)

    def show_progress(block_num, block_size, total_size):
        global pbar
        if pbar is None:
            pbar = progressbar.ProgressBar(maxval=total_size)
            pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None

    for model_name, url in MODELS.items():
        if specific_models is not None and model_name not in specific_models:
            continue
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            continue
        print(f'Downloading {model_name} from {url}...')
        request.urlretrieve(url, model_path, show_progress)
        print('Done.')


def get_model_path(model_name, models_dir=MODELS_DIR):
    """
    Get path to given model, download it if it doesn't exist.
    """
    #  if model_name not in MODELS:
        #  raise ValueError(f'Model {model_name} not found in available models.')
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path) and models_dir == MODELS_DIR:
        download_models([model_name])
    return model_path


def pad_or_truncate(t, length):
    """
    Utility function for forcing <t> to have the specified sequence length, whether by clipping it or padding it with 0s.
    """
    if t.shape[-1] == length:
        return t
    elif t.shape[-1] < length:
        return F.pad(t, (0, length - t.shape[-1]))
    else:
        return t[..., :length]


def load_discrete_vocoder_diffuser(trained_diffusion_steps=4000, desired_diffusion_steps=200, cond_free=True,
                                   cond_free_k=1):
    """
    Helper function to load a GaussianDiffusion instance configured for use as a vocoder.
    """
    return SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]),
                           model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse',
                           betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=cond_free, conditioning_free_k=cond_free_k)


def format_conditioning(clip, cond_length=132300, device='cuda', sampling_rate=22050):
    """
    Converts the given conditioning signal to a MEL spectrogram and clips it as expected by the models.
    """
    gap = clip.shape[-1] - cond_length
    if gap < 0:
        clip = F.pad(clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        clip = clip[:, rand_start:rand_start + cond_length]
    mel_clip = TorchMelSpectrogram(sampling_rate=sampling_rate)(clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).to(device)


def fix_autoregressive_output(codes, stop_token, complain=True):
    """
    This function performs some padding on coded audio that fixes a mismatch issue between what the diffusion model was
    trained on and what the autoregressive code generator creates (which has no padding or end).
    This is highly specific to the DVAE being used, so this particular coding will not necessarily work if used with
    a different DVAE. This can be inferred by feeding a audio clip padded with lots of zeros on the end through the DVAE
    and copying out the last few codes.

    Failing to do this padding will produce speech with a harsh end that sounds like "BLAH" or similar.
    """
    # Strip off the autoregressive stop token and add padding.
    stop_token_indices = (codes == stop_token).nonzero()
    if len(stop_token_indices) == 0:
        if complain:
            print("No stop tokens found in one of the generated voice clips. This typically means the spoken audio is "
                  "too long. In some cases, the output will still be good, though. Listen to it and if it is missing words, "
                  "try breaking up your input text.")
        return codes
    else:
        codes[stop_token_indices] = 83
    stm = stop_token_indices.min().item()
    codes[stm:] = 83
    if stm - 3 < codes.shape[0]:
        codes[-3] = 45
        codes[-2] = 45
        codes[-1] = 248

    return codes


def do_spectrogram_diffusion(diffusion_model, diffuser, latents, conditioning_latents, temperature=1, verbose=True,
                             progress=None, desc=None, sampler="P", input_sample_rate=22050, output_sample_rate=24000):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        output_seq_len = latents.shape[
                             1] * 4 * output_sample_rate // input_sample_rate  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_shape = (latents.shape[0], 100, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(latents, conditioning_latents, output_seq_len,
                                                                      False)

        noise = torch.randn(output_shape, device=latents.device) * temperature

        diffuser.sampler = sampler.lower()
        mel = diffuser.sample_loop(diffusion_model, output_shape, noise=noise,
                                   model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
                                   verbose=verbose, progress=progress, desc=desc)

        mel = denormalize_tacotron_mel(mel)[:, :, :output_seq_len]
        if get_device_name() == "dml":
            mel = mel.cpu()
        return mel


def classify_audio_clip(clip):
    """
    Returns whether or not Tortoises' classifier thinks the given clip came from Tortoise.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: True if the clip was classified as coming from Tortoise and false if it was classified as real.
    """
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    classifier.load_state_dict(torch.load(get_model_path('classifier.pth'), map_location=torch.device('cpu')))
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

class TextToSpeech:
    """
    Main entry point into Tortoise.
    """

    def __init__(self, autoregressive_batch_size=None, models_dir=MODELS_DIR, enable_redaction=True, device=None,
                 minor_optimizations=True, input_sample_rate=22050, output_sample_rate=24000,
                 autoregressive_model_path=None):
        """
        Constructor
        :param autoregressive_batch_size: Specifies how many samples to generate per batch. Lower this if you are seeing
                                          GPU OOM errors. Larger numbers generates slightly faster.
        :param models_dir: Where model weights are stored. This should only be specified if you are providing your own
                           models, otherwise use the defaults.
        :param enable_redaction: When true, text enclosed in brackets are automatically redacted from the spoken output
                                 (but are still rendered by the model). This can be used for prompt engineering.
                                 Default is true.
        :param device: Device to use when running the model. If omitted, the device will be automatically chosen.
        """
        if device is None:
            device = get_device(verbose=True)

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.minor_optimizations = minor_optimizations

        # for clarity, it's simpler to split these up and just predicate them on requesting VRAM-consuming optimizations
        self.preloaded_tensors = minor_optimizations
        self.use_kv_cache = minor_optimizations
        if get_device_name() == "dml":  # does not work with DirectML
            print("KV caching requested but not supported with the DirectML backend, disabling...")
            self.use_kv_cache = False

        self.models_dir = models_dir
        self.autoregressive_batch_size = get_device_batch_size() if autoregressive_batch_size is None or autoregressive_batch_size == 0 else autoregressive_batch_size
        self.enable_redaction = enable_redaction
        self.device = device
        if self.enable_redaction:
            self.aligner = Wav2VecAlignment(device='cpu' if get_device_name() == "dml" else self.device)

        self.tokenizer = VoiceBpeTokenizer()

        self.autoregressive_model_path = autoregressive_model_path if autoregressive_model_path and os.path.exists(
            autoregressive_model_path) else get_model_path('autoregressive.pth', models_dir)
        self.autoregressive_model_hash = hash_file(self.autoregressive_model_path)

        if os.path.exists(f'{models_dir}/autoregressive.ptt'):
            # Assume this is a traced directory.
            self.autoregressive = torch.jit.load(f'{models_dir}/autoregressive.ptt')
            self.diffusion = torch.jit.load(f'{models_dir}/diffusion_decoder.ptt')
        else:
            self.autoregressive = UnifiedVoice(max_mel_tokens=604, max_text_tokens=402, max_conditioning_inputs=2,
                                               layers=30,
                                               model_dim=1024,
                                               heads=16, number_text_tokens=255, start_text_token=255,
                                               checkpointing=False,
                                               train_solo_embeddings=False).cpu().eval()
            self.autoregressive.load_state_dict(torch.load(self.autoregressive_model_path))
            self.autoregressive.post_init_gpt2_config(kv_cache=self.use_kv_cache)

            self.diffusion = DiffusionTts(model_channels=1024, num_layers=10, in_channels=100, out_channels=200,
                                          in_latent_channels=1024, in_tokens=8193, dropout=0, use_fp16=False,
                                          num_heads=16,
                                          layer_drop=0, unconditioned_percentage=0).cpu().eval()
            self.diffusion.load_state_dict(torch.load(get_model_path('diffusion_decoder.pth', models_dir)))

        self.clvp = CLVP(dim_text=768, dim_speech=768, dim_latent=768, num_text_tokens=256, text_enc_depth=20,
                         text_seq_len=350, text_heads=12,
                         num_speech_tokens=8192, speech_enc_depth=20, speech_heads=12, speech_seq_len=430,
                         use_xformers=True).cpu().eval()
        self.clvp.load_state_dict(torch.load(get_model_path('clvp2.pth', models_dir)))
        self.cvvp = None  # CVVP model is only loaded if used.

        self.vocoder = BigVGAN().cpu()
        state_dict_g = load_checkpoint(get_model_path('g_05000000.pt', models_dir), device)
        self.vocoder.load_state_dict(state_dict_g['generator'])
        self.vocoder.eval(inference=True)

        # Random latent generators (RLGs) are loaded lazily.
        self.rlg_auto = None
        self.rlg_diffusion = None

        if self.preloaded_tensors:
            self.autoregressive = self.autoregressive.to(self.device)
            self.diffusion = self.diffusion.to(self.device)
            self.clvp = self.clvp.to(self.device)
            self.vocoder = self.vocoder.to(self.device)

    def load_autoregressive_model(self, autoregressive_model_path):
        previous_path = self.autoregressive_model_path
        self.autoregressive_model_path = autoregressive_model_path if autoregressive_model_path and os.path.exists(
            autoregressive_model_path) else get_model_path('autoregressive.pth', self.models_dir)
        self.autoregressive_model_hash = hash_file(self.autoregressive_model_path)

        del self.autoregressive
        self.autoregressive = UnifiedVoice(max_mel_tokens=604, max_text_tokens=402, max_conditioning_inputs=2,
                                           layers=30,
                                           model_dim=1024,
                                           heads=16, number_text_tokens=255, start_text_token=255, checkpointing=False,
                                           train_solo_embeddings=False).cpu().eval()
        self.autoregressive.load_state_dict(torch.load(self.autoregressive_model_path))
        self.autoregressive.post_init_gpt2_config(kv_cache=self.use_kv_cache)
        if self.preloaded_tensors:
            self.autoregressive = self.autoregressive.to(self.device)

        return previous_path != self.autoregressive_model_path




    def load_cvvp(self):
        """Load CVVP model."""
        self.cvvp = CVVP(model_dim=512, transformer_heads=8, dropout=0, mel_codes=8192, conditioning_enc_depth=8,
                         cond_mask_percentage=0,
                         speech_enc_depth=8, speech_mask_percentage=0, latent_multiplier=1).cpu().eval()
        self.cvvp.load_state_dict(torch.load(get_model_path('cvvp.pth', self.models_dir)))

        if self.preloaded_tensors:
            self.cvvp = self.cvvp.to(self.device)

    def get_conditioning_latents(self, voice_samples, return_mels=False, verbose=False, progress=None, slices=1,
                                 max_chunk_size=None, force_cpu=False):
        """
        Transforms one or more voice_samples into a tuple (autoregressive_conditioning_latent, diffusion_conditioning_latent).
        These are expressive learned latents that encode aspects of the provided clips like voice, intonation, and acoustic
        properties.
        :param voice_samples: List of 2 or more ~10 second reference clips, which should be torch tensors containing 22.05kHz waveform data.
        """
        with torch.no_grad():
            # computing conditional latents requires being done on the CPU if using DML because M$ still hasn't implemented some core functions
            if get_device_name() == "dml":
                force_cpu = True
            device = torch.device('cpu') if force_cpu else self.device

            if not isinstance(voice_samples, list):
                voice_samples = [voice_samples]

            voice_samples = [v.to(device) for v in voice_samples]

            resampler = torchaudio.transforms.Resample(
                self.input_sample_rate,
                self.output_sample_rate,
                lowpass_filter_width=16,
                rolloff=0.85,
                resampling_method="kaiser_window",
                beta=8.555504641634386,
            )

            samples = []
            auto_conds = []
            for sample in voice_samples:
                auto_conds.append(format_conditioning(sample, device=device, sampling_rate=self.input_sample_rate))
                samples.append(resampler(sample.cpu()).to(
                    device))  # icky no good, easier to do the resampling on CPU than figure out how to do it on GPU

            auto_conds = torch.stack(auto_conds, dim=1)

            self.autoregressive = self.autoregressive.to(device)
            auto_latent = self.autoregressive.get_conditioning(auto_conds)
            if self.preloaded_tensors:
                self.autoregressive = self.autoregressive.to(self.device)
            else:
                self.autoregressive = self.autoregressive.cpu()

            diffusion_conds = []
            chunks = []

            concat = torch.cat(samples, dim=-1)
            chunk_size = concat.shape[-1]

            if slices == 0:
                slices = 1
            else:
                if max_chunk_size is not None and chunk_size > max_chunk_size:
                    slices = 1
                    while int(chunk_size / slices) > max_chunk_size:
                        slices = slices + 1

            chunks = torch.chunk(concat, slices, dim=1)
            chunk_size = chunks[0].shape[-1]

            # expand / truncate samples to match the common size
            # required, as tensors need to be of the same length
            for chunk in tqdm_override(chunks, verbose=verbose, progress=progress,
                                       desc="Computing conditioning latents..."):
                check_for_kill_signal()
                chunk = pad_or_truncate(chunk, chunk_size)
                cond_mel = wav_to_univnet_mel(chunk.to(device), do_normalization=False, device=device)
                diffusion_conds.append(cond_mel)

            diffusion_conds = torch.stack(diffusion_conds, dim=1)

            self.diffusion = self.diffusion.to(device)

            diffusion_latent = self.diffusion.get_conditioning(diffusion_conds)

            if self.preloaded_tensors:
                self.diffusion = self.diffusion.to(self.device)
            else:
                self.diffusion = self.diffusion.cpu()

        if return_mels:
            return auto_latent, diffusion_latent, auto_conds, diffusion_conds
        else:
            return auto_latent, diffusion_latent

    def get_random_conditioning_latents(self):
        # Lazy-load the RLG models.
        if self.rlg_auto is None:
            self.rlg_auto = RandomLatentConverter(1024).eval()
            self.rlg_auto.load_state_dict(
                torch.load(get_model_path('rlg_auto.pth', self.models_dir), map_location=torch.device('cpu')))
            self.rlg_diffusion = RandomLatentConverter(2048).eval()
            self.rlg_diffusion.load_state_dict(
                torch.load(get_model_path('rlg_diffuser.pth', self.models_dir), map_location=torch.device('cpu')))
        with torch.no_grad():
            return self.rlg_auto(torch.tensor([0.0])), self.rlg_diffusion(torch.tensor([0.0]))

    def tts_with_preset(self, text, preset='fast', **kwargs):
        """
        Calls TTS with one of a set of preset generation parameters. Options:
            'ultra_fast': Produces speech at a speed which belies the name of this repo. (Not really, but it's definitely fastest).
            'fast': Decent quality speech at a decent inference rate. A good choice for mass inference.
            'standard': Very good quality. This is generally about as good as you are going to get.
            'high_quality': Use if you want the absolute best. This is not really worth the compute, though.
        """
        # Use generally found best tuning knobs for generation.
        settings = {'temperature': .8, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
                    'top_p': .8,
                    'cond_free_k': 2.0, 'diffusion_temperature': 1.0}
        # Presets are defined here.
        presets = {
            'ultra_fast': {'num_autoregressive_samples': 16, 'diffusion_iterations': 30, 'cond_free': False},
            'fast': {'num_autoregressive_samples': 96, 'diffusion_iterations': 80},
            'standard': {'num_autoregressive_samples': 256, 'diffusion_iterations': 200},
            'high_quality': {'num_autoregressive_samples': 256, 'diffusion_iterations': 400},
        }
        settings.update(presets[preset])
        settings.update(kwargs)  # allow overriding of preset settings with kwargs
        return self.tts(text, **settings)

    def tts(self, text, voice_samples=None, conditioning_latents=None, k=1, verbose=True, use_deterministic_seed=None,
            return_deterministic_state=False,
            # autoregressive generation parameters follow
            num_autoregressive_samples=512, temperature=.8, length_penalty=1, repetition_penalty=2.0, top_p=.8,
            max_mel_tokens=500,
            sample_batch_size=None,
            # CVVP parameters follow
            cvvp_amount=.0,
            # diffusion generation parameters follow
            diffusion_iterations=100, cond_free=True, cond_free_k=2, diffusion_temperature=1.0,
            diffusion_sampler="P",
            breathing_room=8,
            half_p=False,
            progress=None,
            **hf_generate_kwargs):
        """
        Produces an audio clip of the given text being spoken with the given reference voice.
        :param text: Text to be spoken.
        :param voice_samples: List of 2 or more ~10 second reference clips which should be torch tensors containing 22.05kHz waveform data.
        :param conditioning_latents: A tuple of (autoregressive_conditioning_latent, diffusion_conditioning_latent), which
                                     can be provided in lieu of voice_samples. This is ignored unless voice_samples=None.
                                     Conditioning latents can be retrieved via get_conditioning_latents().
        :param k: The number of returned clips. The most likely (as determined by Tortoises' CLVP model) clips are returned.
        :param verbose: Whether or not to print log messages indicating the progress of creating a clip. Default=true.
        ~~AUTOREGRESSIVE KNOBS~~
        :param num_autoregressive_samples: Number of samples taken from the autoregressive model, all of which are filtered using CLVP.
               As Tortoise is a probabilistic model, more samples means a higher probability of creating something "great".
        :param temperature: The softmax temperature of the autoregressive model.
        :param length_penalty: A length penalty applied to the autoregressive decoder. Higher settings causes the model to produce more terse outputs.
        :param repetition_penalty: A penalty that prevents the autoregressive decoder from repeating itself during decoding. Can be used to reduce the incidence
                                   of long silences or "uhhhhhhs", etc.
        :param top_p: P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely" (aka boring) outputs.
        :param max_mel_tokens: Restricts the output length. (0,600] integer. Each unit is 1/20 of a second.
        :param typical_sampling: Turns typical sampling on or off. This sampling mode is discussed in this paper: https://arxiv.org/abs/2202.00666
                                 I was interested in the premise, but the results were not as good as I was hoping. This is off by default, but
                                 could use some tuning.
        :param typical_mass: The typical_mass parameter from the typical_sampling algorithm.
        ~~CLVP-CVVP KNOBS~~
        :param cvvp_amount: Controls the influence of the CVVP model in selecting the best output from the autoregressive model.
                            [0,1]. Values closer to 1 mean the CVVP model is more important, 0 disables the CVVP model.
        ~~DIFFUSION KNOBS~~
        :param diffusion_iterations: Number of diffusion steps to perform. [0,4000]. More steps means the network has more chances to iteratively refine
                                     the output, which should theoretically mean a higher quality output. Generally a value above 250 is not noticeably better,
                                     however.
        :param cond_free: Whether or not to perform conditioning-free diffusion. Conditioning-free diffusion performs two forward passes for
                          each diffusion step: one with the outputs of the autoregressive model and one with no conditioning priors. The output
                          of the two is blended according to the cond_free_k value below. Conditioning-free diffusion is the real deal, and
                          dramatically improves realism.
        :param cond_free_k: Knob that determines how to balance the conditioning free signal with the conditioning-present signal. [0,inf].
                            As cond_free_k increases, the output becomes dominated by the conditioning-free signal.
                            Formula is: output=cond_present_output*(cond_free_k+1)-cond_absenct_output*cond_free_k
        :param diffusion_temperature: Controls the variance of the noise fed into the diffusion model. [0,1]. Values at 0
                                      are the "mean" prediction of the diffusion network and will sound bland and smeared.
        ~~OTHER STUFF~~
        :param hf_generate_kwargs: The huggingface Transformers generate API is used for the autoregressive transformer.
                                   Extra keyword args fed to this function get forwarded directly to that API. Documentation
                                   here: https://huggingface.co/docs/transformers/internal/generation_utils
        :return: Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
                 Sample rate is 24kHz.
        """

        if get_device_name() == "dml" and half_p:
            print("Float16 requested but not supported with the DirectML backend, disabling...")
            half_p = False

        self.diffusion.enable_fp16 = half_p
        deterministic_seed = self.deterministic_state(seed=use_deterministic_seed)

        text_tokens = torch.IntTensor(self.tokenizer.encode(text)).unsqueeze(0).to(self.device)
        text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
        assert text_tokens.shape[
                   -1] < 400, 'Too much text provided. Break the text up into separate segments and re-try inference.'

        auto_conds = None
        if voice_samples is not None:
            auto_conditioning, diffusion_conditioning, auto_conds, _ = self.get_conditioning_latents(voice_samples,
                                                                                                     return_mels=True,
                                                                                                     verbose=True)
        elif conditioning_latents is not None:
            latent_tuple = conditioning_latents
            if len(latent_tuple) == 2:
                auto_conditioning, diffusion_conditioning = conditioning_latents
            else:
                auto_conditioning, diffusion_conditioning, auto_conds, _ = conditioning_latents
        else:
            auto_conditioning, diffusion_conditioning = self.get_random_conditioning_latents()

        diffuser = load_discrete_vocoder_diffuser(desired_diffusion_steps=diffusion_iterations, cond_free=cond_free,
                                                  cond_free_k=cond_free_k)

        self.autoregressive_batch_size = get_device_batch_size() if sample_batch_size is None or sample_batch_size == 0 else sample_batch_size

        with torch.no_grad():
            samples = []
            num_batches = num_autoregressive_samples // self.autoregressive_batch_size
            if num_autoregressive_samples < self.autoregressive_batch_size:
                num_autoregressive_samples = 1
            stop_mel_token = self.autoregressive.stop_mel_token
            calm_token = 83  # This is the token for coding silence, which is fixed in place with "fix_autoregressive_output"

            self.autoregressive = self.autoregressive.to(self.device)
            auto_conditioning = auto_conditioning.to(self.device)
            text_tokens = text_tokens.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=half_p):
                for b in tqdm_override(range(num_batches), verbose=verbose, progress=progress,
                                       desc="Generating autoregressive samples"):
                    check_for_kill_signal()
                    codes = self.autoregressive.inference_speech(auto_conditioning, text_tokens,
                                                                 do_sample=True,
                                                                 top_p=top_p,
                                                                 temperature=temperature,
                                                                 num_return_sequences=self.autoregressive_batch_size,
                                                                 length_penalty=length_penalty,
                                                                 repetition_penalty=repetition_penalty,
                                                                 max_generate_length=max_mel_tokens,
                                                                 **hf_generate_kwargs)
                    padding_needed = max_mel_tokens - codes.shape[1]
                    codes = F.pad(codes, (0, padding_needed), value=stop_mel_token)
                    samples.append(codes)

            if not self.preloaded_tensors:
                self.autoregressive = self.autoregressive.cpu()
                auto_conditioning = auto_conditioning.cpu()

            clip_results = []

            if auto_conds is not None:
                auto_conds = auto_conds.to(self.device)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=half_p):
                if not self.minor_optimizations:
                    self.autoregressive = self.autoregressive.cpu()
                    self.clvp = self.clvp.to(self.device)

                if cvvp_amount > 0:
                    if self.cvvp is None:
                        self.load_cvvp()
                    if not self.minor_optimizations:
                        self.cvvp = self.cvvp.to(self.device)

                desc = "Computing best candidates"
                if verbose:
                    if self.cvvp is None:
                        desc = "Computing best candidates using CLVP"
                    else:
                        desc = f"Computing best candidates using CLVP {((1 - cvvp_amount) * 100):2.0f}% and CVVP {(cvvp_amount * 100):2.0f}%"

                for batch in tqdm_override(samples, verbose=verbose, progress=progress, desc=desc):
                    check_for_kill_signal()
                    for i in range(batch.shape[0]):
                        batch[i] = fix_autoregressive_output(batch[i], stop_mel_token)

                    if cvvp_amount != 1:
                        clvp = self.clvp(text_tokens.repeat(batch.shape[0], 1), batch, return_loss=False)

                    if auto_conds is not None and cvvp_amount > 0:
                        cvvp_accumulator = 0
                        for cl in range(auto_conds.shape[1]):
                            cvvp_accumulator = cvvp_accumulator + self.cvvp(
                                auto_conds[:, cl].repeat(batch.shape[0], 1, 1), batch, return_loss=False)
                        cvvp = cvvp_accumulator / auto_conds.shape[1]
                        if cvvp_amount == 1:
                            clip_results.append(cvvp)
                        else:
                            clip_results.append(cvvp * cvvp_amount + clvp * (1 - cvvp_amount))
                    else:
                        clip_results.append(clvp)

            if not self.preloaded_tensors and auto_conds is not None:
                auto_conds = auto_conds.cpu()

            clip_results = torch.cat(clip_results, dim=0)
            samples = torch.cat(samples, dim=0)
            best_results = samples[torch.topk(clip_results, k=k).indices]

            if not self.preloaded_tensors:
                self.clvp = self.clvp.cpu()
                if self.cvvp is not None:
                    self.cvvp = self.cvvp.cpu()

            del samples

            if get_device_name() == "dml":
                text_tokens = text_tokens.cpu()
                best_results = best_results.cpu()
                auto_conditioning = auto_conditioning.cpu()
                self.autoregressive = self.autoregressive.cpu()
            else:
                # text_tokens = text_tokens.to(self.device)
                # best_results = best_results.to(self.device)
                auto_conditioning = auto_conditioning.to(self.device)
                self.autoregressive = self.autoregressive.to(self.device)

            # The diffusion model actually wants the last hidden layer from the autoregressive model as conditioning
            # inputs. Re-produce those for the top results. This could be made more efficient by storing all of these
            # results, but will increase memory usage.
            best_latents = self.autoregressive(auto_conditioning.repeat(k, 1), text_tokens.repeat(k, 1),
                                               torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                                               best_results,
                                               torch.tensor([best_results.shape[
                                                                 -1] * self.autoregressive.mel_length_compression],
                                                            device=text_tokens.device),
                                               return_latent=True, clip_inputs=False)

            diffusion_conditioning = diffusion_conditioning.to(self.device)

            if get_device_name() == "dml":
                self.autoregressive = self.autoregressive.to(self.device)
                best_results = best_results.to(self.device)
                best_latents = best_latents.to(self.device)

                self.vocoder = self.vocoder.cpu()
            else:
                if not self.preloaded_tensors:
                    self.autoregressive = self.autoregressive.cpu()

                self.diffusion = self.diffusion.to(self.device)
                self.vocoder = self.vocoder.to(self.device)

            del text_tokens
            del auto_conditioning

            wav_candidates = []
            for b in range(best_results.shape[0]):
                codes = best_results[b].unsqueeze(0)
                latents = best_latents[b].unsqueeze(0)

                # Find the first occurrence of the "calm" token and trim the codes to that.
                ctokens = 0
                for k in range(codes.shape[-1]):
                    if codes[0, k] == calm_token:
                        ctokens += 1
                    else:
                        ctokens = 0
                    if ctokens > breathing_room:  # 8 tokens gives the diffusion model some "breathing room" to terminate speech.
                        latents = latents[:, :k]
                        break

                mel = do_spectrogram_diffusion(self.diffusion, diffuser, latents, diffusion_conditioning,
                                               temperature=diffusion_temperature, verbose=verbose, progress=progress,
                                               desc="Transforming autoregressive outputs into audio..",
                                               sampler=diffusion_sampler,
                                               input_sample_rate=self.input_sample_rate,
                                               output_sample_rate=self.output_sample_rate)

                wav = self.vocoder.inference(mel)
                wav_candidates.append(wav)

            if not self.preloaded_tensors:
                self.diffusion = self.diffusion.cpu()
                self.vocoder = self.vocoder.cpu()

            def potentially_redact(clip, text):
                if self.enable_redaction:
                    return self.aligner.redact(clip.squeeze(1).to('cpu' if get_device_name() == "dml" else self.device),
                                               text, self.output_sample_rate).unsqueeze(1)
                return clip

            wav_candidates = [potentially_redact(wav_candidate, text) for wav_candidate in wav_candidates]

            if len(wav_candidates) > 1:
                res = wav_candidates
            else:
                res = wav_candidates[0]

            gc.collect()

            if return_deterministic_state:
                return res, (deterministic_seed, text, voice_samples, conditioning_latents)
            else:
                return res

    def deterministic_state(self, seed=None):
        """
        Sets the random seeds that tortoise uses to the current time() and returns that seed so results can be
        reproduced.
        """
        seed = int(time()) if seed is None else seed
        torch.manual_seed(seed)
        random.seed(seed)
        # Can't currently set this because of CUBLAS. TODO: potentially enable it if necessary.
        # torch.use_deterministic_algorithms(True)

        return seed
