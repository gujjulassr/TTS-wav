import numpy as np
from math import pi as PI
from scipy import signal
from itertools import groupby
# import librosa
import scipy

_mel_basis = None
min_level_db = -100
#forward_window_fn = functools.partial(tf.signal.hamming_window, periodic=False)
#forward_window_fn=tf.signal.hann_window

def save_wav(wav, path, sr=16000):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  scipy.io.wavfile.write(path, sr, wav.astype(np.int16))   #scipy.io.wavfile.

# def load_wav(path, sr=16000):
#   return librosa.core.load(path, sr=sr)[0]

# def _stft(y, frame_length=400, frame_step=80, fft_length=1024):
#   return librosa.stft(y=y, n_fft=fft_length, hop_length=frame_step, win_length=frame_length)

# def spectrogram(y, frame_length=400, frame_step=80, fft_length=1024):
#   pe_y = preemphasis(y)
#   D = _stft(pe_y, frame_length, frame_step, fft_length)
#   return D.T
  
def _normalize(S):
  return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def _denormalize(S):
  return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def minmax_norm(S, min_val, max_val):
    return np.clip((S - min_val) /(max_val - min_val), 0., 1.)

def inv_minmax_norm(x, min_val, max_val):
    return np.clip(x,0,1) * (max_val - min_val)+min_val  

def norm_spec(mag_spec):
  S = _amp_to_db(mag_spec) - 20
  return _normalize(S)
  
def inv_norm_spec(norm_spec):
  S = _denormalize(norm_spec) + 20
  S = _db_to_amp(S)
  return S 
  
def inv_norm_fbank(norm_fbank):
  FB = _denormalize(norm_fbank) + 20
  FB = _db_to_amp(FB)
  return FB   

def melspectrogram(mag_spec, num_mel_bins):
  S = _amp_to_db(_linear_to_mel(mag_spec, num_mel_bins)) - 20
  return _normalize(S)

def _linear_to_mel(spectrogram, num_mel_bins):
  global _mel_basis
  if _mel_basis is None:
    _mel_basis = _build_mel_basis(num_mel_bins)
  return np.dot(spectrogram, _mel_basis.T)

def _build_mel_basis(num_mel_bins):
  n_fft = 1024
  return librosa.filters.mel(sr=22050, n_fft=n_fft, n_mels=num_mel_bins)

def log(x):
    epsilon=1e-8
    return np.log(np.maximum(x, epsilon))

def log10(x):
    return log(x)/log(10.)

def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def dB2mag(x):
    return 10**(x/20.)

def pow2dB(x):
    return 10.* log10(x)

def dB2pow(x):
    return 10**(x/10)

def preemphasis(x):
  return scipy.signal.lfilter([1, -0.97], [1], x)   

def inv_preemphasis(x, coeff=0.97):
  return signal.lfilter([1], [1, -coeff], x)

def kaldi_pad(wav, frame_shift):
    old_length=len(wav)
    num_frames = (old_length + frame_shift // 2) // frame_shift
    new_length = num_frames*frame_shift
    diff = new_length-old_length

    if diff > 0:
        wav = np.concatenate([wav, np.zeros(diff)])
    else:
        wav = wav[:new_length]

    return wav

def arange_list(a):
    fwd = np.array([], dtype=np.uint8)
    rev = np.array([], dtype=np.uint8)
    for x in a:
        idx = np.arange(1,x+1)
        fwd = np.append(fwd, idx)
        rev = np.append(rev, np.flip(idx))
    return fwd, rev

def seq2freq(seq):
    data = [[i, sum(1 for i in group)] for i, group in groupby(seq)]
    data=np.asarray(data)
    lab = data[:,0]
    freq = data[:,1]
    return lab, freq

# def _griffin_lim(S):
#   '''librosa implementation of Griffin-Lim
#   Based on https://github.com/librosa/librosa/issues/434
#   '''
#   angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
#   S_complex = np.abs(S).astype(np.complex)
#   y = _istft(S_complex * angles)
#   for i in range(50):
#     angles = np.exp(1j * np.angle(_stft(y)))
#     y = _istft(S_complex * angles)
#   return y

# def _istft(y):
#   hop_length, win_length = 80, 400
#   return librosa.istft(y, hop_length=hop_length, win_length=win_length)

#def freq2feat(freq):
#    repeats=np.repeat(np.arange(len(freq)), freq)


