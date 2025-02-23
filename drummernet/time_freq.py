"""
All the torch toys for time-frequency
"""
from globals import *
import librosa
import torch
import torch.nn as nn
import numpy as np
import math


def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, normalized=False, onesided=True, length=None):
    """stft_matrix = (batch, freq, time, complex)

    All based on librosa
        - http://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#istft
    What's missing?
        - normalize by sum of squared window --> do we need it here?
        Actually the result is ok by simply dividing y by 2.
    """
    assert normalized == False
    assert onesided == True
    assert window == 'hann'
    assert center == True

    device = stft_matrix.device
    n_fft = 2 * (stft_matrix.shape[-3] - 1)

    batch = stft_matrix.shape[0]

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    istft_window = torch.hann_window(n_fft).to(device).view(1, -1)  # (batch, freq)

    n_frames = stft_matrix.shape[-2]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    y = torch.zeros(batch, expected_signal_len, device=device)
    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, :, i]
        iffted = torch.irfft(spec, signal_ndim=1, signal_sizes=(win_length,))

        ytmp = istft_window * iffted
        y[:, sample:(sample + n_fft)] += ytmp

    y = y[:, n_fft // 2:]

    if length is not None:
        if y.shape[1] > length:
            y = y[:, :length]
        elif y.shape[1] < length:
            y = torch.cat(y[:, :length], torch.zeros(y.shape[0], length - y.shape[1], device=y.device))

    coeff = n_fft / float(
        hop_length) / 2.0  # -> this might go wrong if curretnly asserted values (especially, `normalized`) changes.
    return y / coeff


class Melgramer():
    """A class for melspectrogram computation

    Args:
        n_fft (int): number of fft points for the STFT, based on which melspectrogram is computed

        hop_length (int): STFT hop length [samples]

        sr (int): sampling rate

        n_mels (int): number of mel bins

        fmin (float): minimum frequency of mel bins

        fmax (float): maximum frequency of mel bins

        power_melgram (float): the power of the STFT

        window (torch.Tensor): window function for `torch.STFT`

        log (bool): whether the result will be as log(melgram) or not

        dtype: the torch datatype for this melspectrogram
    """

    def __init__(self, n_fft=1024, hop_length=None, sr=22050, n_mels=128, fmin=0.0, fmax=None,
                 power_melgram=1.0, window=None, log=False, dtype=None):
        # thor:  is this correct python idiom for assert / what is the diff between this and raise?
        assert sr > 0
        assert fmin >= 0.0
        if fmax is None:
            fmax = float(sr) / 2
        assert fmax > fmin
        assert isinstance(log, bool)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = int(sr)
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.log = log
        self.power_melgram = power_melgram
        self.window = window
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.dtype = dtype

        self.mel_fb = nn.Parameter(self._get_mel_fb())

    def _get_mel_fb(self):
        """returns (n_mels, n_fft//2+1)"""
        return torch.from_numpy(librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
                                                    fmin=self.fmin, fmax=self.fmax)).type(self.dtype)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, waveforms):
        """x is perhaps (batch, freq, time).
        returns (batch, n_mel, time)"""
        mag_stfts = torch.stft(waveforms, self.n_fft,
                               hop_length=self.hop_length,
                               window=self.window).pow(2).sum(-1)  # (batch, n_freq, time)
        mag_stfts = torch.sqrt(mag_stfts + EPS)  # without EPS, backpropagating can yield NaN.
        # Project onto the pseudo-cqt basis
        mag_melgrams = torch.matmul(self.mel_fb, mag_stfts)
        if self.log:
            mag_melgrams = to_log(mag_melgrams)
        return mag_melgrams


class PseudoCqt:
    def __init__(self, sr=22050, hop_length=512, fmin=None, n_bins=84,
                 bins_per_octave=12, tuning=0.0, filter_scale=1,
                 norm=1, sparsity=0.01, window='hann', scale=True,
                 pad_mode='reflect'):
        if scale is not True:
            raise NotImplementedError('scale=False는 구현되어 있지 않습니다.')
        if window != 'hann':
            raise NotImplementedError('hann 윈도우 외에는 구현되어 있지 않습니다.')

        if fmin is None:
            fmin = 2 * 32.703195  # C2의 주파수 (C1은 너무 낮음)

        if tuning is None:
            tuning = 0.0

        # librosa.filters.constant_q를 사용하여 CQT 필터뱅크 생성
        cqt_filters, lengths = librosa.filters.constant_q(
            sr=sr,
            n_bins=n_bins,
            fmin=fmin,
            bins_per_octave=bins_per_octave,
            norm=norm,
            pad_fft=False
        )
        # cqt_filters는 shape (n_bins, n_fft)이며, 복소수 값입니다.
        # fft_basis는 필터의 절댓값(또는 실수부)을 사용합니다.
        n_fft = (cqt_filters.shape[1] - 1) * 2
        self.fft_basis = torch.tensor(np.abs(cqt_filters), dtype=TCDTYPE, device=DEVICE)

        self.fmin = fmin
        self.fmax = fmin * 2 ** (float(n_bins) / bins_per_octave)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.scale = scale

        win = torch.zeros((self.n_fft,), device=DEVICE)
        start = self.n_fft // 2 - self.n_fft // 8
        end = start + (self.n_fft // 4)
        win[start:end] = torch.hann_window(self.n_fft // 4, device=DEVICE)
        self.window = win

        msg = ('PseudoCQT init with fmin: {}, {} bins, {} bins/oct, '
               'win_len: {}, n_fft: {}, hop_length: {}')
        print(msg.format(int(fmin), n_bins, bins_per_octave, len(self.window), n_fft, hop_length))

    def __call__(self, y):
        return self.forward(y)

    def forward(self, y):
        # y: 입력 신호 (1D 또는 배치 차원의 1D 텐서)

        # torch.stft: return_complex=True 옵션을 사용하여 복소수 결과를 얻습니다.
        stft_complex = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )
        # STFT magnitude 계산: 복소수 텐서의 절댓값을 구합니다.
        stft_mag = stft_complex.abs()
        # 원래는 EPS를 더해 안정성을 확보했습니다.
        stft_mag = torch.sqrt(stft_mag ** 2 + EPS)

        # pseudo-CQT를 계산: 필터뱅크(fft_basis)와 STFT magnitude의 행렬곱
        mag_melgrams = torch.matmul(self.fft_basis, stft_mag)
        # scale 처리를 위해 n_fft의 제곱근으로 나눕니다.
        mag_melgrams /= torch.tensor(np.sqrt(self.n_fft), device=y.device, dtype=mag_melgrams.dtype)

        return to_log(mag_melgrams)


def to_log(mag_specgrams):
    """

    Args:
        mag_specgrams (torch.Tensor), non-power spectrum, and non-negative.

    """
    return (torch.log10(mag_specgrams + EPS) - torch.log10(torch.tensor(EPS, device=mag_specgrams.device)))


def to_decibel(mag_specgrams):
    """
    Args:
        mag_specgrams (torch.Tensor), non-power spectrum, and non-negative.

    """
    return 20 * to_log(mag_specgrams)


def log_stft(waveforms, n_fft, hop, center=True, mode='normal'):
    """
    if mode == 'high', window is hann(n_fft//4) with zero-padded.

    Args:
        waveforms (torch.Tensor): audio signal to perform stft

        n_fft (int): number of fft points

        hop (int): hop length [samples]

        center (bool): if stft is center-windowed or not

        mode (str): 'normal' or 'high'

    """
    assert mode in ('normal', 'high')
    if mode == 'normal':
        win = torch.hann_window(n_fft).to(waveforms.device)
    else:
        win = torch.zeros((n_fft,), device=waveforms.device)
        win[n_fft // 2 - n_fft // 8:n_fft // 2 + n_fft // 8] = torch.hann_window(n_fft // 4)
        assert hop <= (n_fft // 4), 'hop:{}, n_fft:{}'.format(hop, n_fft)
    #

    complex_stfts = torch.stft(torch.tensor(waveforms), n_fft, hop, window=win, center=center)
    mag_stfts = complex_stfts.pow(2).sum(-1)  # (*, freq, time)

    return to_log(mag_stfts)
