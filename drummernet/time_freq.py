"""
All the torch toys for time-frequency
"""
from globals import *
import librosa
import torch
import torch.nn as nn
import numpy as np
import librosa


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


EPS = 1e-8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TCDTYPE = torch.float32

def cqt_filter_fft_manual(sr, fmin, n_bins, bins_per_octave, 
                          tuning=0.0, filter_scale=1, norm=1, sparsity=0.01,
                          hop_length=None, window='hann', pad_mode='reflect'):
    """
    librosa.filters.constant_q가 deprecated됨에 따라,
    직접 constant-Q 필터 뱅크(FFT basis)를 계산하는 함수.
    
    Parameters:
      sr           : 샘플링 레이트
      fmin         : 가장 낮은 주파수 (Hz)
      n_bins       : 총 필터 수
      bins_per_octave: 옥타브당 필터 수
      tuning       : 튜닝(반음 단위)
      filter_scale : 필터 스케일 (보통 1)
      norm         : 정규화 방식 (여기서는 사용하지 않음)
      sparsity     : 희소화 임계값 (절대값이 이 값보다 작으면 0으로 만듦)
      hop_length   : 사용되지 않음 (향후 확장을 위해)
      window       : 윈도우 함수 종류 (현재 'hann'만 지원)
      pad_mode     : 패딩 모드 (여기서는 사용하지 않음)
      
    Returns:
      fft_basis : numpy.ndarray, shape = (n_fft, n_bins), 복소수 타입
      n_fft     : 사용된 FFT 길이
      lengths   : 각 필터의 유효 길이 (정수 배열)
    """
    # tuning 적용: fmin 조정
    fmin_adj = fmin * (2.0 ** (tuning / 12.0))
    # 적절한 n_fft 계산: filter_scale * sr / fmin_adj의 로그2 올림
    n_fft = int(2 ** np.ceil(np.log2(filter_scale * sr / fmin_adj)))
    
    # 중심 주파수 계산
    freqs = librosa.cqt_frequencies(n_bins, fmin=fmin_adj, bins_per_octave=bins_per_octave)
    
    # Q 값 (필터 폭 관련)
    Q = filter_scale / (2 ** (1 / bins_per_octave) - 1)
    
    # 윈도우 함수 생성: n_fft 길이의 hann 윈도우 (여기서는 단순 사용)
    if window == 'hann':
        full_win = np.hanning(n_fft)
    else:
        raise NotImplementedError("현재는 'hann' 윈도우만 지원합니다.")
    
    fft_basis = np.zeros((n_fft, n_bins), dtype=np.complex64)
    lengths = np.zeros(n_bins, dtype=int)
    
    for i, f in enumerate(freqs):
        # 각 필터의 길이 L (이론상 Q * sr / f)
        L = int(np.ceil(Q * sr / f))
        # L이 n_fft보다 크면 자름
        if L > n_fft:
            L = n_fft
        lengths[i] = L
        
        # 시간 축 (중심을 0으로 잡음)
        t = np.arange(L) - (L - 1) / 2.0
        # 필터 커널: 윈도우 함수 적용 후 복소수 지수함수
        kernel = full_win[:L] * np.exp(2j * np.pi * f * t / sr)
        # 0 패딩하여 n_fft 길이로 만듦
        kernel_padded = np.zeros(n_fft, dtype=np.complex64)
        start = (n_fft - L) // 2
        kernel_padded[start:start+L] = kernel
        # 필터의 FFT 계산
        fft_basis[:, i] = np.fft.fft(kernel_padded)
    
    # 희소화 처리: 절대값이 sparsity 미만인 요소를 0으로
    fft_basis[np.abs(fft_basis) < sparsity] = 0
    return fft_basis, n_fft, lengths


class PseudoCqt:
    """
    Pseudo-CQT를 계산하는 PyTorch 클래스.
    최신 librosa의 함수를 사용하지 않고, 직접 필터 뱅크를 구성합니다.
    
    Usage:
        src, _ = librosa.load(filename, sr=22050)
        src_tensor = torch.tensor(src, dtype=TCDTYPE, device=DEVICE)
        cqt_calc = PseudoCqt(sr=22050, hop_length=512, fmin=2*32.703195,
                             n_bins=84, bins_per_octave=12)
        cqt_spec = cqt_calc(src_tensor)
    """
    def __init__(self, sr=22050, hop_length=512, fmin=None, n_bins=84,
                 bins_per_octave=12, tuning=0.0, filter_scale=1,
                 norm=1, sparsity=0.01, window='hann', scale=True,
                 pad_mode='reflect', device=DEVICE, dtype=TCDTYPE):
        if scale is not True:
            raise NotImplementedError('scale=False는 구현되어 있지 않습니다.')
        if window != 'hann':
            raise NotImplementedError("hann 윈도우만 구현되어 있습니다.")
        
        if fmin is None:
            fmin = 2 * 32.703195  # 기본: C2 (C1는 너무 낮음)
        
        # 필터 뱅크 계산 (직접 구현한 함수 사용)
        fft_basis_np, n_fft, lengths = cqt_filter_fft_manual(
            sr, fmin, n_bins, bins_per_octave,
            tuning=tuning, filter_scale=filter_scale, norm=norm,
            sparsity=sparsity, hop_length=hop_length, window=window,
            pad_mode=pad_mode
        )
        # STFT는 n_fft//2+1 주파수 bin만 사용하므로, 그 부분만 취함.
        fft_basis_np = np.abs(fft_basis_np[:n_fft // 2 + 1, :]).T  # shape: (n_bins, n_fft//2+1)
        
        self.fft_basis = torch.tensor(fft_basis_np, dtype=dtype, device=device)
        self.fmin = fmin
        self.fmax = fmin * 2 ** (float(n_bins) / bins_per_octave)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad_mode = pad_mode
        self.scale = scale
        
        # STFT에 사용할 윈도우: n_fft 길이의 hann 윈도우를 사용
        win = torch.hann_window(n_fft, periodic=True, device=device, dtype=dtype)
        self.window = win
        
        print(f'PseudoCqt init with fmin: {fmin}, n_bins: {n_bins}, bins_per_octave: {bins_per_octave}, '
              f'win_len: {len(self.window)}, n_fft: {n_fft}, hop_length: {hop_length}')

    def forward(self, y):
        """
        y: 1D 또는 2D Tensor (배치가 포함된 오디오 신호)
        반환: 로그 스케일의 pseudo-CQT 스펙트럼
        """
        # differentiable STFT 계산 (PyTorch 1.8+의 복소수 지원)
        stft = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            pad_mode=self.pad_mode,
            return_complex=True
        )
        # magnitude 계산
        mag_stfts = torch.abs(stft)  # shape: (..., n_fft//2+1, time)
        
        # pseudo-CQT: 필터 뱅크와 STFT magnitude의 선형 결합
        if y.ndim == 1:
            # (n_fft//2+1, time)와 (n_bins, n_fft//2+1)의 matmul
            cqt_spec = torch.matmul(self.fft_basis, mag_stfts)
        else:
            # 배치 처리: fft_basis shape: (n_bins, n_fft//2+1)
            # stft: (batch, n_fft//2+1, time)
            # expand fft_basis: (1, n_bins, n_fft//2+1)
            fft_basis_exp = self.fft_basis.unsqueeze(0)
            cqt_spec = torch.matmul(fft_basis_exp, mag_stfts)  # (batch, n_bins, time)
        
        # 정규화: scale=True 인 경우, n_fft의 제곱근으로 나눔.
        norm_factor = torch.sqrt(torch.tensor(self.n_fft, dtype=cqt_spec.dtype, device=cqt_spec.device))
        cqt_spec = cqt_spec / norm_factor
        
        # 로그 스케일 변환
        return torch.log(cqt_spec + EPS)

    def __call__(self, y):
        return self.forward(y)

# class PseudoCqt:
#     """A class to compute pseudo-CQT with Pytorch.
#     API (+implementations) follows librosa
#     (
#     https://librosa.github.io/librosa/generated/librosa.core.pseudo_cqt.html
#     )

#     Usage:
#         src, _ = librosa.load(filename)
#         src_tensor = torch.tensor(src)
#         cqt_calculator = PseudoCqt()
#         cqt_calculator(src_tensor)

#     """

#     def __init__(self, sr=22050, hop_length=512, fmin=None, n_bins=84,
#                  bins_per_octave=12, tuning=0.0, filter_scale=1,
#                  norm=1, sparsity=0.01, window='hann', scale=True,
#                  pad_mode='reflect'):

#         if scale is not True:
#             raise NotImplementedError('scale=False is not implemented')
#         if window != 'hann':
#             raise NotImplementedError('Window function other than hann is not implemented')

#         if fmin is None:
#             fmin = 2 * 32.703195  # note_to_hz('C2') because C1 is too low

#         if tuning is None:
#             tuning = 0.0  # let's make it simple

#         fft_basis, n_fft, _ = cqt_filter_fft(sr, fmin, n_bins, bins_per_octave,
#                                              tuning, filter_scale, norm, sparsity,
#                                              hop_length=hop_length, window=window)

#         self.fft_basis = torch.tensor(np.array(np.abs(fft_basis.todense())), dtype=TCDTYPE,
#                                       device=DEVICE)  # because it was sparse. (n_bins, n_fft)

#         self.fmin = fmin
#         self.fmax = fmin * 2 ** (float(n_bins) / bins_per_octave)
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#         self.pad_mode = pad_mode
#         self.scale = scale
#         win = torch.zeros((self.n_fft,), device=DEVICE)
#         win[self.n_fft // 2 - self.n_fft // 8:self.n_fft // 2 + self.n_fft // 8] = torch.hann_window(self.n_fft // 4)
#         self.window = win
#         msg = 'PseudoCQT init with fmin:{}, {}, bins, {} bins/oct, win_len: {}, n_fft:{}, hop_length:{}'
#         print(msg.format(int(fmin), n_bins, bins_per_octave, len(self.window), n_fft, hop_length))

#     def __call__(self, y):
#         return self.forward(y)

#     def forward(self, y):
#         # thor:  lowercase variable names
#         mag_stfts = torch.stft(y, self.n_fft,
#                                hop_length=self.hop_length,
#                                window=self.window).pow(2).sum(-1)  # (batch, n_freq, time)
#         mag_stfts = torch.sqrt(mag_stfts + EPS)  # without EPS, backpropagating through CQT can yield NaN.
#         # Project onto the pseudo-cqt basis
#         # C_torch = torch.stack([torch.sparse.mm(self.fft_basis, D_torch_row) for D_torch_row in D_torch])
#         mag_melgrams = torch.matmul(self.fft_basis, mag_stfts)

#         mag_melgrams /= torch.tensor(np.sqrt(self.n_fft), device=y.device)  # because `scale` is always True
#         return to_log(mag_melgrams)


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
