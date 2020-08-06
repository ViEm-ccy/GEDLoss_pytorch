# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of spectral operations."""
import librosa
import numpy as np
import scipy.signal.windows as W
import scipy
# import tensorflow.compat.v2 as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-8  # Small constant to avoid division by zero.

# Mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

### This is a 'translation' from Google's tf implementation to the torch edition

# TODO: test code
#  I have never test this pytorch edition code,
#  maybe there are some parameter type errors or some backward gradient errors in it.
#  You'd better run a test before you use it.

def torch_aligned_random_crop(waves, frame_length):
    """Get aligned random crops from batches of input waves."""
    n, t = waves[0].shape
    crop_t = frame_length * (t//frame_length - 1)
    # offsets = [tf.random.uniform(shape=(), minval=0,
    #                              maxval=t-crop_t, dtype=tf.int32)
    #            for _ in range(n)]
    offsets = [np.random.randint(size=(),low=0,high=t-crop_t,dtype=torch.int32)
               for _ in range(n)]

    # waves_unbatched = [tf.split(w, n, axis=0) for w in waves]
    waves_unbatched = [torch.split(w, n, dim=0) for w in waves]

    # wave_crops = [[tf.slice(w, begin=[0, o], size=[1, crop_t])
    #                for w, o in zip(ws, offsets)] for ws in waves_unbatched]
    wave_crops = [[torch.narrow(torch.narrow(w,0,0,0+1),1,start=o,length=o+crop_t)
                   for w, o in zip(ws, offsets)] for ws in waves_unbatched]

    #wave_crops = [tf.concat(wc, axis=0) for wc in wave_crops]
    wave_crops = [torch.cat(wc, dim=0) for wc in wave_crops]

    return wave_crops


def torch_mel_to_hertz(frequencies_mel):
    """Converts frequencies in `frequencies_mel` from mel to Hertz scale."""
    # return _MEL_BREAK_FREQUENCY_HERTZ * (
    #         tf.math.exp(frequencies_mel / _MEL_HIGH_FREQUENCY_Q) - 1.)
    return _MEL_BREAK_FREQUENCY_HERTZ * (
            np.exp(frequencies_mel / _MEL_HIGH_FREQUENCY_Q) - 1.)


def torch_hertz_to_mel(frequencies_hertz):
    """Converts frequencies in `frequencies_hertz` in Hertz to the mel scale."""
    # return _MEL_HIGH_FREQUENCY_Q * tf.math.log(
    #     1. + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1. + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def torch_get_spectral_matrix(n, num_spec_bins=256, use_mel_scale=True,
                        sample_rate=24000):
    """DFT matrix in overcomplete basis returned as a TF tensor.

    Args:
      n: Int. Frame length for the spectral matrix.
      num_spec_bins: Int. Number of bins to use in the spectrogram
      use_mel_scale: Bool. Equally spaced on Mel-scale or Hertz-scale?
      sample_rate: Int. Sample rate of the waveform audio.

    Returns:
      Constructed spectral matrix.
    """
    sample_rate = float(sample_rate)
    upper_edge_hertz = sample_rate / 2.
    lower_edge_hertz = sample_rate / n

    if use_mel_scale:
        # upper_edge_mel = hertz_to_mel(upper_edge_hertz)
        # lower_edge_mel = hertz_to_mel(lower_edge_hertz)
        # mel_frequencies = tf.linspace(lower_edge_mel, upper_edge_mel, num_spec_bins)
        # hertz_frequencies = mel_to_hertz(mel_frequencies)

        upper_edge_mel = torch_hertz_to_mel(upper_edge_hertz)
        lower_edge_mel = torch_hertz_to_mel(lower_edge_hertz)
        mel_frequencies = torch.linspace(lower_edge_mel, upper_edge_mel, (upper_edge_mel-lower_edge_mel)/num_spec_bins)
        hertz_frequencies = torch_mel_to_hertz(mel_frequencies)
    else:
        # hertz_frequencies = tf.linspace(lower_edge_hertz, upper_edge_hertz,
        #                                 num_spec_bins)
        hertz_frequencies = torch.linspace(lower_edge_hertz, upper_edge_hertz,
                                        (upper_edge_hertz-lower_edge_hertz)/num_spec_bins)
    # time_col_vec = (tf.reshape(tf.range(n, dtype=tf.float32), [n, 1])
    #                 * np.cast[np.float32](2. * np.pi / sample_rate))
    time_col_vec = (torch.reshape(torch.range(0,n, dtype=torch.float32), [n, 1])
                    * np.cast[np.float32](2. * np.pi / sample_rate))
    tmat = torch.reshape(hertz_frequencies, [1, num_spec_bins]) * time_col_vec
    dct_mat = torch.cos(tmat)
    dst_mat = torch.sin(tmat)
    # dft_mat = tf.complex(real=dct_mat, imag=-dst_mat)
    dft_mat = torch.view_as_complex([dct_mat,-dst_mat])
    # TODO: update my pytoch to support the complex tensor
    # torch.view_as_complex() opreation need the last release edition of Pytorch 1.6.0

    return dft_mat


def torch_matmul_real_with_complex(real_input, complex_matrix):
    real_part = torch.matmul(real_input, torch.view_as_real(complex_matrix)[:,0])
    imag_part = torch.matmul(real_input, torch.view_as_real(complex_matrix)[:,1])
    # return tf.complex(real_part, imag_part)
    return torch.view_as_complex([real_part, imag_part])

def torch_build_mel_basis(
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz,
        dtype=torch.float32
):
    assert upper_edge_hertz <= sample_rate // 2
    return torch.tensor(librosa.filters.mel(sample_rate, num_spectrogram_bins, n_mels=num_mel_bins,
                                            fmin=lower_edge_hertz, fmax=upper_edge_hertz),dtype=dtype)


def torch_calc_spectrograms(waves, window_lengths, spectral_diffs=(0, 1),
                      window_name='hann', use_mel_scale=True,
                      proj_method='matmul', num_spec_bins=256,
                      random_crop=True):
    """Calculate spectrograms with multiple window sizes for list of input waves.

    Args:
      waves: List of float tensors of shape [batch, length] or [batch, length, 1].
      window_lengths: List of Int. Window sizes (frame lengths) to use for
        computing the spectrograms.
      spectral_diffs: Int. order of finite diff. to take before computing specs.
      window_name: Str. Name of the window to use when computing the spectrograms.
        Supports 'hann' and None.
      use_mel_scale: Bool. Whether or not to project to mel-scale frequencies.
      proj_method: Str. Spectral projection method implementation to use.
        Supported are 'fft' and 'matmul'.
      num_spec_bins: Int. Number of bins in the spectrogram.
      random_crop: Bool. Take random crop or not.

    Returns:
      Tuple of lists of magnitude spectrograms, with output[i][j] being the
        spectrogram for input wave i, computed for window length j.
    """
    # waves = [tf.squeeze(w, axis=-1) for w in waves]
    waves = [torch.squeeze(w, dim=-1) for w in waves]

    if window_name == 'hann':
        # windows = [tf.reshape(tf.signal.hann_window(wl, periodic=False), [1, 1, -1])
        #            for wl in window_lengths]
        windows = [torch.reshape(torch.from_numpy(W.hann(wl)), [1, 1, -1])
                   for wl in window_lengths]
    elif window_name is None:
        windows = [None] * len(window_lengths)
    else:
        raise ValueError('Unknown window function (%s).' % window_name)

    spec_len_wave = []
    for d in spectral_diffs:
        for length, window in zip(window_lengths, windows):

            wave_crops = waves
            for _ in range(d):
                wave_crops = [w[:, 1:] - w[:, :-1] for w in wave_crops]

            if random_crop:
                # wave_crops = aligned_random_crop(wave_crops, length)
                wave_crops = torch_aligned_random_crop(wave_crops, length)

            # frames = [tf.signal.frame(wc, length, length // 2) for wc in wave_crops]
            frames = [torch.tensor(librosa.util.frame(wc.numpy(),length,length//2)) for wc in wave_crops]
            # TODO: Whether this method is feasible (in the gradient part) remains to be verified
            if window is not None:
                frames = [f * window for f in frames]

            if proj_method == 'fft':
                # ffts = [tf.signal.rfft(f)[:, :, 1:] for f in frames]
                ffts = [torch.rfft(f,signal_ndim=1)[:, :, 1:] for f in frames]
            elif proj_method == 'matmul':
                # mat = get_spectral_matrix(length, num_spec_bins=num_spec_bins,
                #                           use_mel_scale=use_mel_scale)
                # ffts = [matmul_real_with_complex(f, mat) for f in frames]
                mat = torch_get_spectral_matrix(length, num_spec_bins=num_spec_bins,
                                          use_mel_scale=use_mel_scale)
                ffts = [torch_matmul_real_with_complex(f, mat) for f in frames]

            #sq_mag = lambda x: tf.square(tf.math.real(x)) + tf.square(tf.math.imag(x))
            sq_mag = lambda x: (torch.view_as_real(x)[:,0]) + (torch.view_as_real(x)[:,1]**2)**2
            # torch.view_as_real() opreation need the last release edition of Pytorch 1.6.0
            specs_sq = [sq_mag(f) for f in ffts]

            if use_mel_scale and proj_method == 'fft':
                sample_rate = 24000
                upper_edge_hertz = sample_rate / 2.
                lower_edge_hertz = sample_rate / length
                # lin_to_mel = tf.signal.linear_to_mel_weight_matrix(
                #     num_mel_bins=num_spec_bins,
                #     num_spectrogram_bins=length // 2 + 1,
                #     sample_rate=sample_rate,
                #     lower_edge_hertz=lower_edge_hertz,
                #     upper_edge_hertz=upper_edge_hertz,
                #     dtype=tf.dtypes.float32)[1:]
                # specs_sq = [tf.matmul(s, lin_to_mel) for s in specs_sq]
                lin_to_mel = torch_build_mel_basis(
                    num_mel_bins=num_spec_bins,
                    num_spectrogram_bins=length,
                    sample_rate=sample_rate,
                    lower_edge_hertz=lower_edge_hertz,
                    upper_edge_hertz=upper_edge_hertz,
                    dtype=torch.float32)
                # TODO: I use librosa to build the mel filters here to instead, and i'm not sure whether this method works or not
                specs_sq = [torch.matmul(s, lin_to_mel) for s in specs_sq]

            # specs = [tf.sqrt(s+EPSILON) for s in specs_sq]
            specs = [torch.sqrt(s+EPSILON) for s in specs_sq]

            spec_len_wave.append(specs)

    spec_wave_len = zip(*spec_len_wave)
    return spec_wave_len


def torch_sum_spectral_dist(specs1, specs2, add_log_l2=True):
    """Sum over distances in frequency space for different window sizes.

    Args:
      specs1: List of float tensors of shape [batch, frames, frequencies].
        Spectrograms of the first wave to compute the distance for.
      specs2: List of float tensors of shape [batch, frames, frequencies].
        Spectrograms of the second wave to compute the distance for.
      add_log_l2: Bool. Whether or not to add L2 in log space to L1 distances.

    Returns:
      Tensor of shape [batch] with sum of L1 distances over input spectrograms.
    """

    # l1_distances = [tf.reduce_mean(abs(s1 - s2), axis=[1, 2])
    #                 for s1, s2 in zip(specs1, specs2)]
    # sum_dist = tf.math.accumulate_n(l1_distances)
    l1_distances = [torch.mean(abs(s1 - s2), dim=[1, 2])
                    for s1, s2 in zip(specs1, specs2)]
    sum_dist = np.sum(l1_distances,dim=0)


    if add_log_l2:
        # log_deltas = [tf.math.squared_difference(
        #     tf.math.log(s1 + EPSILON), tf.math.log(s2 + EPSILON))  # pylint: disable=bad-continuation
        #     for s1, s2 in zip(specs1, specs2)]
        # log_l2_norms = [tf.reduce_mean(
        #     tf.sqrt(tf.reduce_mean(ld, axis=-1) + EPSILON), axis=-1)
        #     for ld in log_deltas]
        # sum_log_l2 = tf.math.accumulate_n(log_l2_norms)

        log_deltas = [(
            torch.log(s1 + EPSILON)-torch.log(s2 + EPSILON))**2
            for s1, s2 in zip(specs1, specs2)]
        log_l2_norms = [torch.mean(torch.sqrt(torch.mean(ld, dim=-1) + EPSILON), dim=-1)
            for ld in log_deltas]
        sum_log_l2 = np.sum(log_l2_norms,dim=0)

        sum_dist += sum_log_l2

    return sum_dist


def torch_ged(wav_fake1, wav_fake2, wav_real):
    """Multi-scale spectrogram-based generalized energy distance.

    Args:
      wav_fake1: Float tensors of shape [batch, time, 1].
        Generated audio samples conditional on a set of linguistic features.
      wav_fake2: Float tensors of shape [batch, time, 1].
        Second set of samples conditional on same features, but using new noise.
      wav_real: Float tensors of shape [batch, time, 1].
        Real (data) audio samples corresponding to the same features.

    Returns:
      Tensor of shape [batch] with the GED values.
    """

    specs_fake1, specs_fake2, specs_real = torch_calc_spectrograms(
        waves=[wav_fake1, wav_fake2, wav_real],
        window_lengths=[2**i for i in range(6, 12)])

    dist_real_fake1 = torch_sum_spectral_dist(specs_real, specs_fake1)
    dist_real_fake2 = torch_sum_spectral_dist(specs_real, specs_fake2)
    dist_fake_fake = torch_sum_spectral_dist(specs_fake1, specs_fake2)

    return dist_real_fake1 + dist_real_fake2 - dist_fake_fake

# TODO: Run it!
if __name__=='__main__':
    sample_rate= 22050
    wav_fake1 = librosa.core.load('wav_fake1_path', sr=sample_rate)[0]
    wav_fake2 = librosa.core.load('wav_fake2_path', sr=sample_rate)[0]
    wav_real = librosa.core.load('wav_real_path', sr=sample_rate)[0]

    GEDLoss = torch_ged(wav_fake1,wav_fake2,wav_real)

    print(GEDLoss)