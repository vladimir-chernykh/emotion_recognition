"""this is the module version"""

import librosa
import numpy as np
import math

# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #


def ratio_to_cents(r): 
  return 1200.0 * np.log2(r)


def ratio_to_cents_protected(f1, f2): 
  """avoid divide by zero in here, but expects np.arrays"""
  out = np.zeros_like(f1)
  key = (f1!=0.0) * (f2!=0.0)
  out[key] = 1200.0 * np.log2(f1[key]/f2[key])
  out[f1==0.0] = -np.inf
  out[f2==0.0] = np.inf
  out[(f1==0.0) * (f2==0.0)] = 0.0
  return out


def cents_to_ratio(c): 
  return np.power(2, c/1200.0)


def freq_to_midi(f): 
  return 69.0 + 12.0 * np.log2(f/440.0)


def midi_to_freq(m): 
  return np.power(2, (m-69.0)/12.0) * 440.0 if m!= 0.0 else 0.0


def bin_to_freq(b, sr, n_fft):
  return b * float(sr) / float(n_fft)


def freq_to_bin(f, sr, n_fft): 
  return np.round(f/(float(sr)/float(n_fft))).astype('int')


# --------------------------------------------------------------------------- #
# ppitch
# --------------------------------------------------------------------------- #


def ppitch(y, sr=44100, n_fft=8820, win_length=1024, hop_length=2048, 
  num_peaks=20, num_pitches=3, min_peak=2, max_peak=743, min_fund=55.0, 
  max_fund=1000.0, harm_offset=0, harm_rolloff=0.75, ml_width=25, 
  bounce_width=0, bounce_hist=0, bounce_ratios=None, max_harm=32767, 
  npartial=7):
 
  """Polyphonic pitch estimation.
  parameters:
    - y: signal to analyze
    - sr: sample rate
    - n_fft: fft size
    - win_length: fft window size
    - hop_length: stft hop
    - num_peaks: number of peaks to find
    - num_pitches: number of pitches ot find
    - min_peak: min peak frequency (Hz) <- NOT IMPLEMENTED
    - max_peak: max peak frequency (Hz)
    - min_fund: min fundamental freq
    - max_fund: max fundamental freq
    - harm_offset: deprecated 
    - harm_rolloff: deprecated (use if npartial is None)
    - ml_width: cents range to include peak in maximum likliehood
    - bounce_width: bounce freqs w/in this +/- this range from bounce_ratios
    - bounce_hist: bounce freqs from previous frames as well
    - bounce_ratios: bounce freqs w/in +/- bounce_width of these ratios
    - max_harm: basically deprecated, don't use harms above this one
    - npartial: partial that is half weight
  returns:
    - pitches
    - STFT
    - peaks
  todo:
    -> stats for how much bounced
  notes
    - changing to npartial instead of harm_rolloff, 
      but if npartial=None, use harm_rolloff
    - previously was nearest_multiple+1 # to avoid divide by 0
   """

  # other params FUCKING: (for now)

  # ------------------------------------------------------------------------- #
  # go to time-freq
  # ------------------------------------------------------------------------- #

  if_gram, D = librosa.core.ifgram(y, sr=sr, n_fft=n_fft, 
    win_length=win_length,hop_length=hop_length)

  # ------------------------------------------------------------------------- #
  # find peaks
  # ------------------------------------------------------------------------- #

  peak_thresh = 1e-3

  # bins (freq range) to search 
  min_bin = 2
  max_bin = freq_to_bin(float(max_peak), sr, n_fft)
  # //--> [make sure we have room either side for peak picking]

  # npartial
  if npartial is not None:
    harm_rolloff = math.log(2, float(npartial))

  # things we know
  num_bins, num_frames = if_gram.shape

  # store pitches here
  pitches = np.zeros([num_frames, num_pitches])       # pitch bins
  peaks = np.zeros([num_frames, num_peaks])           # top20 peaks
  fundamentals = np.zeros([num_frames, num_pitches])  # funds (least squares)
  confidences = np.zeros([num_frames, num_pitches])   # confidence scores

  # loop through frames
  for i in range(num_frames):

    # grab frequency, magnitudes, total power
    frqs = if_gram[:,i]
    mags = np.abs(D[:,i])
    total_power = mags.sum()
    max_amp = np.max(mags)

    # neighbor bins
    lower  = mags[(min_bin-1):(max_bin)]
    middle = mags[(min_bin)  :(max_bin+1)]
    upper  = mags[(min_bin+1):(max_bin+2)]
    
    # all local peaks
    peaks_mask_all = (middle > lower) & (middle > upper)

    # resize mask to dimensions of im_gram
    zeros_left = np.zeros(min_bin)
    zeros_right = np.zeros(num_bins - min_bin - max_bin + 1)
    peaks_mask_all = np.concatenate((zeros_left, peaks_mask_all, zeros_right)) 

    # find the first <num_peaks> (at most)
    peaks_mags_all = peaks_mask_all * mags
    top20 = np.argsort(peaks_mags_all)[::-1][0:num_peaks]

    # extract just the peaks (freqs and normed mags)
    peaks_frqs = frqs[top20]
    peaks_mags = mags[top20]

    # note number of peaks found
    num_peaks_found = top20.shape[0]

    # ----------------------------------------------------------------------- #
    # maximum liklihood 
    # ----------------------------------------------------------------------- #

    # range we'll consider for fundamentals
    min_freq = float(min_fund)
    max_freq = float(max_fund)

    # from min_bin to max_bin in 48ths of an octave (1/4 tones)

    # [1] bin index to frequency min_freq to max_freq in 48ths of an octave
    def b2f(index): return min_freq * np.power(np.power(2, 1.0/48.0), index)

    # [2] max_histo_bin is the bin at max_freq (FUCKING: rounds down?)
    max_histo_bin = int(math.log(max_freq / min_freq, math.pow(2, 1.0/48.0))) 

    # now, generate them
    histo = np.fromfunction(lambda x,y: b2f(y), (num_peaks_found, max_histo_bin))
  
    frqs_tile = np.tile(peaks_frqs, (max_histo_bin,1)).transpose()
    mags_tile = np.tile(peaks_mags, (max_histo_bin,1)).transpose()

    # likelihood function for each bin frequency
    def ml_a(amp): 
      """a factor depending on the amplitude of the ith peak"""
      return np.sqrt(np.sqrt(amp/max_amp))
      # return np.sqrt(np.sqrt(amp))
      # return np.ones_like(amp)
      # return amp

    def ml_t(r1, r2):
      """how closely the ith peak is tuned to a multiple of f"""
      max_dist = ml_width # cents
      cents = np.abs(ratio_to_cents_protected(r1,r2))
      dist = np.clip(1.0 - (cents / max_dist), 0, 1)
      return dist 

    def ml_i(nearest_multiple): 
      """whether the peak is closest to a high or low multiple of f"""
      out = np.zeros_like(nearest_multiple)
      out[nearest_multiple.nonzero()] = 1/np.power(nearest_multiple[nearest_multiple.nonzero()], harm_rolloff) 
      return out
      # return 1/np.power(np.clip(nearest_multiple + harm_offset, 1, 32767), harm_rolloff) * (nearest_multiple <= max_harm) 
      # return 1/np.power(nearest_multiple+1, 2)
      # return np.ones_like(nearest_multiple)

    ml = (ml_a(mags_tile) * \
      ml_t((frqs_tile/histo), (frqs_tile/histo).round()) * \
      ml_i((frqs_tile/histo).round())).sum(axis=0)

    # for debugging
    # ml_hat = (ml_a(mags_tile) * \
    #   ml_t((frqs_tile/histo), (frqs_tile/histo).round()) * \
    #   ml_i((frqs_tile/histo).round()))
    

    """super rough hack but work!
    bring this in as a control
    only tests in one direction (new is higher), adjust so ratio goes both ways
    what about which harms we're testing for should that be variable too
    """

    num_found = 0
    maybe = 0
    found_so_far = []
    bounce_list = list(np.ravel(pitches[i-bounce_hist:i]))
    # bounce_ratios = [1,2,3,4,5,6]
    bounce_ratios = [1.0, 2.0, 3.0/2.0, 3.0, 4.0] if bounce_ratios is None else bounce_ratios
    prev_frame = list(pitches[i-1]) if i>0 else []
    indices = ml.argsort()[::-1]
    while num_found < num_pitches and maybe <= ml.shape[0]:
      this_one = b2f(indices[maybe])
      # check bounce with this frame
      bounce1 = any([any([abs(ratio_to_cents(
        this_one/(other_one*harm))) < bounce_width 
        for harm in bounce_ratios]) 
        for other_one in bounce_list])
      bounce2 = any([any([abs(ratio_to_cents
        (other_one/(this_one*harm))) < bounce_width
        for harm in bounce_ratios]) 
        for other_one in bounce_list])
      # if any bounces
      bounce = bounce1 or bounce2 
      if not bounce:
        #print 'notbounce'
        found_so_far += [this_one]
        bounce_list += [this_one]
        num_found += 1
      # if bounce: print 'bounce!!!!'
      maybe += 1

    indices = ml.argsort()[::-1][0:num_pitches]
    pitches[i] = np.array(found_so_far)
    confidences[i] = ml[indices]
    peaks[i] = peaks_frqs


    # indices = ml.argsort()[::-1][0:num_pitches]
    # pitches[i] = b2f(indices)
    # confidences[i] = ml[indices]
    # peaks[i] = peaks_frqs

    # ----------------------------------------------------------------------- #
    # estimate fundamental (least squares)
    # ----------------------------------------------------------------------- #

    """estimate fundamental (least squares)
    here we solve a least squares approximation to give a more precise
    estimate of the fundamental within a histogram bin.
    
    least squares solution :: WAx ~ Wb
   
    A = matrix of harmonic integers (num_peaks x 1)
    b = matrix of actual frequencies (num_peaks x 1)
    W = matrix of weights (on digaonal, num_peaks x num_peaks)
    x = fundamental (singletone matrix)
   
    """

    ml_peaks = pitches[i] # regions strong ml
    width = 25 # count peaks w/in 25 cents

    frame_fundamentals = []
    for bin_frq in ml_peaks:

      # vector of nearest harmonics
      nearest_harmonic = (peaks_frqs/bin_frq).round()

      # mask in range? vector of bools
      mask = np.abs(ratio_to_cents_protected( 
        (peaks_frqs/bin_frq), (peaks_frqs/bin_frq).round())) <= width

      # weight (same harmonic weight as above)
      weights = ml_i( (peaks_frqs/bin_frq).round() )

      # build matrices
      A = np.matrix(nearest_harmonic).T
      b = np.matrix(peaks_frqs).T
      W = np.matrix(np.diag(mask * weights))

      # do least squares
      fund = np.linalg.lstsq(W*A, W*b)[0][0].item()

      # append
      frame_fundamentals += [fund]

    fundamentals[i] = np.array(frame_fundamentals)

  # ------------------------------------------------------------------------- #
  # pitch tracks
  # ------------------------------------------------------------------------- #

  tracks = np.copy(pitches)

  return fundamentals, pitches, D, peaks, confidences, tracks


# --------------------------------------------------------------------------- #
# extras
# --------------------------------------------------------------------------- #


def voice_tracks(pitches, confidences):

  # alloc output data
  out = np.zeros_like(pitches)
  out[0] = pitches[0]

  out_confidences = np.zeros_like(confidences)
  out_confidences[0] = confidences[0]

  num_frames, num_voices = pitches.shape

  # loop through frames
  for i in range(1,num_frames):

    # setup
    prev_frame = out[i-1]
    next_frame = pitches[i]

    # pairwise distances indexed [from, to]
    delta = np.abs(ratio_to_cents(np.atleast_2d(prev_frame).T/np.repeat(
      np.atleast_2d(next_frame),num_voices,axis=0)))
    # delta = np.abs(ratio_to_cents_protected(np.atleast_2d(prev_frame).T,
      # np.repeat(np.atleast_2d(next_frame),num_voices,axis=0)))

    # indices of sorted delta [from, to]
    delta_sorted = np.unravel_index(np.argsort(delta.ravel()), delta.shape)

    # step through and reject if either prev or next is already found
    num_found = 0; found_prevs = []; found_nexts = []; index = 0

    while num_found < num_voices:
      prev_voice,next_voice = delta_sorted[0][index], delta_sorted[1][index]
      if prev_voice not in found_prevs and next_voice not in found_nexts:
        out[i][prev_voice] = pitches[i][next_voice]
        out_confidences[i][prev_voice] = confidences[i][next_voice]
        found_prevs += [prev_voice]
        found_nexts += [next_voice]
        num_found += 1
      index += 1

  return out, out_confidences


# --------------------------------------------------------------------------- #

"""pitchsets.
  this is slightly ineffient b/c we filter twice
    1. first for change (vibrato depth)
    2. then for sustain (vibrato length)
    3. whatever is left are ampsets
    - it could all be done in one pass, but this makes it easier to swap in
      change measures (absolute vs relative grid) and sustain measures
  * currently pitchsets are detected by voice but reported by frame to be 
    consistent with ampsets. 
    
  -> this should change.
"""


def pitchsets(pitches, win=3):

  # 1. filter for change on an asbolute 1/2-tone grid (depth)
  pitches_change = change_filter(pitches)

  # 2. filter for sustain (length)
  pitches_sustain = sustain_filter(pitches_change, win=win)

  # 3. what is left are pitchsets
  pitchsets = np.sort(np.array(list(set(np.argwhere(np.nan_to_num(pitches_sustain))[:,0]))))
  
  return pitchsets, np.where(np.nan_to_num(pitches_sustain))

# --------------------------------------------------------------------------- #

"""filter for change.
  * filter for change on grid rounded to nearest 1/2 tone
  * non-change pitches are changed to 0's
  -> maybe this should be np.nan's instead
"""


def change_filter(pitches):

  # copy working data
  wpitches = np.copy(pitches)

  # alloc output data
  out = np.zeros_like(wpitches)
  out[out==0] = np.nan

  # convert to midi pitch and round
  wpitches = freq_to_midi(wpitches).round()

  # indices of pitch change
  indices = np.diff(wpitches, axis=0).nonzero()

  # copy pitch values from original data where change
  out[0] = pitches[0]
  out[1:][indices] = pitches[1:][indices]

  return out 

# --------------------------------------------------------------------------- #

"""filter for sustain.
  * say something here: wish could be numpy...
"""


def sustain_filter(pitches, win=3):

  # alloc output data
  out = np.zeros_like(pitches)

  # loop through frames and pitches
  for i in range(pitches.shape[0]-win):
    for j in range(pitches.shape[1]):

        # keep pitch if win after are nan
        out[i,j] = pitches[i,j] * np.isnan(pitches[i+1:i+win,j]).all()

  # convert 0 to nan
  out[out==0] = np.nan

  return out 


# --------------------------------------------------------------------------- #

  """pitch tracks
  here we parse the raw frequencies into pitch tracks. currently, we do this 
  by minimizing the abs of the frame to frame difference, but we could use
  any feature, such as distance from the running mean, variance, etc.
    todo: --> implement more pitch track options
    
  """

  # move through the frames
  # have pitch tracks ready to go
  # for each num_pitches put in track with nearest frame to frame difference
  # note: what if the pitch confidence is low, might want to put track to sleep...
  # or if it's too far from anything just for a sec
  # or more sophisticated for like minimizing total error...
  # here we do it dumb way from strong to weak closest match (pref to first)
  # tracks = np.zeros([num_frames, num_pitches])   # confidence scores
  
  # tracks[0] = fundamentals[0]

  # tracks[0] = fundamentals[0]
  # for i in range(1, num_frames):
  #   prev_tracks = list(tracks[i-1])
  #   new_funds = list(fundamentals[i])
  #   for j in range(num_pitches):
  #     wewant = min(new_funds, key=lambda x: abs(x-prev_tracks[j]))
  #     index = new_funds.index(wewant)
  #     new_funds.pop(index)
  #     tracks[i,j]= wewant


def pitch_onsets(funds, win=2, depth=25):
  
  onsets = np.zeros_like(funds)
  for i in range(funds.shape[0]):
    for j in range(funds.shape[1]):
      onsets[i,j] = (np.abs(pyfid.ratio_to_cents(funds[i-win:i,:] / funds[i,j])) <= thresh).any(axis=1).all()

  return onsets
