import numpy as np
import wave
import time
import calculate_features as cf
from utils import *
from CTC import *

def get_label(emotions, utterance_number, framerate, istart, iend):
  ''' Returns True label and utterance number (for optimized search) '''
  if utterance_number < len(emotions):
    s = emotions[utterance_number]['start'] * framerate
    e = emotions[utterance_number]['end'] * framerate
    if iend <= s:
      label = '---'
    elif iend > s and iend <= e:
      label = emotions[utterance_number]['emotion']
    elif iend > e:
      utterance_number += 1
      s = emotions[utterance_number]['start'] * framerate
      e = emotions[utterance_number]['end'] * framerate
      if iend <= s:
        label = '---'
      elif iend > s and iend <= e:
        label = emotions[utterance_number]['emotion']
  else:
    label = '---'
  return label, utterance_number

c = Constants()
path_to_data = 'iemocap_data/'

framerate = c.framerate
length_sec = 4
length_pts = length_sec * framerate
step_sec = 0.2
step_pts = int(step_sec * framerate)
window_sec = 0.2
window_pts = int(window_sec * framerate)
window_step_sec = 0.1
window_step_pts = int(window_step_sec * framerate)


def save_samples():
  ''' Calculates and saves features into csv files '''
  for session in c.sessions:
    print(session)
    path = path_to_data + session + '/dialog/wav/'
    files = os.listdir(path)
    files = [f[:-4] for f in files]
    for fi, f in enumerate(files):
      print(fi, ' out of ', len(files))
      params, audio = get_audio(path, f + '.wav')
      audio = audio[::params[1]] # use the only one channel 
      framerate = params[2]

      n = params[3]

      emotions = get_emotions(path[:-4] + 'EmoEvaluation/', f + '.txt')
      m_emotions = []
      f_emotions = []
      for i in range(len(emotions)):
        if emotions[i]['id'].find('_M') > 0:
          m_emotions.append(emotions[i])
        if emotions[i]['id'].find('_F') > 0:
          f_emotions.append(emotions[i])
      m_emotions.append({'start':m_emotions[-1]['start'] + 1000, 'end':m_emotions[-1]['end'] + 1000, 'emotion':'---'})  # the last utterance should be --- 
      f_emotions.append({'start':f_emotions[-1]['start'] + 1000, 'end':f_emotions[-1]['end'] + 1000, 'emotion':'---'})  # the last utterance should be --- 

      #                           label intersection
      #--------------------------------------------------------------------------#
      # ms = []
      # me = []
      # fs = []
      # fe = []
      # for i in xrange(len(emotions)):
      #   print emotions[i]['id'], emotions[i]['id'].find('_M')
      #   if emotions[i]['id'].find('_M') > 0:
      #     ms.append(emotions[i]['start'])
      #     me.append(emotions[i]['end'])
      #   if emotions[i]['id'].find('_F') > 0:
      #     fs.append(emotions[i]['start'])
      #     fe.append(emotions[i]['end'])
      # fs = np.array(fs)
      # fe = np.array(fe)
      # ms = np.array(ms)
      # me = np.array(me)

      # print fs
      # print fe
      # print ms
      # print me
      # from matplotlib import pyplot
      # pyplot.axis([0, 110, -1, 1])
      # pyplot.plot([0, 110], [0, 0], color='black')
      # for i in xrange(len(fs)):
      #   pyplot.plot([fs[i], fe[i]], [0.1, 0.1], color='r', linewidth=3)
      # for i in xrange(len(ms)):
      #   pyplot.plot([ms[i], me[i]], [-0.1, -0.1], color='b', linewidth=3)
      # pyplot.show()
      #--------------------------------------------------------------------------#

      #print emotions

      f_utterance_number = 0
      m_utterance_number = 0
      f_labels_by_frame = []
      m_labels_by_frame = []
      features_by_frame = []
      for i in range((n - window_pts) / window_step_pts):
        istart = i * window_step_pts
        iend = istart + window_pts
        signal = audio[istart:iend]
        features = cf.calculate_features(signal, framerate)[:, 0]
        features_by_frame.append(features)

        m_label, m_utterance_number = get_label(m_emotions, m_utterance_number, framerate, istart, iend)
        f_label, f_utterance_number = get_label(f_emotions, f_utterance_number, framerate, istart, iend)

        m_labels_by_frame.append(m_label)
        f_labels_by_frame.append(f_label)
        # print istart / float(framerate), iend / float(framerate), n / float(framerate), m_utterance_number, len(m_emotions), f_utterance_number, len(f_emotions), m_label, f_label, m_emotions[m_utterance_number]['start'], m_emotions[m_utterance_number]['end'], m_emotions[m_utterance_number]['emotion'], f_emotions[f_utterance_number]['start'], f_emotions[f_utterance_number]['end'], f_emotions[f_utterance_number]['emotion']

      #with  as fil:
      with open('stream_features/' + f + '.csv', 'w') as fil:
        w = csv.writer(open('stream_features/' + f + '.csv', 'w'), delimiter=',')
        for i in range(len(features_by_frame)):
          d = features_by_frame[i].tolist()
          d.append(m_labels_by_frame[i])
          d.append(f_labels_by_frame[i])
          w.writerow(d)

def load_sample(path, filename):
  ''' Loads features and labels from file ''' 
  features = []
  ml = []
  fl = []
  with open(path + filename + '.csv', 'r') as fil:
    r = csv.reader(fil, delimiter=',')
    for row in r:
      features.append(row[:-2])
      ml.append(row[-2])
      fl.append(row[-1])
  return np.array(features, dtype=float), np.array(ml), np.array(fl)

def get_samples(path, files, calculate=False):
  ''' Returns the array of features and labels for an input files list '''
  if calculate:
    return get_samples(calculate=False)
  else:
    data = []
    m_labels = []
    f_labels = []
    for fi, f in enumerate(files):
      features, ml, fl = load_sample(path, f)
      for i in range(len(features)):
        time_sec = i * window_step_sec
        xs, y1s, y2s = get_sequence_sample(features, ml, fl, time_sec)
        data.append(xs)
        m_labels.append(y1s)
        f_labels.append(y2s)

    return np.array(data, dtype=float), np.array(m_labels), np.array(f_labels)

def get_sequence_sample(x, y1, y2, time_sec):
  length = int((length_sec - window_sec) / window_step_sec + 1)
    
  x_seq = np.zeros((length, x.shape[1]), dtype=float)
  y1_seq = ['---']*length
  y2_seq = ['---']*length

  i1 = int((time_sec - window_sec) / window_step_sec)
  i0 = i1 - length

  if i0 < 0:
    i0 = 0
  if i1 < 0:
    i1 = 0
  x_seq[length - (i1-i0):, :] = x[i0:i1, :]
  y1_seq[length - (i1-i0):] = y1[i0:i1]
  y2_seq[length - (i1-i0):] = y2[i0:i1]

  return x_seq, np.array(y1_seq), np.array(y2_seq)
