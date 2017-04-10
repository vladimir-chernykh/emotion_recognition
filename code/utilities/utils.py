import os
import csv
import wave
import sys
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import calculate_features as calc_feat


########################################################################################################################
#                                                 constants                                                            #
########################################################################################################################


class Constants:
    def __init__(self):
        real_path = os.path.dirname(os.path.realpath(__file__))
        self.available_emotions = np.array(['ang', 'exc', 'neu', 'sad'])
        self.path_to_data = real_path + "/../../data/sessions/"
        self.path_to_features = real_path + "/../../data/features/"
        self.sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        self.conf_matrix_prefix = 'iemocap'
        self.framerate = 16000
        self.types = {1: np.int8, 2: np.int16, 4: np.int32}
    
    def __str__(self):
        def display(objects, positions):
            line = ''
            for i in range(len(objects)):
                line += str(objects[i])
                line = line[:positions[i]]
                line += ' ' * (positions[i] - len(line))
            return line
        
        line_length = 100
        ans = '-' * line_length
        members = [attr for attr in dir(self) if not callable(attr) and not attr.startswith("__")]
        for field in members:
            objects = [field, getattr(self, field)]
            positions = [30, 100]
            ans += "\n" + display(objects, positions)
        ans += "\n" + '-' * line_length
        return ans


########################################################################################################################
#                                                 data reading                                                         #
########################################################################################################################


def get_audio(path_to_wav, filename, params=Constants()):
    wav = wave.open(path_to_wav + filename, mode="r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    content = wav.readframes(nframes)
    samples = np.fromstring(content, dtype=params.types[sampwidth])
    return (nchannels, sampwidth, framerate, nframes, comptype, compname), samples


def get_transcriptions(path_to_transcriptions, filename, params=Constants()):
    f = open(path_to_transcriptions + filename, 'r').read()
    f = np.array(f.split('\n'))
    transcription = {}
    for i in range(len(f) - 1):
        g = f[i]
        i1 = g.find(': ')
        i0 = g.find(' [')
        ind_id = g[:i0]
        ind_ts = g[i1+2:]
        transcription[ind_id] = ind_ts
    return transcription


def get_emotions(path_to_emotions, filename, params=Constants()):
    f = open(path_to_emotions + filename, 'r').read()
    f = np.array(f.split('\n'))
    idx = f == ''
    idx_n = np.arange(len(f))[idx]
    emotion = []
    for i in range(len(idx_n) - 2):
        g = f[idx_n[i]+1:idx_n[i+1]]
        head = g[0]
        i0 = head.find(' - ')
        start_time = float(head[head.find('[') + 1:head.find(' - ')])
        end_time = float(head[head.find(' - ') + 3:head.find(']')])
        actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:
                        head.find(filename[:-4]) + len(filename[:-4]) + 5]
        emo = head[head.find('\t[') - 3:head.find('\t[')]
        vad = head[head.find('\t[') + 1:]

        v = float(vad[1:7])
        a = float(vad[9:15])
        d = float(vad[17:23])
        
        j = 1
        emos = []
        while g[j][0] == "C":
            head = g[j]
            start_idx = head.find("\t") + 1
            evoluator_emo = []
            idx = head.find(";", start_idx)
            while idx != -1:
                evoluator_emo.append(head[start_idx:idx].strip().lower()[:3])
                start_idx = idx + 1
                idx = head.find(";", start_idx)
            emos.append(evoluator_emo)
            j += 1

        emotion.append({'start': start_time,
                        'end': end_time,
                        'id': filename[:-4] + '_' + actor_id,
                        'v': v,
                        'a': a,
                        'd': d,
                        'emotion': emo,
                        'emo_evo': emos})
    return emotion


def split_wav(wav, emotions, params=Constants()):
    (nchannels, sampwidth, framerate, nframes, comptype, compname), samples = wav

    left = samples[0::nchannels]
    right = samples[1::nchannels]

    frames = []
    for ie, e in enumerate(emotions):
        start = e['start']
        end = e['end']

        e['right'] = right[int(start * framerate):int(end * framerate)]
        e['left'] = left[int(start * framerate):int(end * framerate)]

        frames.append({'left': e['left'], 'right': e['right']})
    return frames


def read_iemocap_data(params=Constants()):
    data = []
    for session in params.sessions:
        path_to_wav = params.path_to_data + session + '/dialog/wav/'
        path_to_emotions = params.path_to_data + session + '/dialog/EmoEvaluation/'
        path_to_transcriptions = params.path_to_data + session + '/dialog/transcriptions/'

        files = os.listdir(path_to_wav)
        files = [f[:-4] for f in files if f.endswith(".wav")]
        for f in files:           
            wav = get_audio(path_to_wav, f + '.wav')
            transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
            emotions = get_emotions(path_to_emotions, f + '.txt')
            sample = split_wav(wav, emotions)

            for ie, e in enumerate(emotions):
                e['signal'] = sample[ie]['left']
                e.pop("left", None)
                e.pop("right", None)
                e['transcription'] = transcriptions[e['id']]
                if e['emotion'] in params.available_emotions:
                    data.append(e)
    sort_key = get_field(data, "id")
    return np.array(data)[np.argsort(sort_key)]


########################################################################################################################
#                                                 features generation                                                  #
########################################################################################################################


def get_features(data, params=Constants()):
    excluded = 0
    overall = 0
    IPy = False
    if "IPython.display" in sys.modules.keys():
        from IPython.display import clear_output
        IPy = True
    for di, d in enumerate(data):
        if di % 100 == 0:
            if IPy:
                clear_output()
            print(di, ' out of ', len(data))
        st_features = calc_feat.calculate_features(d['signal'], params.framerate, None).T
        x = []
        y = []
        for f in st_features:
            overall += 1
            if f[1] > 1.e-4:
                x.append(f)
                y.append(d['emotion'])
            else:
                excluded += 1
        x = np.array(x, dtype=float)
        y = np.array(y)
        save_sample(x, y, params.path_to_features + d['id'] + '.csv')
    return overall, excluded


########################################################################################################################
#                                                 helpers                                                              #
########################################################################################################################


def get_field(data, key):
    return np.array([e[key] for e in data])


def to_categorical(y, params=Constants()):
    return label_binarize(y, params.available_emotions)


def save_sample(x, y, name):
    with open(name, 'w') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        for i in range(x.shape[0]):
            row = x[i, :].tolist()
            row.append(y[i])
            w.writerow(row)

            
def load_sample(name):
    with open(name, 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        x = []
        y = []
        for row in r:
            x.append(row[:-1])
            y.append(row[-1])
    return np.array(x, dtype=float), np.array(y)


def get_sample(ids, take_all=False, params=Constants()):
    if take_all:
        files = os.listdir(params.path_to_features)
        ids = np.sort([f[:-4] for f in files if f.endswith(".csv")])
    tx = []
    ty = []
    valid_ids = []
    for i in ids:
        x, y = load_sample(params.path_to_features + i + '.csv')
        if len(x) > 0:
            tx.append(np.array(x, dtype=float))
            ty.append(y[0])
            valid_ids.append(i)
    tx = np.array(tx)
    ty = np.array(ty)
    return tx, ty, np.array(valid_ids)


def pad_sequence_old(x, ts, params=Constants()):
    xp = []
    for i in range(len(x)):
        x0 = np.zeros((ts, x[i].shape[1]), dtype=float)
        if ts > x[i].shape[0]:
            x0[ts - x[i].shape[0]:, :] = x[i]
        else:
            maxe = np.sum(x[i][0:ts, 1])
            x0 = x[i][0:ts, :]
            for j in range(x[i].shape[0] - ts):
                if np.sum(x[i][j:j + ts, 1]) > maxe:
                    x0 = x[i][j:j + ts, :]
                    maxe = np.sum(x[i][j:j + ts, 1])
        xp.append(x0)
    return np.array(xp)


def pad_sequence_into_array(Xs, maxlen=None, truncating='post', padding='post', value=0.):
    """
    Padding sequence (list of numpy arrays) into an numpy array
    :param Xs: list of numpy arrays. The arrays must have the same shape except the first dimension.
    :param maxlen: the allowed maximum of the first dimension of Xs's arrays. Any array longer than maxlen is truncated to maxlen
    :param truncating: = 'pre'/'post', indicating whether the truncation happens at either the beginning or the end of the array (default)
    :param padding: = 'pre'/'post',indicating whether the padding happens at either the beginning or the end of the array (default)
    :param value: scalar, the padding value, default = 0.0
    :return: Xout, the padded sequence (now an augmented array with shape (Narrays, N1stdim, N2nddim, ...)
    :return: mask, the corresponding mask, binary array, with shape (Narray, N1stdim)
    """
    Nsamples = len(Xs)
    if maxlen is None:
        lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
        maxlen = np.max(lengths)

    Xout = np.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
    Mask = np.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
    for i in range(Nsamples):
        x = Xs[i]
        if truncating == 'pre':
            trunc = x[-maxlen:]
        elif truncating == 'post':
            trunc = x[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
        if padding == 'post':
            Xout[i, :len(trunc)] = trunc
            Mask[i, :len(trunc)] = 1
        elif padding == 'pre':
            Xout[i, -len(trunc):] = trunc
            Mask[i, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return Xout, Mask


def convert_gt_from_array_to_list(gt_batch, gt_batch_mask=None):
    """
    Convert groundtruth from ndarray to list
    :param gt_batch: ndarray (B, L)
    :param gt_batch_mask: ndarray (B, L)
    :return: gts <list of size = B>
    """
    B, L = gt_batch.shape
    gt_batch = gt_batch.astype('int')
    gts = []
    for i in range(B):
        if gt_batch_mask is None:
            l = L
        else:
            l = int(gt_batch_mask[i, :].sum())
        gts.append(gt_batch[i, :l].tolist())
    return gts


########################################################################################################################
#                                               metrics                                                                #
########################################################################################################################


def weighted_accuracy(y_true, y_pred):
    return np.sum((np.array(y_pred).ravel() == np.array(y_true).ravel()))*1.0/len(y_true)


def unweighted_accuracy(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    classes = np.unique(y_true)
    classes_accuracies = np.zeros(classes.shape[0])
    for num, cls in enumerate(classes):
        classes_accuracies[num] = weighted_accuracy(y_true[y_true == cls], y_pred[y_true == cls])
    return np.mean(classes_accuracies)