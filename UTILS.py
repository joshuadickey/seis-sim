import numpy as np
import pandas as pd
import errno
import time
import sys
import re
import os
import datetime
import obspy
from obspy.clients.fdsn import Client
from time import perf_counter

from pathlib import Path
import urllib.request
from math import radians, cos, sin, asin, sqrt
from obspy.signal import filter
from matplotlib.colors import LogNorm

import keras
import keras.layers
import tensorflow as tf
import keras.backend as K
from keras.models import load_model, Input, Model
from keras.layers import Conv1D, SpatialDropout1D, Flatten, Activation, Lambda, Convolution1D, Dense
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import matplotlib
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from shapely.geometry.polygon import LinearRing
from sklearn.manifold import TSNE

##################################################################################
#                           DOWNLOADER UTILITIES                                 #
##################################################################################



def detrend(X, axis=1):
    return X - np.expand_dims(X.mean(axis), axis)


def normalize(X):
    return X / np.expand_dims(np.max(np.abs(X), axis=1), 1)


def calc_snr(x_raw, s_p_time, n_filt=9, n_win=4, sta_time=2, arr_time=30, samp_rate=40, filt_order=2, plot=False):
    ''' This function takes in a raw seismogram and returns a spectrogram.
        The spectrogram windowing is based on the theoretical S-P time difference.

        x_raw:           RAW SEISMOGRAM
        samp_rate:       SAMPLE RATE OF SEISMOGRAM
        arr_time:        ARRIVAL TIME OF EVENT WITHIN SEISMOGRAM (IN SECONDS)
        sta_time:        LENGTH OF THE SHORT TERM AVERAGING WINDOW (IN SECONDS)
        s_p_time:        THEORETICAL S-P ARRIVAL TIME DIFFERENCE (IN SECONDS)
        n_win:           NUMBER OF WINDOWS IN SPECTROGRAM
        n_filt:          NUMBER OF FILTERS IN SPECTROGRAM
        filt_order:      ORDER OF FILTERS IN SPECTROGRAM
        plot:            BOOLEAN FOR PLOTTING

        The default phase windows contain the first P wave, P coda,
        the first S wave and S coda, all with the same length:
        half of the theoretical S–P arrival time difference.
        Two noise windows are also included at the start and end.
        The length of the noise windows is equal to the
        theoretical S–P arrival time difference.
    '''

    sta_samps = int(sta_time * samp_rate)
    arr_samps = int(arr_time * samp_rate)
    win_samps = int(s_p_time * samp_rate)

    # Built lists of time and freq bins for the spectrogram
    time_bins = list(map(tuple, (win_samps * np.array([(2 + t, 3 + t) for t in range(n_win)]) / 2).astype(int)))
    freq_bins = [(1, 3)] + [(f * 2, f * 2 + 3) for f in range(1, n_filt)]

    # Extract the spectrogram features
    x_sta = []
    feat = np.zeros((n_filt, n_win))
    for filt, (freqmin, freqmax) in enumerate(reversed(freq_bins)):

        # Perform the filtering
        x_filt = filter.bandpass(x_raw, freqmin, freqmax, samp_rate, filt_order, zerophase=True)

        # Compute the STA characteristic function
        x_sta.append(np.convolve(x_filt ** 2, np.ones((sta_samps,)) / sta_samps, mode='valid')[
                     int(arr_samps - win_samps):int(arr_samps + 3 * win_samps)])

        # Create the normalized averages for each phase window
        for win, (st, en) in enumerate(time_bins):
            feat[filt, win] = np.mean(x_sta[filt][st:en]) / np.mean(x_sta[filt])

    # Plot the filtered waveforms
    if plot:
        fig, ax = plt.subplots(n_filt, 1, sharey=False, sharex=True)
        t = np.array(sorted(set([i for sl in time_bins for i in sl])))
        for filt in range(n_filt):
            ax[filt].plot(x_sta[filt])
            ax[filt].set_xticks(t)
            ax[filt].set_xticklabels(np.round(t / samp_rate - t[0] / samp_rate, 2))
            ax[filt].set_yticklabels([])

            # Plot vertical lines defining the phase windows
            for st, en in time_bins:
                ax[filt].axvline(x=st, c='g', ls='--', lw=1)
                ax[filt].axvline(x=en, c='g', ls='--', lw=1)

        # print the numerical spectrogram
        print(feat)

    return 20 * np.log10(max(feat[:, 0]))


def cat_intfr(cat):
    cat_tmp = []
    for sta in cat.STA.unique():
        print(sta)
        Y_sta = cat.loc[cat.STA == sta]
        Y_sta['INTFR'] = Y_sta.TIME.diff() / np.timedelta64(1, 's') < 180
        cat_tmp.append(Y_sta)
    cat_tmp = pd.concat(cat_tmp)
    cat_tmp.sort_values('TIME', inplace=True)
    return cat_tmp


def catalog_snr(dataset_name, data_folder):
    dat_name = os.path.join(data_folder, f'X_{dataset_name}.npy')
    cat_name = os.path.join(data_folder, f'Y_{dataset_name}.csv')

    print('loading catalog and waveforms...')
    cat = read_cat(cat_name)
    dat = np.load(dat_name)

    if len(dat.shape) > 2:
        dat = dat[:, :, 0]
    print('calculating S-P time difference...')
    SPdiff_model = load_model('models/SPdiff_model.h5')
    cat['SPdiff'] = SPdiff_model.predict(cat[['DIST', 'DEPTH']].values)

    print('calculating SNR...')
    X = np.zeros((len(dat)))
    for row, event in cat.iterrows():
        event.SPdiff
        X[row] = calc_snr(dat[row], event.SPdiff)

    cat['SNR'] = X
    cat.to_csv(cat_name, index=False)

    print('complete')

    return cat


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def url_maker(lat, lon, sta_rad, evt_rad, yr, mm, pc, pt):
    if evt_rad == 0:
        evt_sh = 'GLOBAL'
    else:
        evt_sh = 'CIRC'

    if sta_rad == 0:
        sta_sh = 'GLOBAL'
    else:
        sta_sh = 'CIRC'

    if (mm == 12) & (pc == pt - 1):
        en_yr = yr + 1
        en_mm = 1
    elif (mm != 12) & (pc == pt - 1):
        en_yr = yr
        en_mm = mm + 1
    else:
        en_mm = mm
        en_yr = yr

    dd = np.ones(pt + 1)
    for i in range(1, pt):
        dd[i] = 30 * i / pt
    dd = dd.astype('int')
    url = f'http://www.isc.ac.uk/cgi-bin/web-db-v4?iscreview=&out_format=CSV&ttime=on&ttres=on&tdef=on&phaselist=&sta_list=&stnsearch={sta_sh}&stn_ctr_lat={lat}&stn_ctr_lon={lon}&stn_radius={sta_rad}&max_stn_dist_units=deg&stn_top_lat=&stn_bot_lat=&stn_left_lon=&stn_right_lon=&stn_srn=&stn_grn=&bot_lat=&top_lat=&left_lon=&right_lon=&searchshape={evt_sh}&ctr_lat={lat}&ctr_lon={lon}&radius={evt_rad}&max_dist_units=deg&srn=&grn=&start_year={yr}&start_month={mm}&start_day={dd[pc]}&start_time=00%3A00%3A00&end_year={en_yr}&end_month={en_mm}&end_day={dd[pc + 1]}&end_time=00%3A00%3A00&min_dep=&max_dep=&min_mag=&max_mag=&req_mag_type=Any&req_mag_agcy=Any&include_links=on&request=STNARRIVALS'

    return url


def url_open(url, filename):
    # Make 10 attempts at downloading the catalog
    reattempt = 0
    while reattempt < 10:

        # Try 10 times to retrieve the url
        retrycount = 0
        s = None
        while s is None:
            try:
                s = urllib.request.urlretrieve(url, filename)
            except Exception as e:
                print(str(e))
                retrycount += 1
                if retrycount > 5:
                    print(" download failed")
                    # silentremove(filename)

        # Check to see if the retrieval was rejected by the server
        # If the retrieval was rejected, reattempt!
        if open(filename).readlines()[23][:5] == 'Sorry':
            reattempt += 1
            print(' reattempt', reattempt, '...', sep='', end='')
            sys.stdout.flush()
            time.sleep(60)
        else:
            return
    print(" download failed")
    # silentremove(filename)


def clean_csv(filename):
    lines = open(filename).readlines()
    first_chars = [line[:6] for line in lines]

    idx_st = [i for i in range(len(first_chars)) if first_chars[i] == 'EVENTI']
    idx_en = [i for i in range(len(first_chars)) if first_chars[i] == 'STOP\n']

    if len(idx_en) > 0:
        lines = lines[idx_st[0]:idx_en[0] - 1]
        lines = [re.sub('\<(.*?)\>', '', line, count=0, flags=0) for line in lines]

        with open('tmp.csv', 'w') as f:
            f.writelines(lines)

        cat_event = pd.read_csv('tmp.csv')

        cat_event.columns = cat_event.columns.str.replace('.', '_')
        cat_event.columns = cat_event.columns.str.replace(' ', '')
        cat_event = cat_event.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        cat_event['TIME'] = (cat_event['DATE'] + ' ' + cat_event['TIME'])
        cat_event['TIME'] = pd.to_datetime(cat_event['TIME'])
        cat_event['Y'] = np.where(cat_event['DEPTH'] == 0, 1, 0)
        for col in ['LAT', 'LON', 'ELEV', 'AMPLITUDE', 'PER', 'MAG']:
            cat_event[col] = pd.to_numeric(cat_event[col])

        return cat_event


def get_cat(yr, mm, lat, lon, sta_rad, evt_rad, cat_dir, pt=5, sta_list=None):
    filename = os.path.join(cat_dir, f'{yr}_{mm}_arrivals.csv')
    print(f'downloading {filename}...', end=' ', flush=True)
    if Path(filename).is_file():
        print("already downloaded")
        cat = read_cat(filename)
    else:

        cat = pd.DataFrame()
        for i in range(pt):
            url = url_maker(lat, lon, sta_rad, evt_rad, yr, mm, i, pt)
            print(url)
            url_open(url, 'tmp.csv')
            cat = cat.append(clean_csv('tmp.csv'))

        if sta_list is not None:
            cat = cat.loc[cat.STA.isin(sta_list)]

        cat = cat.sort_values('TIME', axis=0)
        cat = cat.drop_duplicates(subset=['EVENTID', 'STA']).reset_index(drop=True)

        cat.to_csv(filename, index=False)
        print("download complete")

    return cat


def read_cat(cat_file):
    cat = pd.read_csv(cat_file)
    cat['TIME'] = pd.to_datetime(cat['TIME'])
    cat['EVENTID'] = cat['EVENTID'].astype('int')
    return cat


def cat_to_dat(cat, output_name, pre_trim=30, post_trim=150, sample_rate=40):
    dataset_name = os.path.basename(output_name)
    output_dir = os.path.dirname(output_name)

    X_name = os.path.join(output_dir, f'X_{dataset_name}.npy')
    Y_name = os.path.join(output_dir, f'Y_{dataset_name}.csv')

    if not os.path.exists(X_name):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        c = Client("IRIS")
        X = []
        Y = []

        for i, event in cat.iterrows():
            print(f'{i:06d} - fetching {event.STA: <6}{event.TIME}... ', end='', flush=True)
            dat = load_dat(event.STA, event.TIME, c, pre_trim, post_trim, sample_rate)

            if dat is not None:
                print('ADDED')
                X.append(dat)
                Y.append(pd.DataFrame(event).T)
            else:
                print('FAILED')

        if len(X) > 0:
            X = np.stack(X)
            Y = pd.concat(Y, ignore_index=True)

            np.save(X_name, X)
            Y.to_csv(Y_name, index=False)

    return


def load_dat(sta, time, c, pre_trim, post_trim, sample_rate=40):
    win_len = sample_rate * (pre_trim + post_trim)

    try:
        stream = c.get_waveforms(network="*", location="",
                                 station=sta, channel="BH?",
                                 starttime=obspy.UTCDateTime(time) - pre_trim,
                                 endtime=obspy.UTCDateTime(time) + post_trim).resample(sample_rate)

        if (len(stream) == 3) & (len(stream[0]) >= win_len) & (len(stream[1]) >= win_len) & (len(stream[2]) >= win_len):
            return np.vstack([stream.select(component="Z")[0][:win_len],
                              stream.select(component="N")[0][:win_len],
                              stream.select(component="E")[0][:win_len]]).T
    except:
        pass

    return None


def yrmm_range(st, en):
    yrmm = []
    my_yr, my_mm = [int(x) for x in st.split('-')]
    en_yr, en_mm = [int(x) for x in en.split('-')]

    while (my_yr, my_mm) != (en_yr, en_mm):
        yrmm.append((my_yr, my_mm))
        my_mm += 1
        if my_mm == 13:
            my_mm = 1
            my_yr += 1

    return yrmm


def agg_dat(st, en, d, dataset_name=None, exc_stalst=None, exc_coords=None, inc_coords=None):
    files = os.listdir(d)
    if dataset_name is None:
        dataset_name = f'USArray{st}_{en}'
    x_names = [f'X_{yr}{mm:02d}.npy' for (yr, mm) in yrmm_range(st, en)]
    y_names = [f'Y_{yr}{mm:02d}.csv' for (yr, mm) in yrmm_range(st, en)]
    x_paths = [os.path.join(d, f) for f in x_names if f in files]
    y_paths = [os.path.join(d, f) for f in y_names if f in files]

    X = []
    Y = []
    for xp, yp in zip(x_paths, y_paths):
        print(f'loading {xp}...')
        x = np.load(xp)

        print(f'loading {yp}...')
        y = read_cat(yp)

        if exc_stalst is not None:
            y = y.loc[~ y.STA.isin(exc_stalst)]
        if exc_coords is not None:
            y = y.loc[~ ((y.LAT_1 > exc_coords['minlat']) & (y.LAT_1 < exc_coords['maxlat']) &
                         (y.LON_1 < exc_coords['maxlon']) & (y.LON_1 > exc_coords['minlon']))]
        if inc_coords is not None:
            y = y.loc[(y.LAT_1 > inc_coords['minlat']) & (y.LAT_1 < inc_coords['maxlat']) &
                      (y.LON_1 < inc_coords['maxlon']) & (y.LON_1 > inc_coords['minlon'])]

        x = np.take(x, y.index, axis=0)
        y.reset_index(drop=True, inplace=True)

        X.append(x)
        Y.append(y)

    print('merging dataframes...')
    X = np.concatenate(X)
    Y = pd.concat(Y, ignore_index=True)
    print('saving dataframes...')
    X_name = os.path.join(d, f'X_{dataset_name}.npy')
    Y_name = os.path.join(d, f'Y_{dataset_name}.csv')
    np.save(X_name, X)
    Y.to_csv(Y_name, index=False)
    print('COMPLETE!')

    return X, Y


##################################################################################
#                               TRAINER UTILITIES                                #
##################################################################################



def load_custom_model(pdict, model_folder='models'):
    if 'iniW' in pdict.keys():
        model_name = [n for n in os.listdir(model_folder) if f"|time:{pdict['iniW']}" in n][0]
        model_file = os.path.join(model_folder, model_name)
        
        print('loading previous model:\n', model_file)
        pdict = name2param(os.path.basename(model_file))
        my_loss, my_acc = get_loss_acc(pdict['numP'], pdict['numK'], pdict['a'], pdict['m'])
        model = load_model(model_file, custom_objects={'triplet_loss': my_loss, 'triplet_acc': my_acc})
    else:
        model_name = param2name(pdict)
        model_file = os.path.join(model_folder, model_name)
        
        print('building new model:\n', model_file)
        model = encoder_network(input_shape=((pdict['pre'] + pdict['post']) * 40, 3),
                                nb_filters=pdict['f'],
                                kernel_size=pdict['k'],
                                dilatations=pdict['d'],
                                nb_stacks=pdict['s'],
                                norm=pdict['n'],
                                out_dim=pdict['od'])
    return model, model_name


def name2param(name):
    name = name[:-3]
    regnumber = re.compile(r'^\d+(\.\d+)?$')
    pdict = dict([p.split(':') for p in name.split('|')])
    for key in pdict.keys():
        if regnumber.match(pdict[key]):
            try:
                pdict[key] = int(pdict[key])
            except:
                pdict[key] = float(pdict[key])
        else:
            if 'x' in pdict[key]:
                pdict[key] = list(map(int, pdict[key].split('x')))
            try:
                pdict[key] = float(pdict[key])
            except:
                pass
    return pdict


def load_custom_model2(model_def):
    if type(model_def) is dict:
        pdict = model_def
        model = encoder_network(input_shape=((pdict['pre']+pdict['post'])*40, 3),
                                nb_filters=pdict['f'],
                                kernel_size=pdict['k'],
                                dilatations=pdict['d'],
                                nb_stacks=pdict['s'],
                                norm=pdict['n'],
                                out_dim=pdict['od'])
    else:
        model_file = model_def
        pdict = name2param(os.path.basename(model_file))
        my_loss, my_acc = get_loss_acc(pdict['numP'], pdict['numK'], pdict['a'], pdict['m'])
        model = load_model(model_file, custom_objects={'triplet_loss': my_loss, 'triplet_acc': my_acc})

    return model, param2name(pdict)


def compile_custom_model(model, pdict):
    my_loss, my_acc = get_loss_acc(pdict['numP'], pdict['numK'], pdict['a'], pdict['m'])
    o = keras.optimizers.Adam(lr=pdict['lr'], clipnorm=1.)
    model.compile(loss=my_loss, optimizer=o, metrics=[my_acc])
    return model, param2name(pdict)


def get_callbacks(model_name, model_folder, log_folder):
    pdict = name2param(os.path.basename(model_name))
    tensor_foldername = os.path.join(log_folder, model_name)
    model_filename = os.path.join(model_folder, model_name + '.h5')

    sv = keras.callbacks.ModelCheckpoint(filepath=model_filename, monitor='val_triplet_acc', save_best_only=True,
                                         save_weights_only=False, mode='max')
    tbd = keras.callbacks.TensorBoard(log_dir=tensor_foldername)
    stp = keras.callbacks.EarlyStopping(monitor='val_triplet_acc', min_delta=0, patience=pdict['pat'],
                                        verbose=0, mode='max', baseline=None)

    return [sv, tbd, stp]


def param2name(pdict):
    name = []
    for key in pdict.keys():
        if type(pdict[key]) is list:
            name.append(f'{key}:{"x".join(map(str, pdict[key]))}')
        else:
            name.append(f'{key}:{pdict[key]}')
    return '|'.join(name)


def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def residual_block(x, s, i, nb_filters, kernel_size):
    original_x = x
    conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=2 ** i, padding='same',
                  name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)
    x = Activation('relu')(conv)
    x = Lambda(channel_normalization)(x)
    x = SpatialDropout1D(0.05)(x)

    # 1x1 conv.
    x = Convolution1D(nb_filters, 1, padding='same')(x)
    res_x = keras.layers.add([original_x, x])
    return res_x, x


def encoder_network(input_shape, nb_filters, kernel_size, dilatations, nb_stacks, norm=False, out_dim=64):
    activation = 'relu'
    input_layer = Input(name='input_layer', shape=input_shape)
    x = input_layer
    x = Convolution1D(nb_filters, kernel_size, padding='same', name='initial_conv')(x)

    skip_connections = []
    for s in range(nb_stacks):
        for i in dilatations:
            x, skip_out = residual_block(x, s, i, nb_filters, kernel_size)
            skip_connections.append(skip_out)

    x = keras.layers.add(skip_connections)
    x = Activation('relu')(x)

    x = Conv1D(1, kernel_size=kernel_size, activation=activation)(x)  # CONV1D

    # regression
    x = Flatten()(x)
    x = Dense(out_dim)(x)
    x = Activation('linear', name='output_dense')(x)

    # normalize output for cosine similarity
    if norm:
        x = Lambda(lambda xx: K.l2_normalize(xx, axis=1))(x)

    output_layer = x
    print(f'model.x = {input_layer.shape}')
    print(f'model.y = {output_layer.shape}')
    model = Model(input_layer, output_layer, name='base_model')

    return model



def shakenbake2(orig_x, sh=9):
    orig_len = len(orig_x)
    sh = int(sh*40)
    new_x = np.pad(orig_x, ((sh, sh), (0, 0)), 'edge')
    st_idx = np.random.randint(2 * sh)
    en_idx = st_idx + orig_len
    new_x = new_x[st_idx:en_idx, :]

    return new_x


def shakenbake(orig_x, pct_sh=.05):
    orig_len = len(orig_x)
    sh = int(orig_len * pct_sh)
    new_x = np.pad(orig_x, ((sh, sh), (0, 0)), 'edge')
    st_idx = np.random.randint(2 * sh)
    en_idx = st_idx + orig_len
    new_x = new_x[st_idx:en_idx, :]

    return new_x


def generator_hard2(data, catalog, n_P, n_K, sh_pct=9, return_cat = False):
    vc = catalog.EVENTID.value_counts()
    catalog = catalog[catalog.EVENTID.isin(vc.index[vc.values >= n_K])]
    bs = n_P*n_K
    while 1:

        # Declare variables to store the inputs (x) and outputs (labels) for a single batch
        label_cat = pd.DataFrame()
        labels = np.zeros(bs)
        x = np.zeros((bs, data.shape[1], data.shape[2]))

        # sample the catalog for P*K rows around some central tendency defined by r,m,d,t
        while len(label_cat) < bs:
            try:
                for my_event_id in np.random.choice(catalog.EVENTID.unique(), size=n_P, replace=False):
                    label_cat = label_cat.append(catalog.loc[catalog.EVENTID == my_event_id].sample(n_K))
            except:
                label_cat = pd.DataFrame()

        # For each row in the sample, generate the input/output pairs
        label_cat = label_cat.rename_axis('idx').reset_index()
        for row, event in label_cat.iterrows():
            labels[row] = event.EVENTID
            x[row] = shakenbake(data[event.idx], sh_pct)
        if return_cat:
            yield x, labels.astype(int), label_cat
        else:
            yield x, labels.astype(int)


def generator_hard(data, catalog, n_P, n_K, r=99, m=99, d=99, t=1e9, sh_pct=.05, return_cat = False):
    vc = catalog.EVENTID.value_counts()
    catalog = catalog[catalog.EVENTID.isin(vc.index[vc.values >= n_K])]
    bs = n_P*n_K
    while 1:

        # Declare variables to store the inputs (x) and outputs (labels) for a single batch
        label_cat = pd.DataFrame()
        labels = np.zeros(bs)
        

        # sample the catalog for P*K rows around some central tendency defined by r,m,d,t
        while len(label_cat) < bs:
            central_event = catalog.sample(1).iloc[0]
            my_cat = catalog[(catalog.LAT_1 < central_event.LAT_1 + r) &
                                    (catalog.LAT_1 > central_event.LAT_1 - r) &
                                    (catalog.LON_1 < central_event.LON_1 + r) &
                                    (catalog.LON_1 > central_event.LON_1 - r) &
                                    (catalog.MAG < central_event.MAG + m) &
                                    (catalog.MAG > central_event.MAG - m) &
                                    (catalog.DEPTH < central_event.DEPTH + d) &
                                    (catalog.DEPTH > central_event.DEPTH - d) &
                                    (catalog.TIME < (central_event.TIME + datetime.timedelta(seconds=t))) &
                                    (catalog.TIME > (central_event.TIME - datetime.timedelta(seconds=t)))
                                    ]
            try:
                for my_event_id in np.random.choice(my_cat.EVENTID.unique(), size=n_P, replace=False):
                    label_cat = label_cat.append(catalog.loc[catalog.EVENTID == my_event_id].sample(n_K))
            except:
                label_cat = pd.DataFrame()


        if return_cat:
            labels = label_cat.EVENTID.values
            yield label_cat, labels.astype(int)
        else:
            x = np.zeros((bs, data.shape[1], data.shape[2]))
            # For each row in the sample, generate the input/output pairs
            label_cat = label_cat.rename_axis('idx').reset_index()
            for row, event in label_cat.iterrows():
                labels[row] = event.EVENTID
                x[row] = shakenbake(data[event.idx], sh_pct)
            yield x, labels.astype(int)


def get_triplet_dists(x, margin=0.5):
    x = K.transpose(x)
    max_pos = tf.gather(x, 0)
    min_neg = tf.gather(x, 1)
    # Use relu or softplus
    L_triplet = K.expand_dims(K.softplus(margin + max_pos - min_neg), 1)
    return L_triplet


def get_pairwise_dists(x, num_p, num_k):
    # pairwise distances for whole batch
    # (redundant computation but probably still faster than alternative)
    norms = tf.reduce_sum(x * x, 1)
    norms = tf.reshape(norms, [-1, 1])
    dists = norms - 2 * tf.matmul(x, x, transpose_b=True) + tf.transpose(norms)
    dists = K.sqrt(K.relu(dists))

    # get the max intra-class distance for each sample
    max_pos = [tf.reduce_max(tf.slice(dists, [i * num_k, i * num_k], [num_k, num_k]), axis=1) for i in range(0, num_p)]
    max_pos = K.concatenate(max_pos, axis=0)

    # get the min inter-class distance for each sample
    min_neg = []
    for i in range(0, num_p):
        left = tf.slice(dists, [i * num_k, 0], [num_k, i * num_k])
        right = tf.slice(dists, [i * num_k, (i + 1) * num_k], [num_k, (num_p - i - 1) * num_k])
        min_neg.append(tf.reduce_min(K.concatenate([left, right], axis=1), axis=1))
    min_neg = K.concatenate(min_neg, axis=0)

    min_max = K.concatenate([K.expand_dims(max_pos, axis=-1), K.expand_dims(min_neg, axis=-1)], axis=1)
    return min_max, dists


def get_loss_acc(n_P, n_K, alpha=.001, margin=0.5):
    def triplet_loss(y_true, y_pred):
        min_max_dists, all_dists = get_pairwise_dists(y_pred, n_P, n_K)
        loss1 = get_triplet_dists(min_max_dists, margin)
        loss2 = 1 / (all_dists + 1e-8)
        return K.mean(loss1) + alpha * K.mean(loss2)

    def triplet_acc(y_true, y_pred):
        dists, _ = get_pairwise_dists(y_pred, n_P, n_K)
        loss = get_triplet_dists(dists, margin)
        pos = K.less(loss, .5)
        return K.mean(pos)

    return triplet_loss, triplet_acc


##################################################################################
#                                TESTER UTILITIES                                #
##################################################################################

                        



def test_all_models(dataset_name, model_folder='models/', data_folder='data/datasets/', overwrite=False):
    results_path = f'Results_{dataset_name}.csv'
    print(results_path)
    results_cols = ['model_id', 'assoc_precis', 'assoc_recall', 'assoc_accury', \
                    'discr_precis', 'discr_recall', 'discr_accury']

    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path, index_col='model_id')
    else:
        results_df = pd.DataFrame(columns=results_cols).set_index('model_id')

    dat = np.load(os.path.join(data_folder, f'X_{dataset_name}.npy'))

    model_ids = [name2param(n)['time'] for n in os.listdir(model_folder) if '|time:' in n]
    print(model_ids)

    if overwrite == False:
        model_ids = list(set(set(model_ids) - set(results_df.index.values)))
    print(model_ids)

    for model_id in model_ids:
        print(model_id)

        try:
            results_df.loc[model_id] = test_model(model_id, dataset_name, dat)
            results_df.to_csv(results_path)
        except:
            pass

    return results_df



def test_model(model_id, dataset_name, dat=None, data_folder='data/datasets/'):
                        
    total_results = []
    # LOAD CATALOG AND EMBEDDINGS
    cat_path = os.path.join(data_folder, f'Y_{dataset_name}.csv')
    dat_path = os.path.join(data_folder, f'X_{dataset_name}.npy')
    emb_path = os.path.join(data_folder, f'E_{dataset_name}_{model_id}.npy')

    cat = read_cat(cat_path)

    if os.path.exists(emb_path):
        emb = np.load(emb_path)
    else:
        pdict = {}
        pdict['iniW'] = model_id
        model, model_name = load_custom_model(pdict)
        pdict = name2param(model_name)
        if dat is None:
            dat = np.load(dat_path)
        if pdict['norm']:
            dat = normalize(dat)
        if pdict['dtrd']:
            dat = detrend(dat)

        emb = model.predict(dat)
        np.save(emb_path, emb)

    # get association results
    _, assoc_results = assoc_test_gen(cat, emb, samps=5000)

    # get discrimination results
    _, discr_results = classPerf(cat, 'Y', emb, samps=1000, alpha=0.05)
    
    total_results.extend(assoc_results)
    total_results.extend(discr_results)
    print(total_results)

    return total_results


def classPerf(cat, feat, data_embeds, samps, alpha=None):
    new_cat = cat.copy(deep=True)
    pos_idx = new_cat.loc[new_cat[feat] == 1].sample(samps).rename_axis('idx').reset_index().idx.values
    neg_idx = new_cat.loc[new_cat[feat] == 0].sample(1000).rename_axis('idx').reset_index().idx.values
    y_score = NN_score(data_embeds, data_embeds[pos_idx[:,]].mean(axis=0), data_embeds[neg_idx[:,]].mean(axis=0))
    new_cat[f'{feat}_score'] = y_score
    new_cat, stats = score_to_pred(new_cat, alpha, labels=[feat, f'{feat}_score', f'{feat}_pred'])
    return new_cat, stats


def tsnePlot(cat, feat, data_embeds, trials=200, title='discrimination_tsne.png'):
    tsne = TSNE()

    pos_cat = cat.loc[cat[feat] == 1].sample(trials)
    neg_cat = cat.loc[cat[feat] == 0].sample(trials)
    src_cat = pd.concat([pos_cat, neg_cat])

    for row, event in src_cat.iterrows():
        lbl = src_cat[feat]
        emb = np.take(data_embeds, src_cat.index, axis=0)

    # Embed the images using the network
    tsne = tsne.fit_transform(emb)

    fig, axes = plt.subplots(figsize=(5,4))
    scatter(tsne, lbl, ax=axes, mark_sz=10, annotate=True)

    fig.tight_layout(pad=0.0, w_pad=0, h_pad=0)
    fig.subplots_adjust(top=.75)

    plt.savefig(os.path.join('images/', title), transparent=False, dpi=300, bbox_inches='tight')


def histPlot(cat, feat, data_embeds, trials=1000, title='discrimination_histograms.png'):
    fig, axes = plt.subplots(figsize=(5,4))

    out = axes.hist(get_random_pos_pairs(data_embeds, cat, feat, trials), bins=np.linspace(.01,2,100), label='Matched Pairs', alpha=.5)
    out = axes.hist(get_random_neg_pairs(data_embeds, cat, feat, trials), bins=np.linspace(.01,2,100), label='Unmatched Pairs', alpha=.5)
    axes.legend()
    axes.set_xlabel('Distance')
    axes.set_ylabel('Count')

    fig.tight_layout(pad=0.0, w_pad=0, h_pad=0)
    fig.savefig(os.path.join('images/', title), transparent=False, dpi=300, bbox_inches='tight')


def NN_classifier(x, pos_ex, neg_ex):
    pos_dists = np.linalg.norm(x - pos_ex, axis=1)
    neg_dists = np.linalg.norm(x - neg_ex, axis=1)
    return neg_dists/pos_dists > 1.05


def NN_score(x, pos_ex, neg_ex):
    pos_dists = np.linalg.norm(x - pos_ex, axis=1)
    neg_dists = np.linalg.norm(x - neg_ex, axis=1)
    return neg_dists/(pos_dists+1e-15)


def oneshotPlot(cat, feat, data_embeds, samps_list=[1,3,10,50], trials=1000, title='oneshot_ROCs.png', ref_curves=None):
    roc_auc = np.zeros((len(samps_list),trials))
    fig, ax = plt.subplots(1,len(samps_list), figsize=(3*len(samps_list),3), sharey=True)

    if len(samps_list) == 1:
        my_ax = ax
        ax = []
        ax.append(my_ax)

    for j, samps in enumerate(samps_list):

        if len(samps_list) > 1:
            ax[j].set_title(f'{samps} Known Examples')
            if samps == 1:
                ax[j].set_title(f'{samps} Known Example')

        ax[j].set_xlabel('FPR')

        for i in range(trials):
            pos_idx = cat.loc[cat[feat] == 1].sample(samps).rename_axis('idx').reset_index().idx.values
            neg_idx = cat.loc[cat[feat] == 0].sample(1000).rename_axis('idx').reset_index().idx.values

            y_score = NN_score(data_embeds, data_embeds[pos_idx[:,]].mean(axis=0), data_embeds[neg_idx[:,]].mean(axis=0))
            y_true = cat[feat].values == 1

            fpr, tpr, th = roc_curve(y_true, y_score)
            roc_auc[j, i] = auc(fpr, tpr)

            ax[j].plot(fpr, tpr, color='darkorange', alpha=5/trials, lw=2)

        if ref_curves is not None:
            ls = [':', '--', '-']
            for i, key in enumerate(ref_curves.keys()):
                fpr, tpr = ref_curves[key]
                ax[j].plot(fpr, tpr, color='black', lw=2, ls=ls[i], label=key)

            ax[j].plot(fpr, tpr, color='darkorange', lw=1, label='Proposed Discriminator')
            ax[j].legend(loc='center right')
        ax[j].grid(linestyle='-', linewidth='0.5')

        if len(samps_list) > 1:
            textstr = f'$AUC_\mu$: {np.mean(roc_auc[j]*100):0.1f}%\n$AUC_\sigma$: {np.var(roc_auc[j]):0.0E}'
            ax[j].text(0.6, 0.02, textstr, fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


    ax[0].set_ylabel('TPR')

    fig.tight_layout(pad=0.0, w_pad=0, h_pad=0)
    fig.subplots_adjust(top=.80)

    fig.savefig(os.path.join('images/', title), transparent=False, dpi=300, bbox_inches='tight')


def plot_confusion(cm, class_names, title='', ax=None, save_filename=None):
    ax = ax or plt.gca()
    ax.cla()

    cm = np.array([[2093, 758], [750, 32093]])
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, norm=LogNorm(vmin=100, vmax=10000))

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           xlabel='Predicted label', ylabel='True Label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    if save_filename is not None:
        plt.savefig(save_filename, transparent=False, dpi=300, bbox_inches='tight')


def plot_cat_map(df, img_name='', explosions=False, novel_sta=False, ax=None, extents=None):
    ax1 = ax or plt.gca(projection=ccrs.PlateCarree())
    ax1.cla()
    # Set map extents.
    if extents is None:
        llcrnrlon = np.min(np.append(df.LON_1, df.LON)) - 1
        llcrnrlat = np.min(np.append(df.LAT_1, df.LAT)) - 2
        urcrnrlon = np.max(np.append(df.LON_1, df.LON)) + 2
        urcrnrlat = np.max(np.append(df.LAT_1, df.LAT))
        extents = [llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat]
        print(extents)
    ax1.set_extent(extents, ccrs.PlateCarree())

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    ax1.add_image(cimgt.Stamen('terrain-background'), 6)
    ax1.add_feature(states_provinces, edgecolor='gray')
    ax1.coastlines('110m')
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.RIVERS)
    ax1.add_feature(cfeature.LAKES)

    df_events = df.drop_duplicates(subset=['EVENTID'], keep='first', inplace=False)
    if not explosions:
        # Plot events.
        lons = df_events.LON_1.values
        lats = df_events.LAT_1.values
        ax1.plot(lons, lats, '.', markersize=3, alpha=1, color='#ff6666', label='events', markerfacecolor='none')

    else:
        # Plot earthquakes.
        df_eq = df_events.loc[df_events.DEPTH != 0]
        lons = df_eq.LON_1.values
        lats = df_eq.LAT_1.values
        ax1.plot(lons, lats, '.', markersize=1.5, alpha=1, color='#ff6666', label='earthquakes', markerfacecolor='none')

        # Plot explosions.
        df_ex = df_events.loc[df_events.DEPTH == 0]
        lons = df_ex.LON_1.values
        lats = df_ex.LAT_1.values
        ax1.plot(lons, lats, '.', markersize=3, alpha=1, color='blue', label='explosions', markerfacecolor='none')

        # Plot stations.
    df_sta = df.sort_values('TIME').drop_duplicates(subset=['STA'], keep='first', inplace=False)
    if not novel_sta:
        lons = df_sta.LON.values
        lats = df_sta.LAT.values
        ax1.plot(lons, lats, 'x', markersize=7, alpha=1, color='black', label='stations', markerfacecolor='none')
        novel_events = None
    else:
        # Plot old stations.
        df_old = df_sta.loc[df_sta.novel_sta == 0]
        lons = df_old.LON.values
        lats = df_old.LAT.values
        ax1.plot(lons, lats, 'x', markersize=7, alpha=1, color='black', label='stations', markerfacecolor='none')

        # Plot new stations.
        df_new = df_sta.loc[df_sta.novel_sta == 1]
        lons = df_new.LON.values
        lats = df_new.LAT.values
        ax1.plot(lons, lats, 'o', markersize=7, alpha=1, color='#006475', label='novel stations', markerfacecolor='none')

        # Plot new region.
        lons = [-107.2, -107.2, -105.8, -105.8]
        lats = [45, 46.2, 46.2, 45]
        ax1.add_geometries([LinearRing(list(zip(lons, lats)))], ccrs.PlateCarree(), facecolor='#5ee8ff', edgecolor='#006475',
                           linewidth=2)
        novel_events = matplotlib.patches.Patch(edgecolor='#006475', facecolor='#5ee8ff', label='novel location', linewidth=2)

    handles, labels = ax1.get_legend_handles_labels()
    if novel_sta:
        handles.append(novel_events)

    ax1.legend(handles=handles, fontsize=10, loc='lower right')

    ax1.set_title(img_name)


def get_stream_from_X(idx, df, arr, pre_trim=30, post_trim=150, return_cat=False, conn=None):
    trace_list = []
    event = df.iloc[idx]
    sta = event.STA
    time = event.TIME

    st_time = obspy.UTCDateTime(time) - pre_trim
    en_time = obspy.UTCDateTime(time) + post_trim

    stats = {'sampling_rate': 40.0,
             'station': sta,
             'starttime': st_time,
             'endtime': en_time}

    for c, chan in enumerate(['BHZ', 'BHN', 'BHE']):
        stats['channel'] = chan
        trace_list.append(obspy.core.trace.Trace(arr[idx, :, c], stats))

    stream = obspy.core.stream.Stream(trace_list)

    return stream


def get_stream_from_ISC(idx, df, c, pre_trim=30, post_trim=150, return_cat=False):
    event = df.iloc[idx]
    sta = event.STA
    time = event.TIME
    cha = "BH?"
    net = "*"
    loc = ""

    st_time = obspy.UTCDateTime(time) - pre_trim
    en_time = obspy.UTCDateTime(time) + post_trim

    stream = c.get_waveforms(network=net, location=loc,
                             station=sta, channel=cha,
                             starttime=st_time, endtime=en_time)

    return stream


def plot_stream(stream):
    stream.detrend('constant')
    stream.normalize()
    stream.filter("bandpass", freqmin=1, freqmax=10)

    # stream.taper(.05)
    fig, ax = plt.subplots(3, 1, figsize=(15, 5))
    for i, comp in enumerate(['Z', 'N', 'E']):
        my_s = stream.select(component=comp)[0]
        ax[i].plot(my_s.times("matplotlib"), my_s.data, "b-")
        ax[i].xaxis_date()
        fig.autofmt_xdate()


def test_dataset_waveforms(cat, dat):
    idx = cat.sample().index[0]
    plot_stream(get_stream_from_X(idx, cat, dat))
    plot_stream(get_stream_from_ISC(idx, cat, Client('iris')))


def scatter(x, labels, subtitle=None, ax=None, annotate=True, mk_sh=['o'], mk_c=sns.color_palette('bright'), mk_sz=20):
    ax = ax or plt.gca()
    ax.cla()

    classes = list(set(labels))
    colors = np.array(mk_c * 1000)[:len(classes)]
    markers = np.array(mk_sh * 1000)[:len(classes)]

    color_dict = dict(zip(classes, colors))
    marker_dict = dict(zip(classes, markers))

    df = pd.DataFrame(columns=['x', 'y', 'label', 'marker', 'color'])
    df.x = x[:, 0]
    df.y = x[:, 1]
    df.label = labels
    df.color = df.label.map(color_dict)
    df.marker = df.label.map(marker_dict)

    grouped = df.groupby('label')

    for key in grouped.groups.keys():

        my_group = grouped.get_group(key)
        ax.scatter(my_group.x, my_group.y, marker=marker_dict[key], label=key, s=mk_sz, c=np.expand_dims(color_dict[key], axis=0))

    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.axis('tight')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # We add the labels for each digit.
    if annotate:
        txts = []
        for i in np.unique(labels):
            # Position of each label.
            xtext, ytext = np.median(x[labels == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(int(i)), fontsize=12)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    else:
        pass
        #plt.legend()

    if subtitle != None:
        ax.set_xlabel(subtitle, fontsize=12)


def score_to_pred(cat, alpha=None, print_stats=True, labels=['Y', 'Y_score', 'Y_pred']):
    ytrue, yscore, ypred = labels

    # print(len(cat.loc[cat.Y == 1]))

    cat = cat.copy(deep=True)
    fpr, tpr, th = roc_curve(cat[ytrue], cat[yscore])

    if alpha is None:
        idx = np.argmax(tpr * (1 - fpr))
        th = th[idx]
        alpha = fpr[idx]
    else:
        th = th[np.argmax(fpr > alpha)]

    cat[ypred] = False
    cat.loc[cat[yscore] > th, ypred] = True
    cat[ytrue] = cat[ytrue].astype('bool')

    z = classification_report(cat[ytrue], cat[ypred], output_dict=True)
    recall = z['True']['recall']
    precis = z['True']['precision']
    accury = (1 - len(cat.loc[cat[ytrue] ^ cat[ypred]]) / len(cat))

    cat['COND'] = ''
                        

    cat.loc[cat[ytrue] > cat[ypred], 'COND'] = 'FN'
    cat.loc[cat[ytrue] < cat[ypred], 'COND'] = 'FP'
    cat.loc[cat[ytrue] & cat[ypred], 'COND'] = 'TP'
    cat.loc[~(cat[ytrue] | cat[ypred]), 'COND'] = 'TN'

    if print_stats:

        # print('threshold:', th)
                        
        print('total pairs:         ', len(cat))
        print('type-I error rate:   ', alpha)
        print('-------------')
        # print('misclassified pairs: ', len(a.loc[a.Y ^ a.Y_PRED]))
        print('accuracy:            ',
              str(np.round(100 * accury, 1)) + '%')
        print('recall:              ', str(np.round(100 * recall, 1)) + '%')
        print('precision:           ', str(np.round(100 * precis, 1)) + '%')
        print('-------------')
        print(cat.COND.value_counts())

    cm = np.array([[len(cat.loc[cat.COND == 'TP']), len(cat.loc[cat.COND == 'FN'])],
                   [len(cat.loc[cat.COND == 'FP']), len(cat.loc[cat.COND == 'TN'])]])
                        
    stats = [precis, recall, accury, cm]
    return cat, stats


def assoc_test_gen(cat, emb, samps=100):
    assoc_cat = pd.DataFrame(columns=['idx1', 'idx2', 'Y'])

    for i in range(samps // 2):
        try:
            event = cat.sample()
            idx = event.index[0]
            event = cat.iloc[idx]
            idxp = cat.loc[(cat.EVENTID == event.EVENTID) & ~(cat.STA == event.STA)].sample().index[0]
            idxn = cat.loc[(cat.EVENTID != event.EVENTID)].sample().index[0]
        except:
            pass
        assoc_cat = assoc_cat.append(pd.DataFrame([[idx, idxp, True]], columns=['idx1', 'idx2', 'Y']))
        assoc_cat = assoc_cat.append(pd.DataFrame([[idx, idxn, False]], columns=['idx1', 'idx2', 'Y']))
    assoc_cat.Y = assoc_cat.Y.astype(bool)
    assoc_cat = assoc_cat.reset_index(drop=True)

    for row, event in assoc_cat.iterrows():
        assoc_cat.loc[row, 'Y_score'] = 1 / np.linalg.norm(emb[event.idx1] - emb[event.idx2])
    assoc_cat.Y_score = np.nan_to_num(assoc_cat.Y_score.values)
    assoc_cat, assoc_results = score_to_pred(assoc_cat)

    d = []
    for row, pair in assoc_cat.iterrows():
        loc1 = cat.loc[pair.idx1][['LON', 'LAT']].values
        loc2 = cat.loc[pair.idx2][['LON', 'LAT']].values
        d.append(haversine(loc1, loc2))
    assoc_cat['STA_DIST'] = d


    return assoc_cat, assoc_results


def association_residualizer(ass_cat, event_cat, feat_list):
    ass_cat = ass_cat.copy(deep=True)
    for i in [1, 2]:
        ass_cat[f'idx{i}'] = ass_cat[f'idx{i}'].astype('int64')
        ass_cat = ass_cat.join(event_cat[feat_list], on=f'idx{i}').rename(index=str, columns=dict(zip(feat_list, [f'{f}{i}' for f in feat_list])))

    for feat in feat_list:

        if ass_cat[f'{feat}1'].dtype == 'float':
            ass_cat[f'{feat}max'] = np.max(np.vstack([ass_cat[f'{feat}1'].values, ass_cat[f'{feat}2'].values]), axis=0)
            ass_cat[f'{feat}min'] = np.min(np.vstack([ass_cat[f'{feat}1'].values, ass_cat[f'{feat}2'].values]), axis=0)
        else:
            ass_cat[feat] = ass_cat[f'{feat}1'] | ass_cat[f'{feat}2']

    return ass_cat


def residual_by_dist(assoc_cat, samp_size=350, dist_interval=250, dist_bins=9):
    print('  Distance\t  P\t  R\t Acc')
    for i in range(dist_bins):
        dist_min = i*dist_interval
        dist_max = (i+1)*dist_interval
        my_assoc_cat = pd.concat([assoc_cat.loc[(assoc_cat.Y == True) &
                                                (assoc_cat.STA_DIST > dist_min) &
                                                (assoc_cat.STA_DIST < dist_max)].sample(samp_size),
                                  assoc_cat.loc[(assoc_cat.Y == False)].sample(samp_size)])

        _, stats = score_to_pred(my_assoc_cat, print_stats=False)
        precis, recall, accury = stats[:3]
        print(f'{dist_min:04d}-{dist_max:04d} km\t{precis:.3f}\t{recall:.3f}\t{accury:.3f}')


def get_emb(dataset_name, model_id, use_provided_emb=False, data_folder='data/datasets/'):

    pdict = {}
    pdict['iniW'] = model_id
    model, model_name = load_custom_model(pdict)
    pdict = name2param(model_name)
    model, _ = compile_custom_model(model, pdict)

    st = (30 - pdict['pre']) * 40
    en = (30 + pdict['post']) * 40

    if use_provided_emb:
        print('\nloading embeddings...', end=' ', flush=True)
        emb = np.load(os.path.join(data_folder, f'EMB_{dataset_name}_{model_id}.npy'))
        print('generating placeholder waveforms...', end=' ', flush=True)
        dat = np.zeros((len(emb), en - st, 3))
        print('complete')

    else:
        print('\nloading waveforms...', end=' ', flush=True)
        dat = np.load(os.path.join(data_folder, f'X_{dataset_name}.npy'))[:, st:en, :]

        if pdict['dtrd']:
            print('detrending...', end=' ', flush=True)
            dat = detrend(dat)
        if pdict['norm']:
            print('normalizing...', end=' ', flush=True)
            dat = normalize(dat)

        print('transforming...', end=' ', flush=True)

        t1_start = perf_counter()
        emb = model.predict(dat)
        t1_stop = perf_counter()

        print('complete')

        print(model.summary())

        print("\n\nTotal Number of Events:", len(dat))
        print("Elapsed time in seconds:", t1_stop - t1_start)
        print("Time to process 1 Event:", (t1_stop - t1_start) / len(dat))

        np.save(os.path.join(data_folder, f'EMB_{dataset_name}_{model_id}.npy'), emb)

    return dat, emb


def get_random_neg_pairs(x, cat, y_label, num_pairs):
    dists = np.zeros((num_pairs,))
    for i in range(0, num_pairs):
        ind_1 = cat.sample().index.values[0]
        neg_label = cat.loc[ind_1][y_label]
        ind_2 = cat.loc[cat[y_label] != neg_label].sample().index.values[0]
        sample1 = x[ind_1]
        sample2 = x[ind_2]

        dists[i] = np.linalg.norm(sample1 - sample2)
    return dists


def get_random_pos_pairs(x, cat, y_label, num_pairs):
    dists = np.zeros((num_pairs,))
    for i in range(0, num_pairs):
        ind_1 = cat.sample().index.values[0]
        pos_label = cat.loc[ind_1][y_label]
        ind_2 = cat.loc[cat[y_label] == pos_label].sample().index.values[0]
        sample1 = x[ind_1]
        sample2 = x[ind_2]

        dists[i] = np.linalg.norm(sample1 - sample2)
    return dists


def residual_by_feat(cat, feat):
    print(feat)
    tot_count = cat[feat].value_counts()
    bad_count = cat.loc[(cat.Y ^ cat.Y_pred)][feat].value_counts()

    Residuals = pd.DataFrame(index=tot_count.keys())
    Residuals['TOT'] = tot_count

    tmp = pd.DataFrame(index=bad_count.keys())
    tmp['ERR'] = bad_count

    # Residuals_by_STA.update(tmp)

    Residuals['ERR'] = 0
    Residuals.update(tmp)
    Residuals['ERR'] = Residuals['ERR'].astype(int)

    Residuals['ACC'] = np.round(1 - Residuals['ERR'] / Residuals['TOT'], 2)

    return Residuals.sort_index()


def haversine(point1, point2, unit='km'):
    """ Calculate the great-circle distance between two points on the Earth surface.

    :input: two 2-tuples, containing the latitude and longitude of each point
    in decimal degrees.

    Keyword arguments:
    unit -- a string containing the initials of a unit of measurement (i.e. miles = mi)
            default 'km' (kilometers).

    Example: haversine((45.7597, 4.8422), (48.8567, 2.3508))

    :output: Returns the distance between the two points.

    The default returned unit is kilometers. The default unit can be changed by
    setting the unit parameter to a string containing the initials of the desired unit.
    Other available units are miles (mi), nautic miles (nmi), meters (m),
    feets (ft) and inches (in).

    """
    # mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
    AVG_EARTH_RADIUS_KM = 6371.0088

    # Units values taken from http://www.unitconversion.org/unit_converter/length.html
    conversions = {'km': 1,
                   'm': 1000,
                   'mi': 0.621371192,
                   'nmi': 0.539956803,
                   'ft': 3280.839895013,
                   'in': 39370.078740158}

    # get earth radius in required units
    avg_earth_radius = AVG_EARTH_RADIUS_KM * conversions[unit]

    # unpack latitude/longitude
    lat1, lng1 = point1
    lat2, lng2 = point2

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    # calculate haversine
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2

    return 2 * avg_earth_radius * asin(sqrt(d))



def dataset_to_featureset(dataset_name, overwrite=False, n_filt=9, n_win=4, sta_time=2, arr_time=30, samp_rate=40, filt_order=2, data_folder = 'data/datasets/'):
    SPdiff_model = load_model('models/SPdiff_model.h5')

    print('building catalog...', end=' ', flush=True)
    cat = read_cat(os.path.join(data_folder, f'Y_{dataset_name}.csv'))
    cat['SPdiff'] = SPdiff_model.predict(cat[['DIST', 'DEPTH']].values)
    print('complete')

    print('building features...', end=' ', flush=True)
    featureset_path = os.path.join(data_folder, f'F_{dataset_name}.npy')

    if (not os.path.exists(featureset_path)) | overwrite:
        dat = np.load(os.path.join(data_folder, f'X_{dataset_name}.npy'))
        if len(dat.shape) > 2:
            dat = dat[:, :, 0]
        X = np.zeros((len(dat), n_filt * n_win))
        for row, event in cat.iterrows():
            X[row] = extract_features(dat[row], event.SPdiff, n_filt, n_win, sta_time, arr_time, samp_rate, filt_order)
        np.save(featureset_path, X)
    else:
        X = np.load(featureset_path)
    print('complete')

    return cat, X


def prune_featureset(cat, X):
    print('pruning features...', end=' ', flush=True)
    msk = ~np.isnan(X).any(axis=1)
    cat = cat[msk]
    X = np.take(X, cat.index, axis=0)
    cat.reset_index(drop=True, inplace=True)
    print('complete')
    return cat, X


def extract_features(x_raw, s_p_time, n_filt=9, n_win=4, sta_time=2, arr_time=30, samp_rate=40, filt_order=2,
                     plot=False):
    ''' This function takes in a raw seismogram and returns a spectrogram.
        The spectrogram windowing is based on the theoretical S-P time difference.

        x_raw:           RAW SEISMOGRAM
        samp_rate:       SAMPLE RATE OF SEISMOGRAM
        arr_time:        ARRIVAL TIME OF EVENT WITHIN SEISMOGRAM (IN SECONDS)
        sta_time:        LENGTH OF THE SHORT TERM AVERAGING WINDOW (IN SECONDS)
        s_p_time:        THEORETICAL S-P ARRIVAL TIME DIFFERENCE (IN SECONDS)
        n_win:           NUMBER OF WINDOWS IN SPECTROGRAM
        n_filt:          NUMBER OF FILTERS IN SPECTROGRAM
        filt_order:      ORDER OF FILTERS IN SPECTROGRAM
        plot:            BOOLEAN FOR PLOTTING

        The default phase windows contain the first P wave, P coda,
        the first S wave and S coda, all with the same length:
        half of the theoretical S–P arrival time difference.
        Two noise windows are also included at the start and end.
        The length of the noise windows is equal to the
        theoretical S–P arrival time difference.
    '''

    sta_samps = int(sta_time * samp_rate)
    arr_samps = int(arr_time * samp_rate)
    win_samps = int(s_p_time * samp_rate)

    # Built lists of time and freq bins for the spectrogram
    time_bins = list(map(tuple, (win_samps * np.array([(2 + t, 3 + t) for t in range(n_win)]) / 2).astype(int)))
    freq_bins = [(1, 3)] + [(f * 2, f * 2 + 3) for f in range(1, n_filt)]

    # Extract the spectrogram features
    x_sta = []
    feat = np.zeros((n_filt, n_win))
    for filt, (freqmin, freqmax) in enumerate(reversed(freq_bins)):

        # Perform the filtering
        x_filt = filter.bandpass(x_raw, freqmin, freqmax, samp_rate, filt_order, zerophase=True)

        # Compute the STA characteristic function
        x_sta.append(np.convolve(x_filt ** 2, np.ones((sta_samps,)) / sta_samps, mode='valid')[
                     int(arr_samps - win_samps):int(arr_samps + 3 * win_samps)])

        # Create the normalized averages for each phase window
        for win, (st, en) in enumerate(time_bins):
            feat[filt, win] = np.mean(x_sta[filt][st:en]) / np.mean(x_sta[filt])

    # Plot the filtered waveforms
    if plot:
        fig, ax = plt.subplots(n_filt, 1, sharey='col', sharex=True)
        t = np.array(sorted(set([i for sl in time_bins for i in sl])))
        for filt in range(n_filt):
            ax[filt].plot(x_sta[filt])
            ax[filt].set_xticks(t)
            ax[filt].set_xticklabels(np.round(t / samp_rate - t[0] / samp_rate, 2))
            ax[filt].set_yticklabels([])

            # Plot vertical lines defining the phase windows
            for st, en in time_bins:
                ax[filt].axvline(x=st, c='g', ls='--', lw=1)
                ax[filt].axvline(x=en, c='g', ls='--', lw=1)

        # print the numerical spectrogram
        print(feat)

    return feat.flatten()
