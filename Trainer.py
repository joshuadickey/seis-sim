import numpy as np
import os
from time import localtime, strftime
from UTILS import generator_hard, read_cat, detrend, normalize, \
    load_custom_model, get_callbacks, compile_custom_model, generator_hard2


# OS Parameters
model_folder = 'models'
log_folder = 'logs'
data_folder = 'data/datasets/'

GPU = '0'
print('GPU:', GPU)
os.environ["CUDA_VISIBLE_DEVICES"]=GPU


##################################################################################
#                                    LOAD MODEL                                  #
##################################################################################

# Model Parameters
pdict = {}
pdict['f']      = 50
pdict['k']      = 15
pdict['d']      = [1, 2, 4, 8]
pdict['s']      = 2
pdict['n']      = 1
pdict['od']     = 32

# Dataset Parameters
pdict['trn']    = 'USArray_07-13_conus_holdout'
pdict['val']    = 'USArray_14_conus'
pdict['dtrd']   = 1
pdict['norm']   = 1
pdict['pre']    = 30
pdict['post']   = 150

# Training Parameters
pdict['sh'] = 0.05
pdict['numP']   = 3
pdict['numK']   = 3
pdict['m']      = 0.1
pdict['a']      = 0.001
pdict['lr']     = 0.005
pdict['pat']    = 7
#pdict['iniW']   = '19-07-25-23-22-11'
pdict['time']   = strftime("%y-%m-%d-%H-%M-%S", localtime())


model, _ = load_custom_model(pdict)
print(model.summary())

##################################################################################
#                                   LOAD DATASET                                 #
##################################################################################

print('loading dataset...', end=' ', flush=True)
st = (30 - pdict['pre']) * 40
en = (30 + pdict['post']) * 40
Y_trn = read_cat(os.path.join(data_folder, f"Y_{pdict['trn']}.csv"))
X_trn = np.load(os.path.join(data_folder, f"X_{pdict['trn']}.npy"))[:, st:en, :]
Y_val = read_cat(os.path.join(data_folder, f"Y_{pdict['val']}.csv"))
X_val = np.load(os.path.join(data_folder, f"X_{pdict['val']}.npy"))[:, st:en, :]
if pdict['dtrd']:
    X_trn = detrend(X_trn)
    X_val = detrend(X_val)
if pdict['norm']:
    X_trn = normalize(X_trn)
    X_val = normalize(X_val)
print('complete')


##################################################################################
#                                      TRAIN                                     #
##################################################################################

for pdict['numP'] in range(2, 10):
    pdict['numK'] = pdict['numP']
    pdict['time'] = strftime("%y-%m-%d-%H-%M-%S", localtime())


    train_gen = generator_hard(X_trn, Y_trn, n_P=pdict['numP'], n_K=pdict['numK'], sh_pct=pdict['sh'])
    val_gen = generator_hard(X_val, Y_val, n_P=pdict['numP'], n_K=pdict['numK'], sh_pct=pdict['sh'])

    model, model_name = compile_custom_model(model, pdict)
    print(model_name)


    t_step = len(Y_trn) // (pdict['numP'] * pdict['numK']) + 1
    v_step = len(Y_val) // (pdict['numP'] * pdict['numK']) + 1

    history = model.fit_generator(train_gen, steps_per_epoch=t_step, epochs=5000,
                                  validation_data=val_gen, validation_steps=v_step,
                                  callbacks=get_callbacks(model_name, model_folder, log_folder),
                                  max_queue_size=10)
    pdict['iniW'] = pdict['time']
    model, model_name = load_custom_model(pdict)