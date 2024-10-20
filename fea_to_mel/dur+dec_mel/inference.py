import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ['XLA_FLAGS'] = '--xla_disable_constant_folding'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, History, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Conv1D, GRU, Bidirectional, Add, concatenate, Embedding, BatchNormalization, Activation, Dropout, Lambda, Multiply, LSTM, Concatenate, MaxPool1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from itertools import groupby
import random
from scipy.io import wavfile
from utils import kaldi_pad, preemphasis
# from dataloader import dg_kaldi_tts
import kaldiio
from scipy.io.wavfile import write
import sys
from swan import MHAttn
import time
import torch
import mplcursors  # For interactive cursor
from denoiser import Denoiser
import matplotlib.pyplot as plt

tf.profiler.experimental.start('logdir')  # Start profiler







SEED = 2
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


MAX_WAV_VALUE = 32768.0


sp_file_name = 'speakers.txt'
speakers = {}
with open(sp_file_name) as f:
    for line in f:
        (key, val) = line.strip().split('|')
        speakers[key] = int(val)







def minmax_norm(S, min_val, max_val):
    return np.clip((S - min_val) /(max_val - min_val), 0., 1.)


def inv_minmax_norm(x, min_val, max_val):
    return np.clip(x,0,1) * (max_val - min_val)+min_val  




def text_encoder(inputs, inp_dim):
    text_lang_emb = inputs
    conv_bank=[]
    for i in range(1,9):
        x = Conv1D(filters=128, kernel_size=i, activation='relu', padding='same')(text_lang_emb)
        x = BatchNormalization()(x)
        conv_bank.append(x)


    
  
    

    x = concatenate(conv_bank, axis=-1)

   
    x = MaxPool1D(pool_size=2, strides=1, padding='same')(x) 

    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv1D(filters=inp_dim, kernel_size=3, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
   
    x = x + inputs

    x = Bidirectional(GRU(64, return_sequences=True))(x)
    
    return x


def duration_predictor(inputs,n_layers=4, kernel_size=3, dropout_rate=0.1):
    x=inputs
    for _ in range(n_layers):
        x = Conv1D(filters=128, kernel_size=kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)  
        x = Dropout(dropout_rate)(x)


    x = Dense(96,activation='relu')(x) 
    est_dur=Dense(64, activation='relu')(x)
    est_dur=Dense(64, activation='relu')(est_dur)
    est_dur=Dense(1, activation='relu')(est_dur)
    return est_dur



def new_acoustic_decoder(inputs, n_blocks=1, n_heads=4, head_size=64, context=10, inter_dim=128, out_dim=128):
    x = Dense(inter_dim, activation='relu')(inputs)
    for i in range(n_blocks):
        cx =  MHAttn(n_heads, head_size, context)(x)
        x = BatchNormalization()(x+cx)
        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)
        x = BatchNormalization()(x+cx)


        xe = Dense(inter_dim, activation='relu')(x)
        xe = Dense(inter_dim, activation='relu')(xe)
        x=tf.keras.layers.BatchNormalization()(xe+x)                                    
    
    x = Dense(out_dim, activation='relu')(x)
    return x



def acoustic_decoder(inputs):
   
    conv_bank=[]
    for i in range(1,11):
        x = Conv1D(filters=128, kernel_size=i+1, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        conv_bank.append(x)

    x = concatenate(conv_bank, axis=-1)
    x = MaxPool1D(pool_size=2, strides=1, padding='same')(x) 
  
    
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    x = Bidirectional(GRU(64, return_sequences=True))(x)
    return x












spkr_embedding = Embedding(30, 16,
        embeddings_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.5), 
        trainable=False, mask_zero=False) 




phn_lab=tf.keras.Input(shape=(None, 768), dtype=tf.float32)
phn_mask=tf.keras.Input(shape=(None,1), dtype=tf.float32)
phn_repeats=tf.keras.Input(shape=(None,2), dtype=tf.int32)

len_mask=tf.keras.Input(shape=(None,1), dtype=tf.float32)
spkr_lab_enc=tf.keras.Input(shape=(None, ), dtype=tf.int32)


                       
encoder_output = text_encoder(phn_lab, inp_dim=768)

spkr_embeddings = spkr_embedding(spkr_lab_enc) 





text_spkr_emb = Concatenate(axis=-1)([encoder_output, spkr_embeddings])

#Duration Estimation
est_dur=Dense(64, activation='relu')(text_spkr_emb)
est_dur=Dense(64, activation='relu')(est_dur)
est_dur=Dense(1, activation='relu')(est_dur)
est_dur = Multiply(name='dur')([est_dur, phn_mask])

# est_dur=duration_predictor(text_spkr_emb)
# est_dur = Multiply(name='dur')([est_dur, phn_mask])











x = Lambda(lambda x: tf.gather_nd(x[0],x[1]),output_shape=(None,144))([text_spkr_emb, phn_repeats])










upsampled_enc = Multiply()([x, len_mask])












x = acoustic_decoder(upsampled_enc)



x = new_acoustic_decoder(x, n_blocks=3, n_heads=4, head_size=64, context=10, inter_dim=128, out_dim=128)

est_mel = Dense(80, activation='relu')(x)
mel_gate = Dense(80, activation='sigmoid')(x)
est_mel = Multiply()([est_mel, mel_gate])
est_mel = Multiply(name='mel')([est_mel, len_mask])





model = tf.keras.Model(inputs=[phn_lab, phn_mask, phn_repeats , len_mask, spkr_lab_enc], 
        outputs=[est_dur, est_mel])


model.load_weights('/Users/samarasimhareddygujjula/Desktop/GameChanger/fea_to_mel/dur+dec_mel/chekpoints/weights-0228.keras')

dur_model = tf.keras.models.Model(inputs=[model.input[0], model.input[1],model.input[4]], outputs=model.output[0])




phn_lab=torch.load('/Users/samarasimhareddygujjula/Desktop/GameChanger/train_bengalimale_04138_ctc_test.pt', map_location=torch.device('cpu'))

# phn_lab=phn_lab.cpu()




phn_mask=np.ones(len(phn_lab))


predicted_mel_list=[]

for i in range(1,3):


    speaker=f'speaker{i}'

    spkr_id = speakers[speaker]
    spkr_lab = np.repeat(spkr_id, len(phn_lab))

    print(speaker)




    est_dur = dur_model.predict([tf.expand_dims(phn_lab, axis=0), tf.expand_dims(phn_mask, axis=0), tf.expand_dims(spkr_lab, axis=0)])
    est_dur = est_dur[0,:,0]

    phn_freq=inv_minmax_norm(est_dur, min_val=1, max_val=50)
    phn_freq=np.round(phn_freq).astype('uint8')


    original_phn_freq=torch.load('/Users/samarasimhareddygujjula/Desktop/GameChanger/train_bengalimale_04138_ranges_test.pt')


    print(phn_freq,"predietc",sum(phn_freq),len(phn_freq))

    print(original_phn_freq,"original",sum(original_phn_freq),len(original_phn_freq))

    # Plotting the predicted and original phoneme frequencies
    plt.figure(figsize=(10, 6))

    # Plot predicted phoneme frequencies
    predicted_plot, = plt.plot(phn_freq, label="Predicted Phoneme Frequencies", marker='', color='blue', linestyle='-', linewidth=2)

    # Plot original phoneme frequencies
    original_plot, = plt.plot(original_phn_freq, label="Original Phoneme Frequencies", marker='', color='red', linestyle='--', linewidth=2)

    # Adding title and labels
    plt.title('Comparison of Predicted and Original Phoneme Frequencies')
    plt.xlabel('Phoneme Index')
    plt.ylabel('Frequency Count')
    plt.legend()

    # Show grid
    plt.grid(True)

    # Enable interactive cursor for both predicted and original plots
    mplcursors.cursor(predicted_plot).connect("add", lambda sel: sel.annotation.set_text(f"Predicted: {phn_freq[sel.index]}"))
    mplcursors.cursor(original_plot).connect("add", lambda sel: sel.annotation.set_text(f"Original: {original_phn_freq[sel.index]}"))

    # Save the plot into the "plots" folder
    output_path = os.path.join('/Users/samarasimhareddygujjula/Desktop/GameChanger/results/Durations', f'phoneme_frequencies_comparison_{speaker}.png')
    plt.savefig(output_path, format='png', dpi=300)

    # # Show the plot (interactive with hover)
    # plt.show()

    # Confirm where the file is saved
    print(f"Plot saved to: {output_path}")




    ###############  PRedicted  ############################################
    phn_repeats=np.repeat(np.arange(len(phn_freq)), phn_freq)
    phn_repeats=np.stack((np.zeros(phn_repeats.shape[0]), phn_repeats), axis=-1)
    len_mask = np.ones(len(phn_repeats))




    ##############Original #############################################################

    # phn_repeats_original=np.repeat(np.arange(len(original_phn_freq)), original_phn_freq)
    # phn_repeats_original=np.stack((np.zeros(phn_repeats_original.shape[0]), phn_repeats_original), axis=-1)
    # len_mask_original = np.ones(len(phn_repeats_original))



    ################################### Prediced Mel ##########################################################################
    est_dur,est_mel= model.predict(
            [tf.expand_dims(phn_lab, axis=0), tf.expand_dims(phn_mask, axis=0), 
                tf.expand_dims(phn_repeats, axis=0),  tf.expand_dims(len_mask, axis=0), tf.expand_dims(spkr_lab, axis=0)])


    fbank = torch.from_numpy(est_mel[0])
  
    torch.save(fbank,"pred_mel.pt")
    torch.save(fbank,f'/Users/samarasimhareddygujjula/Desktop/GameChanger/Vococers/hifi-gan/test_mel_files/pred_mel_{speaker}.pt')

    # sys.exit()




    #############################################################################################################################



    ################################### Original duration  Mel ##########################################################################


    # est_dur_orig,est_mel_orig= model.predict(
    #         [tf.expand_dims(phn_lab, axis=0), tf.expand_dims(phn_mask, axis=0), 
    #             tf.expand_dims(phn_repeats_original, axis=0),  tf.expand_dims(len_mask_original, axis=0), tf.expand_dims(spkr_lab, axis=0)])


    # fbank_orig = est_mel[0]   #Y[-2][0]  #est_mel[0]
    # fbank_orig = (fbank_orig*5)-5
    # inv_norm_fbanks_orig = np.power(10.0, fbank_orig)
   

    #############################################################################################################################


#     waveglow_path='/raid/ai23mtech02001/GameChanger/checkpoints/waveglow_checkpoint'
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     waveglow = torch.load(waveglow_path,map_location=device)['model']
#     waveglow = waveglow.remove_weightnorm(waveglow).to(device)
#     waveglow.eval()


#     sampling_rate = 22050
#     is_fp16 = False
#     sigma = 1.0
#     denoiser_strength = 0.0
#     if is_fp16:
#         from apex import amp
#         waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

#     if denoiser_strength > 0:
#         denoiser = Denoiser(waveglow).cuda()


#     mel = torch.Tensor(inv_norm_fbanks_orig.T)
#     #mel = torch.Tensor(est_mel[0].T)
#     mel = torch.log(mel)

   
#     predicted_mel_list.append(mel)
    

#     predicted_mel=mel

#     original_mel=torch.load('/raid/ai23mtech02001/GameChanger/Data/female_telugu_english_mels/train_telugufemale_00001_split_0.pt')
#     original_mel = (original_mel*5)-5
#     original_mel= np.power(10.0, original_mel)
#     original_mel = torch.log(original_mel)

#     mel1=original_mel
#     mel2=predicted_mel


#     # Plotting the two Mel spectrograms side by side
#     fig, axes = plt.subplots(2, 1, figsize=(12, 6))

#     # Plotting the first Mel spectrogram
#     axes[0].imshow(mel1.T, aspect='auto', origin='lower', cmap='inferno')
#     axes[0].set_title('Mel Spectrogram 1')
#     axes[0].set_xlabel('Time Frames')
#     axes[0].set_ylabel('Mel Bins')

#     # Plotting the second Mel spectrogram
#     axes[1].imshow(mel2, aspect='auto', origin='lower', cmap='inferno')
#     axes[1].set_title('Mel Spectrogram 2')
#     axes[1].set_xlabel('Time Frames')
#     axes[1].set_ylabel('Mel Bins')

#     # Save the plot as an image file (e.g., PNG format)
#     plt.tight_layout()
#     plt.savefig(f'/raid/ai23mtech02001/SLMS/Results_test/Fixed_embe_lr_4/Speaker2_Male/mel_spectrograms-1-{i}.png')  # Change filename and format as needed

#     mel = torch.autograd.Variable(mel)
#     mel = torch.unsqueeze(mel, 0)
#     mel=mel.to(device)
#     with torch.no_grad():
#         audio = waveglow.infer(mel, sigma=sigma)
#         audio = audio * MAX_WAV_VALUE
#     audio = audio.squeeze()
#     audio = audio.cpu().numpy()
#     audio = audio/max(abs(audio))*0.95*(2**15)        
#     audio = audio.astype('int16')
        
#     print("Audio: ",audio)
#     file_name = 'synth' 
#     audio_path = os.path.join(
#         "/raid/ai23mtech02001/SLMS/Results_test/Fixed_embe_lr_4/Speaker2_Male", f"{file_name}_mel_tap_5-{speaker}.wav")
#     write(audio_path, sampling_rate, audio)
#     print(audio_path)

# original_mel = original_mel

# # Simulate predicted Mels for 10 speakers (list of 10 arrays, each of shape [sequence_length, mel_bins])
# predicted_mels = predicted_mel_list

# # Create a plot with 11 rows (1 for original, 10 for predicted) and 1 column
# fig, axes = plt.subplots(11, 1, figsize=(10, 20))  # 11 rows, 1 column

# # Plot the original Mel at the top
# axes[0].imshow(original_mel.T, aspect='auto', origin='lower', cmap='inferno')
# axes[0].set_title('Original Mel Spectrogram')
# axes[0].set_xlabel('Time Frames')
# axes[0].set_ylabel('Mel Bins')

# # Plot the predicted Mels for 10 speakers below
# for i in range(10):
#     axes[i+1].imshow(predicted_mels[i], aspect='auto', origin='lower', cmap='inferno')
#     axes[i+1].set_title(f'Predicted Mel Spectrogram for Speaker {i+1}')
#     axes[i+1].set_xlabel('Time Frames')
#     axes[i+1].set_ylabel('Mel Bins')

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Save the plot as an image file
# plt.savefig('/raid/ai23mtech02001/SLMS/Results_test/Fixed_embe_lr_4/Speaker2_Male/mel_spectrograms_comparison.png')  # Change filename and format as needed

