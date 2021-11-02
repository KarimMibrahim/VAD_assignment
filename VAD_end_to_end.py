import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
from pyroomacoustics.denoise import apply_spectral_sub, apply_iterative_wiener
from scipy import signal as sps

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio 

from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score, classification_report, roc_auc_score
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm

FRAMES_3SEC = 92

# Load  audio file
# load metadata
# apply spatial processing (pass chosen pipeline)
# Defining a room through rt60 
def get_room(rt60_tgt, fs): 
    # Setting up room dims
    room_dim = [3, 3, 3]  # meters (Assuming a cubic room of 3x3x3 meters - some rt60 were too low for bigger)
    room_center = [dim / 2 for dim in room_dim]
    # put our michrophone in the center of the room
    R = np.array([[room_center[0] - (0.071/2), room_center[0] + (0.071/2)], 
                  [room_center[1], room_center[1]], 
                  [room_center[2], room_center[2]]])  # [[x], [y], [z]]
    # Setting up room based on rt60
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    room_bf = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    return room_bf, R, room_center

def enhance_signal(audio_file, metadata_file, beamformer = "mvdr", denoiser = "SS", 
                   speech_band_pass = True , save_dir = "/tmp/"):
    # Load files
    fs, signal = wavfile.read(audio_file)
    signal = pra.normalize(signal.astype(float))
    with open(meta_file) as json_file:
        meta_data = json.load(json_file)

    # Setting up room based on rt60
    rt60_tgt = float(meta_data['rt60'])
    room_bf, R, room_center = get_room(rt60_tgt, fs)

    # Load azimuth and elevation
    source_azimuth, source_elevation =  meta_data['source_doa']
    noise_azimuth, noise_elevation = meta_data['noise_doa']

    # Compute noise variance [Not sure about this bit], is SNR from the source speech?? 
    # Maybe better use the first 3 seconds for that? 
    SNR = meta_data['SNR']
    if (SNR != None):
        SNR = int(SNR)
        mono_signal = 0.5 * (signal[:,0] + signal[:,1])
        # compute Signal power [not correct if SNR is pure speech to noise instead of noisy speech to noise]
        P_s = sp.sum(mono_signal*mono_signal)/mono_signal.size 
        sigma2_n = P_s / 10**(SNR/10) # compute noise power
    else:
        sigma2_n = 5e-7 # set small value for noise power

    # Putting sources in room
    # Assume all sources are 1 meter away from the center (but if SNR is pure speech to noise, 
    # maybe we can set the distance for each source proportional to the SNR)
    r = 1 
    # Setup sources for speech and noise 
    speech_source = [r*np.cos(source_azimuth)*np.cos(source_elevation) + room_center[0],
                    r*np.sin(source_azimuth)*np.cos(source_elevation) + room_center[1],
                    r*np.sin(source_elevation) + room_center[2]] # [x,y,z]
    noise_source = [r*np.cos(noise_azimuth)*np.cos(noise_elevation) + room_center[0],
                    r*np.sin(noise_azimuth)*np.cos(noise_elevation) + room_center[1],
                    r*np.sin(noise_elevation) + room_center[2]] # [x,y,z]

    # Here I pass a dummy signal to the source just to simulate. We ignore this signal later
    room_bf.add_source(speech_source, delay=0., signal=signal.T[0,:])
    room_bf.add_source(noise_source, delay=0, signal=np.zeros_like(signal.T[0,:]))

    # define our beamformer
    Lg_t = 0.100                # filter size in seconds
    Lg = np.ceil(Lg_t*fs)       # in samples
    fft_len = 512
    mics = pra.Beamformer(R, room_bf.fs, N=fft_len, Lg=Lg)
    room_bf.add_microphone_array(mics)

    # Choose beamformer algorithm
    if (beamformer == "das"):
        mics.rake_delay_and_sum_weights(room_bf.sources[0][:1],room_bf.sources[1][:1])
    elif (beamformer == "mvdr"):
        mics.rake_mvdr_filters(room_bf.sources[0][:1] , room_bf.sources[1][:1] , sigma2_n * 
                               np.eye(mics.Lg * mics.M))
    elif (beamformer == "percuptual"):
        mics.rake_perceptual_filters(room_bf.sources[0][0:1] , room_bf.sources[1][0:1] , 
                                     sigma2_n * np.eye(mics.Lg * mics.M))

    # Run simulation
    room_bf.compute_rir()
    room_bf.simulate()

    # Replace microphone signals with the recorded signals in our dataset, i.e. ignoring dummy sources
    room_bf.mic_array.record(signal.T, fs)

    # Get enhanced signal
    enhanced_signal = mics.process(FD=False)
    enhanced_signal = pra.normalize(enhanced_signal)

    #Apply denoising 
    # 1) Spectral Subtractio
    if (denoiser == "SS"):
        enhanced_signal = apply_spectral_sub(enhanced_signal, nfft=512, db_reduc=12, lookback=15, beta=20, alpha=3)
        enhanced_signal = pra.normalize(enhanced_signal)
    # 2) iterative wiener
    elif (denoise == "wiener"):
        enhanced_signal = apply_iterative_wiener(enhanced_signal, frame_len=512,lpc_order=20, 
                                                 iterations=2, alpha=0.8, thresh=0.01)
        enhanced_signal = pra.normalize(enhanced_signal)

    # Apply narrowband speech filtering (does not really seem to make things better, but can save memory)
    if (speech_band_pass == True)
        sos = sps.butter(10, [30, 3000], 'bandpass', fs=fs, output='sos')
        enhanced_signal = sps.sosfilt(sos, enhanced_signal)
        enhanced_signal = pra.normalize(enhanced_signal)
        
    # save enhanced signal   
    output_file = save_dir  + "temp.wav"
    wavfile.write(output_file, fs, enhanced_signal)
        
    return output_file, enhanced_signal, fs

def generate_mels_labeled(audio_file,meta_file):
    # metafile name needs some preprocessing to match (because we renamed it with a prefix)
    x,sr = lb.load(audio_file, sr=None, mono=False)
    x = np.asfortranarray(x)
    with open(meta_file) as json_file:
        meta_data = json.load(json_file)

    X = lb.stft(x, n_fft=n_fft , hop_length=hop)
    X_mel = lb.feature.melspectrogram(x, sr=sr , n_mels=128)

    # Initializing labels per each frame
    labels = np.zeros([1,X.shape[1]])

    for section in meta_data['speech_segments']:
        start_time = section['start_time']
        end_time = section['end_time']
        start_frame, end_frame = lb.time_to_frames([start_time,end_time], sr=sr, hop_length=hop, n_fft= n_fft)
        labels[0,start_frame:end_frame+1] = 1 

    #np.savez(top_directory + spatial_process '_mels_labels/' + str(counter) +'.npz', mel=X_mel, labels=labels)   
    return X_mel, labels
        
# load trained model 
# VAD model
class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        #out = self.mp(self.relu(self.conv(x)))
        return out

class VAD(nn.Module):
    def __init__(self):
        super(VAD, self).__init__()
        self.a_norming = nn.BatchNorm2d(1) 
        self.to_db = torchaudio.transforms.AmplitudeToDB() 

        self.conv1 = Conv_2d(1,32)
        self.conv2 = Conv_2d(32,64)
        self.conv3 = Conv_2d(64,128)
        self.conv4 = Conv_2d(128,256)
        
        self.a_fc1 =  nn.Linear(10240, 512)
        self.a_fc2 = nn.Linear(512, 256)
        self.a_fc3 = nn.Linear(256, 128)       

        self.drop = nn.Dropout(p=0.3)
        self.logits  = nn.Linear(128, 1)
        
    def forward(self,audio_input):
        #Audio Branch 
        audio_db = self.to_db(audio_input) #[FIX! think need to upgrade torch]
        audio_norm = self.a_norming(audio_db) 
        
        x_audio = self.conv1(audio_norm)
        x_audio = self.conv2(x_audio)
        x_audio = self.conv3(x_audio)
        x_audio = self.conv4(x_audio)

        x_audio = x_audio.view(x_audio.size(0), -1)
        x_audio = F.relu(self.a_fc1(x_audio))
        x_audio = F.relu(self.a_fc2(x_audio))
        x_audio = F.relu(self.a_fc3(x_audio))
        
        #Merged Branch
        x_audio = self.drop(x_audio)
        logits = self.logits(x_audio)
        output = torch.sigmoid(logits)
        return output, logits

# get VAD
def get_VAD(device):
    # Define loss and optimizer
    vad_model = VAD()
    vad_model.to(device)
    return vad_model

# [TODO] convert input from np to torch
def run_VAD(vad_model, X_mel): 
    vad_model.eval()
    with torch.no_grad():
        mel_in = X_mel.to(device)

        # Choosing 3 seconds partitioning -> 92 frames
        half_window = int(FRAMES_3SEC/2)
        padded_mel = torch.zeros(1,1,128,mel_in.shape[3] + FRAMES_3SEC) #Padding input to have 3 seconds of silence at the end
        padded_mel[:,:,:,half_window:mel_in.shape[3]+half_window] = mel_in
        #num_batches = (padded_mel.shape[3] - FRAMES_3SEC) / BATCH_SIZE # Because we will ignore the first 92 frames

        #for batch in np.arange(0,num_batches):
        partitioned_mels_3secs = torch.zeros(mel_in.shape[3],1,128,FRAMES_3SEC)

        # Process all the frames (which starts from half_window in the padded mel, and lasts for all frames)
        for idx, central_frame in enumerate(np.arange(half_window,mel_in.shape[3]+half_window-1,1)):
            partitioned_mels_3secs[idx,:,:,:] = padded_mel[:,:,:,central_frame-half_window:central_frame+half_window]

        partitioned_mels_3secs = partitioned_mels_3secs.to(device)
        outputs, logits = vad_model(partitioned_mels_3secs)
        rounded_output = torch.round(outputs.data)
        return rounded_output, outputs, labels
    
def run(model_file, meta_file, audio_file):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))
    vad_model = get_VAD(device)
    vad_model.load_state_dict(torch.load(model_file))

    output_file, enhanced_signal, fs = enhance_signal(audio_file, metadata_file)
    X_mel, labels = generate_mels_labeled(output_file,meta_file)
    rounded_output, outputs, labels = run_VAD(vad_model, X_mel)
    # print accuracy
    # Save segmented bit?

if __name__ == "__main__":
    model_file = ""
    meta_file = ""
    audio_file = ""
    run(model_file, meta_file, audio_file)