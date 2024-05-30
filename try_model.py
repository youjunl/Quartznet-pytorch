'''
This is a demo for testing the pretrained QuartzNet model.
'''
import wave
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.data_layer import AudioDataLayer
from utils.audio_preprocessing import AudioToMelSpectrogramPreprocessor
from quartznet import QuartzNet

# set inference device to cpu
device = torch.device("cpu")

# parameters for chunking audio
vocabulary = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", "_"]

# spectrogram normalization constants
normalization = {}
normalization['fixed_mean'] = [
     -14.95827016, -12.71798736, -11.76067913, -10.83311182,
     -10.6746914,  -10.15163465, -10.05378331, -9.53918999,
     -9.41858904,  -9.23382904,  -9.46470918,  -9.56037,
     -9.57434245,  -9.47498732,  -9.7635205,   -10.08113074,
     -10.05454561, -9.81112681,  -9.68673603,  -9.83652977,
     -9.90046248,  -9.85404766,  -9.92560366,  -9.95440354,
     -10.17162966, -9.90102482,  -9.47471025,  -9.54416855,
     -10.07109475, -9.98249912,  -9.74359465,  -9.55632283,
     -9.23399915,  -9.36487649,  -9.81791084,  -9.56799225,
     -9.70630899,  -9.85148006,  -9.8594418,   -10.01378735,
     -9.98505315,  -9.62016094,  -10.342285,   -10.41070709,
     -10.10687659, -10.14536695, -10.30828702, -10.23542833,
     -10.88546868, -11.31723646, -11.46087382, -11.54877829,
     -11.62400934, -11.92190509, -12.14063815, -11.65130117,
     -11.58308531, -12.22214663, -12.42927197, -12.58039805,
     -13.10098969, -13.14345864, -13.31835645, -14.47345634]
normalization['fixed_std'] = [
     3.81402054, 4.12647781, 4.05007065, 3.87790987,
     3.74721178, 3.68377423, 3.69344,    3.54001005,
     3.59530412, 3.63752368, 3.62826417, 3.56488469,
     3.53740577, 3.68313898, 3.67138151, 3.55707266,
     3.54919572, 3.55721289, 3.56723346, 3.46029304,
     3.44119672, 3.49030548, 3.39328435, 3.28244406,
     3.28001423, 3.26744937, 3.46692348, 3.35378948,
     2.96330901, 2.97663111, 3.04575148, 2.89717604,
     2.95659301, 2.90181116, 2.7111687,  2.93041291,
     2.86647897, 2.73473181, 2.71495654, 2.75543763,
     2.79174615, 2.96076456, 2.57376336, 2.68789782,
     2.90930817, 2.90412004, 2.76187531, 2.89905006,
     2.65896173, 2.81032176, 2.87769857, 2.84665271,
     2.80863137, 2.80707634, 2.83752184, 3.01914511,
     2.92046439, 2.78461139, 2.90034605, 2.94599508,
     2.99099718, 3.0167554,  3.04649716, 2.94116777]

# simple data layer for passing audio signal
data_layer = AudioDataLayer(sample_rate=16000)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)

# audio preprocessing layer for model
preprocessor = AudioToMelSpectrogramPreprocessor(sample_rate=16000, normalize=normalization, dither = 0.0, pad_to=0).to(device)

# now initialize a model instance
model = QuartzNet()

# load the demo audio and conver all the frames to numpy array
filename = "./audio/demo.wav"
audioWave = wave.open(filename, 'rb')
audioData = audioWave.readframes(audioWave.getnframes())
audioData = np.frombuffer(audioData, dtype=np.int16)

# signal preprocessing and feature extraction for model
data_layer.set_signal(audioData)
batch = next(iter(data_loader))
audioDataFeed, audioDataLength = batch
modelInput = preprocessor.get_features(audioDataFeed, audioDataLength)

# feed data to the model and get log probability output
modelOutput = model.forward(modelInput)
logProbabilities = modelOutput[0].detach().numpy()

# convert log probability to characters using argmax
characters = ''
for i in range(logProbabilities.shape[0]):
     characters += vocabulary[np.argmax(logProbabilities[i])]

# convert characters to sentence using greedy merging
sentence = ''
previousCharacter = ''
for i in range(len(characters)):
     if characters[i] != previousCharacter:
          previousCharacter = characters[i]
          if previousCharacter != '_':
               sentence += previousCharacter

# print result
print(sentence)