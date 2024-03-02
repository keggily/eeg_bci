
"""
Source: https://github.com/labstreaminglayer/pylsl/tree/master
Author: Christian Kothe
Modified HD
"""

from pylsl import StreamInlet, resolve_stream


# first resolve an EEG stream on the lab network
print("looking for an EEG stream named 'BioSemi'...")
streams = resolve_stream('type', 'EEG', 'name', 'BioSemi')


# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    print(timestamp, sample)

