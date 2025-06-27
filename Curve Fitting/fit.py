from Lorentzian import LorentzianModel
import numpy as np

def get_peaks(data, n: int):
    assert isinstance(data, (list, np.ndarray))
    peaks = {}
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1]:
            peaks[i] = data[i]
    highest_peaks = dict(sorted(peaks.items(), key=lambda x: x[1], reverse=True))
    return list(highest_peaks.keys())

file_path = '/Volumes/BEAR/Images_new/Raman/Sample 13/FeMoS2_1_loc1.txt'

# initialize model
with open(file_path, encoding='latin-1') as f:
    data = f.readlines()
    data = data[32:]
    data = [data[i].split() for i in range(len(data))]

    x = []
    y = []
    for i in range(len(data)):
        point = float(data[i][0])
        if (point > 300 and point < 450):
            x.append(point)
            y.append(float(data[i][1]) - ((float(data[-1][1]) + float(data[0][1])) / 3)  if float(data[i][1]) - float(data[-1][1]) > 0 else 0) # subtracts baseline

    x0 = [x[get_peaks(y, 2)[i]] for i in range(2)]
    a = sorted(y, reverse=True)[0]
    fit = LorentzianModel(x0, a)
    epochs = 1000
    for _ in range(epochs):
        fit.batch_update(x,y)
        
   # fit.graphvsdata(x,y,file_path)
    print(fit.parameters)
    print(f'Peak diff: {fit.parameters[0][0] - fit.parameters[1][0]}')