import numpy as np
from pathlib import Path

def guess_seq_len(seq):
    guess = 1
    max_len = int(len(seq) / 2)
    for x in range(2, max_len):
        if seq[0:x] == seq[x:2*x]:
            return x, seq[0:x]

    return guess, 0

def nbr_of_reps(array, rep):
    step = len(rep)
    reps = 0
    first_idxs = []
    indexes_to_remove = []

    for x in range(0, len(array), step):
        if array[x:x+step] == rep:
            first_idxs.append(x)
            reps += 1

    for i in first_idxs:
        for j in range(0, step):
            indexes_to_remove.append(i + j)

    array = np.delete(array, indexes_to_remove).tolist()

    return reps, array

FOLDER_FILES = str(Path("C:/Users/vabicheq/Documents/Repos/MotionMatching/Assets/output/Dual/Mixamo"))

loaded = np.load(FOLDER_FILES + '/1/Right/' + "0.5729578_dominant_clip_every_frame.npz", allow_pickle=True)
all_clips = loaded['clips']
a = [int(item['AnimId']) for item in all_clips]

b = 2

while b > 1 and len(a) > 2:
    print("Array:", a)

    b, seq = guess_seq_len(a)

    print("Length of the pattern:", b, "Pattern:", seq)

    reps, a = nbr_of_reps(a, seq)

    print("Number of repetitions:", reps, "Remaining array:", a)