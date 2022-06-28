from pynimation.inout import load
from pynimation.common import data as data_
from pynimation.inout.fbx.exporter import FBXExporter
from pynimation.anim.metrics.similarity import Similarity
from pynimation.anim.metrics import GlobalPositionAndVelocityWithoutRootMetric
from os import walk

ANIMATION_PATH = "C:/Users/vabicheq/Documents/Repos/mocap-evaluation/animations/DemoMocap/"

FPS = 30

time_between_frames = 1/FPS

f = []
for (dirpath, dirnames, filenames) in walk(ANIMATION_PATH):
    f.extend(filenames)

for i in range(1, len(f)):
    anim1 = load(ANIMATION_PATH + f[i - 1])
    print(ANIMATION_PATH + f[i - 1], anim1)    
    anim2 = load(ANIMATION_PATH + f[i])
    print(ANIMATION_PATH + f[i], anim2)
    # for i, frame in enumerate(animation.frames):
        # root = frame.getBoneGlobalPosition(frame.skeleton.root.id)
    print(i, ':', Similarity(GlobalPositionAndVelocityWithoutRootMetric, anim1, anim2))
