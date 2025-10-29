import os
import shutil
#files = os.listdir("/mnt/QNAP/comdav/DATA/DATA/LJSpeech/wavs/")

#files = files[:5000]


with open("ljspeech_manipulated_metadata.txt", "r") as f:
    lines = [next(f).strip() for _ in range(5000)]


for line in lines:
    file_name = line.split(',')[0]
    full_path = os.path.join("/mnt/QNAP/comdav/DATA/DATA/LJSpeech/wavs/", file_name)
    shutil.copy(full_path, "LJSpeech_vocoded/")

    

        
