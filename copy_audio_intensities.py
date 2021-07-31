import shutil
import os
import glob

root_src = 'MEAD_AUDIO'
keep = 'level_3'
root_dst = 'MEAD_AUDIO_3'

f = glob.glob('{}\\**\\*.m4a'.format(root_src), recursive=True)
f = [file for file in f if keep in file.split('\\')]

for ff in f:
    dst = root_dst+'\\'+'\\'.join(ff.split('\\')[1:])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(ff, dst)
