from ffmpy import FFmpeg
import os

if not os.path.exists('./frames'):
    os.mkdir('./frames')
for video in os.listdir('./downsampled_videos'):
    print("********************************")
    print(video)
    print("********************************")
    name = video[0:video.find('.')]
    if not os.path.exists('./frames/'+name):
        os.mkdir('./frames/'+name)
    ff = FFmpeg(inputs={'./downsampled_videos/' + video: None}, outputs={'./frames/'+name+'/'+name+"_%d.png": ['-y', '-vf', 'fps=1/5']})
    ff.run()
