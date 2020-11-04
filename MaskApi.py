import subprocess
from sound import Play
import json 

Mask = "build/Mask"
pyramidbox_lite = "/home/pi/Mask/Data/pyramidbox_lite.nb"
mask_detector = "/home/pi/Mask/Data/mask_detector.nb"

o = subprocess.Popen([Mask,pyramidbox_lite,mask_detector],stdout=subprocess.PIPE, stderr=subprocess.STDOUT,bufsize=1)

play = Play(5,"zh")
j = None
while True:
    line = str(o.stdout.readline(),encoding = "utf-8")

    try:
        j = json.loads(line)
        for i in j["Data"]:
            if (float(i["p"]) < 0.8):
                play.Play("你没有戴口罩")
    except:
        pass
    print(j)

    
    # o.stdout.flush()