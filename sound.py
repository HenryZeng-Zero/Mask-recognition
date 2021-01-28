import requests,os
class Speech():
    def __init__(self,spd,lang):
        spd = str(spd)
        lang = lang
        self.url = "https://fanyi.baidu.com/gettts?lan={}&spd={}&source=web&text=".format(lang,spd)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
        }

    def Speak(self,text):
        U = self.url+text
        return requests.get(U,headers=self.headers).content

class Play():
    def __init__(self,spd,lang):
        self.Say = Speech(spd,lang)

    def Play(self,text):
        with open('/home/pi/Mask/tts.mp3','wb+') as f:
            f.write(self.Say.Speak(text))
        os.system('mplayer /home/pi/Mask/tts.mp3')