from pygame import mixer

def playSound(filedir):
	mixer.init()
	mixer.music.load(filedir)
	mixer.music.play()
