import speech_recognition as sr


def chinese_speech2text(duration = 2, language = 'zh-CN'):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.record(source, duration = duration)
        audio = r.listen(source)
    
    return r.recognize_sphinx(audio, language = language)