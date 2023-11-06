from pydub import AudioSegment

# M4A 파일 경로
m4a_file = "/home/yai/team2/junwan/VITS/myquestion.m4a"

# WAV 파일 경로 (변환 후)
wav_file = "output.wav"

# M4A 파일을 로드
audio = AudioSegment.from_file(m4a_file, format="m4a")

# WAV로 저장
audio.export(wav_file, format="wav")

print(f"{m4a_file}을 {wav_file}로 변환 완료")
