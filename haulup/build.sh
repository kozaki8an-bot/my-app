#!/usr/bin/env bash
# Full pipeline: transcribe -> render frames -> mux original voice-over (loudness-normalised, no speed change)
set -e
cd "$(dirname "$0")"
mkdir -p build out
[ -f build/words.json ] || { ffmpeg -y -v error -i assets/audio/voiceover.mp3 -ar 16000 -ac 1 -f f32le build/vo.f32; python3 transcribe.py; }
python3 render.py
ffmpeg -y -v error -i build/video_silent.mp4 -i assets/audio/voiceover.mp3 \
  -filter_complex "[1:a]loudnorm=I=-16:TP=-1.5:LRA=7,aresample=44100,apad=whole_dur=21,afade=t=out:st=20.6:d=0.4[a]" \
  -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -ar 44100 -t 21 -movflags +faststart out/HaulUp_CarHauler_Square_1080.mp4
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,duration -of default=nw=1 out/HaulUp_CarHauler_Square_1080.mp4
