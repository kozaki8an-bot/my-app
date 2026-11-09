import json, sys
import numpy as np
from faster_whisper import WhisperModel
m = WhisperModel("small.en", device="cpu", compute_type="int8")
segs, _ = m.transcribe(np.fromfile("build/vo.f32", dtype=np.float32), word_timestamps=True, language="en")
words = [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)} for s in segs for w in s.words]
json.dump(words, open("build/words.json", "w"), indent=1)
print(" ".join(f'{w["w"]}[{w["s"]}]' for w in words))
