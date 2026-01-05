import sounddevice as sd
import numpy as np
from pynput import keyboard
import threading
import time
from faster_whisper import WhisperModel

# ===============================
# CONFIG
# ===============================
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

CHUNK_SECONDS = 5.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)

model_size = "small"
device = "cuda"

whisper_model = WhisperModel(
    model_size,
    device=device,
    compute_type="float16"
)

# ===============================
# CONTROLE
# ===============================
stop_event = threading.Event()
audio_queue = []
lock = threading.Lock()
last_printed_text = ""

# ===============================
# CALLBACK DE ÁUDIO
# ===============================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    with lock:
        audio_queue.append(indata.copy())

# ===============================
# THREAD DE TRANSCRIÇÃO
# ===============================
def transcription_loop():
    global last_printed_text

    while not stop_event.is_set():
        time.sleep(0.1)

        # pega snapshot do buffer com lock
        with lock:
            if not audio_queue:
                continue
            audio_np = np.concatenate(audio_queue, axis=0)

            # ainda não atingiu o tamanho mínimo
            if len(audio_np) < CHUNK_SAMPLES:
                continue
        
        # fora do lock: processamento pesado
        audio_float = (audio_np.astype(np.float32) / 32768.0).squeeze()

        segments, _ = whisper_model.transcribe(
            audio_float,
            beam_size=1,
            language="pt"
        )

        text = " ".join(s.text for s in segments).strip()

        if text and text != last_printed_text:
            print("🗣️", text)
            last_printed_text = text
        # limpa o buffer compartilhado
        audio_queue.clear()
# ===============================
# TECLADO
# ===============================
def on_press(key):
    try:
        if key.char == "q":
            print("\n🛑 Encerrando...")
            stop_event.set()
            return False
    except AttributeError:
        pass

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    stt_thread = threading.Thread(target=transcription_loop, daemon=True)
    stt_thread.start()

    print("Gravando... pressione 'q' para parar\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=audio_callback,
        blocksize=2048
    ):
        stop_event.wait()

    print("Finalizado.")
