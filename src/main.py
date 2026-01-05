import sounddevice as sd
import numpy as np
from pynput import keyboard
import threading
import time
from faster_whisper import WhisperModel
from models.agent import OllamaClient

# ===============================
# CONFIG
# ===============================
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

CHUNK_SECONDS = 5.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)

model_size = "turbo"
device = "cuda"

whisper_model = WhisperModel(
    model_size,
    device=device,
    compute_type="float16"
)

llm_chatbot = OllamaClient()
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
    history = []

    while not stop_event.is_set():
        time.sleep(0.1)

        # snapshot do buffer
        with lock:
            if not audio_queue:
                continue

            audio_np = np.concatenate(audio_queue, axis=0)

            if len(audio_np) < CHUNK_SAMPLES:
                continue

            # limpa buffer corretamente
            audio_queue.clear()

        # processamento pesado fora do lock
        audio_float = (audio_np.astype(np.float32) / 32768.0).squeeze()

        segments, _ = whisper_model.transcribe(
            audio_float,
            beam_size=1,
            language="pt"
        )

        text = " ".join(s.text for s in segments).strip()

        if not text or text == last_printed_text:
            continue

        print("🗣️ Você:", text)
        last_printed_text = text

        # adiciona mensagem do usuário
        history.append({
            "role": "user",
            "content": text
        })

        # chama o LLM
        llm_response = llm_chatbot.chat(messages=history)

        print("🤖 Bot:", llm_response)

        # adiciona resposta do assistente
        history.append({
            "role": "assistant",
            "content": llm_response
        })

        # limpa o buffer compartilhado
        audio_queue.clear()
        audio_np = 0
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
