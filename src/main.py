import sounddevice as sd
import numpy as np
from pynput import keyboard
import threading
import time
import re
from faster_whisper import WhisperModel
from models.agent import OllamaClient

# ===============================
# CONFIG
# ===============================
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

CHUNK_SECONDS = 10.0
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
MAX_HISTORY_MESSAGES = 12
SILENCE_RMS_THRESHOLD = 0.008
SPEECH_BAND_MIN_HZ = 85
SPEECH_BAND_MAX_HZ = 3000
MIN_SPEECH_BAND_RATIO = 0.45
MIN_PEAK_AMPLITUDE = 0.03
MIN_TRANSCRIPT_CHARS = 4

model_size = "turbo"
PREFERRED_DEVICE = "cuda"


def build_whisper_model():
    try:
        return WhisperModel(
            model_size,
            device=PREFERRED_DEVICE,
            compute_type="float16"
        )
    except Exception as exc:
        print(
            f"Falha ao iniciar Whisper em {PREFERRED_DEVICE} ({exc}). "
            "Usando CPU."
        )
        return WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )


whisper_model = build_whisper_model()

llm_chatbot = OllamaClient(model="qwen3:0.6b")
# ===============================
# CONTROLE
# ===============================
stop_event = threading.Event()
audio_queue = []
lock = threading.Lock()

# ===============================
# CALLBACK DE ÁUDIO
# ===============================
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    with lock:
        audio_queue.append(indata.copy())


def has_human_voice(audio_float: np.ndarray) -> bool:
    if audio_float.size == 0:
        return False

    rms = float(np.sqrt(np.mean(np.square(audio_float))))
    peak = float(np.max(np.abs(audio_float)))

    if rms < SILENCE_RMS_THRESHOLD or peak < MIN_PEAK_AMPLITUDE:
        return False

    windowed = audio_float * np.hanning(len(audio_float))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / SAMPLE_RATE)
    power = np.abs(spectrum) ** 2

    total_energy = float(np.sum(power)) + 1e-12
    speech_band = (freqs >= SPEECH_BAND_MIN_HZ) & (freqs <= SPEECH_BAND_MAX_HZ)
    speech_energy = float(np.sum(power[speech_band]))
    speech_ratio = speech_energy / total_energy

    return speech_ratio >= MIN_SPEECH_BAND_RATIO


def is_spurious_transcript(text: str) -> bool:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if len(cleaned) < MIN_TRANSCRIPT_CHARS:
        return True

    known_noise = {
        "e ai",
        "ei",
        "ai",
        "obrigado",
        "obrigada",
        "valeu",
        "hum",
        "hmm",
        "aham",
        "uh",
    }
    return cleaned in known_noise

# ===============================
# THREAD DE TRANSCRIÇÃO
# ===============================
def transcription_loop():
    history = []
    pending_audio = np.empty((0, CHANNELS), dtype=np.int16)

    while not stop_event.is_set():
        time.sleep(0.05)

        with lock:
            if audio_queue:
                captured_audio = np.concatenate(audio_queue, axis=0)
                audio_queue.clear()
            else:
                captured_audio = None

        if captured_audio is None:
            continue

        pending_audio = np.concatenate((pending_audio, captured_audio), axis=0)

        while len(pending_audio) >= CHUNK_SAMPLES and not stop_event.is_set():
            audio_np = pending_audio[:CHUNK_SAMPLES]
            pending_audio = pending_audio[CHUNK_SAMPLES:]

            audio_float = (audio_np.astype(np.float32) / 32768.0).squeeze()
            if not has_human_voice(audio_float):
                continue

            try:
                segments, _ = whisper_model.transcribe(
                    audio_float,
                    beam_size=1,
                    language="pt",
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 500,
                        "speech_pad_ms": 250,
                    },
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6
                )
                text = " ".join(s.text for s in segments).strip()
            except Exception as exc:
                print(f"Erro na transcrição: {exc}")
                continue

            if not text or is_spurious_transcript(text):
                continue

            print("🗣️ Você:", text)

            history.append({
                "role": "user",
                "content": text
            })

            if len(history) > MAX_HISTORY_MESSAGES:
                history = history[-MAX_HISTORY_MESSAGES:]

            try:
                llm_response = llm_chatbot.chat(messages=history)
            except Exception as exc:
                print(f"Erro ao consultar o LLM: {exc}")
                continue

            print("🤖 Bot:", llm_response)

            history.append({
                "role": "assistant",
                "content": llm_response
            })

            if len(history) > MAX_HISTORY_MESSAGES:
                history = history[-MAX_HISTORY_MESSAGES:]


# ===============================
# TECLADO
# ===============================
def on_press(key):
    try:
        if key.char and key.char.lower() == "q":
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

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=audio_callback,
            blocksize=2048
        ):
            stop_event.wait()
    except KeyboardInterrupt:
        stop_event.set()
    except Exception as exc:
        print(f"Erro no stream de áudio: {exc}")
        stop_event.set()
    finally:
        listener.stop()
        stt_thread.join(timeout=2)

    print("Finalizado.")
