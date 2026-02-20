import whisper
import os

# Load Whisper model once when this file runs
# "base" is fast but less accurate. "small" is better for interviews.
# We start with "base" for testing, upgrade later.
model = whisper.load_model("base")

def transcribe_audio(filepath):
    """Take an audio file, return the text Whisper heard."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    print(f"Transcribing: {filepath}")
    result = model.transcribe(filepath)
    text = result["text"].strip()
    print(f"Heard: {text}")
    return text

if __name__ == "__main__":
    # Test: record 5 seconds, then transcribe it
    from audio_capture import record_clip
    
    print("Recording 5 seconds... say something!")
    filepath = record_clip("test_clip.wav")
    
    print("Now transcribing...")
    text = transcribe_audio(filepath)
    
    if text:
        print(f"\nFinal result: {text}")