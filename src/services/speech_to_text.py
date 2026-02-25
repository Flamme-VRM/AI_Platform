import asyncio
import tempfile
import os
import logging
from pydub import AudioSegment
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import soundfile as sf
import gc

logger = logging.getLogger(__name__)


class SpeechToTextService:
    def __init__(self):
        self.asr_model_name = "abilmansplus/whisper-turbo-ksc2"
        try:
            self.processor = AutoProcessor.from_pretrained(self.asr_model_name)
            self.asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(self.asr_model_name)

            if torch.cuda.is_available():
                self.asr_model.to("cuda")
                logger.info(f"ASR model '{self.asr_model_name}' moved to GPU.")
            else:
                logger.warning(
                    f"No GPU found. ASR model '{self.asr_model_name}' will run on CPU, which might be slow.")

            logger.info(f"Hugging Face ASR model '{self.asr_model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Hugging Face ASR model '{self.asr_model_name}': {e}")
            self.asr_model = None
            self.processor = None

    def cleanup(self):
        """Освобождение ресурсов GPU и памяти."""
        if hasattr(self, 'asr_model'):
            del self.asr_model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("SpeechToText resources released.")

    async def convert_voice_to_text(self, voice_file_path: str, language: str = "kk-KZ") -> str:
        wav_path = None

        try:
            wav_path = await self._convert_to_wav(voice_file_path)

            try:
                audio_input, sample_rate = sf.read(wav_path)

                if sample_rate != 16000:
                    logger.warning(f"Audio sample rate is {sample_rate}Hz, converting to 16kHz for ASR.")
                    audio_segment = AudioSegment.from_file(wav_path)
                    audio_segment = audio_segment.set_frame_rate(16000)
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav_16k_file:
                        temp_wav_path_16k = temp_wav_16k_file.name
                    audio_segment.export(temp_wav_path_16k, format="wav")
                    audio_input, sample_rate = sf.read(temp_wav_path_16k)
                    os.unlink(temp_wav_path_16k)

                input_features = self.processor(
                    audio_input,
                    sampling_rate=sample_rate,
                    return_tensors="pt"
                ).input_features

                if torch.cuda.is_available():
                    input_features = input_features.to("cuda")

                predicted_ids = self.asr_model.generate(
                    input_features,
                    language="kazakh",
                    task="transcribe"
                )

                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                logger.info(f"Hugging Face ASR successful: {transcription}")

                return transcription

            except Exception as e:
                logger.error(f"Fallback: {e}")

        except Exception as e:
            logger.error(f"Speech recognition (overall) failed: {e}")
            return ""
        finally:
            try:
                if wav_path and os.path.exists(wav_path):
                    os.unlink(wav_path)
                if voice_file_path and os.path.exists(voice_file_path):
                    os.unlink(voice_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary files: {e}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    async def _convert_to_wav(self, ogg_path: str) -> str:
        wav_path = ogg_path.replace('.ogg', '.wav')

        def convert():
            audio = AudioSegment.from_ogg(ogg_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(wav_path, format="wav")

        await asyncio.to_thread(convert)
        return wav_path