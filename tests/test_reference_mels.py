import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from finetuning.dataset import TTSDataset, encode_reference_mels
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder


class Processor:
    def __call__(self, text, return_tensors, padding):
        return {"input_ids": torch.arange(20, dtype=torch.long).unsqueeze(0)}


class ReferenceMelBatchTests(unittest.TestCase):
    def test_different_reference_durations_preserve_individual_embeddings(self):
        talker = SimpleNamespace(
            codec_nothink_id=1,
            codec_think_bos_id=2,
            codec_think_eos_id=3,
            codec_pad_id=4,
            codec_bos_id=5,
            codec_eos_token_id=6,
        )
        config = SimpleNamespace(
            tts_pad_token_id=7,
            tts_bos_token_id=8,
            tts_eos_token_id=9,
            talker_config=talker,
        )
        encoder_config = Qwen3TTSSpeakerEncoderConfig(
            enc_dim=16,
            enc_channels=[16, 16, 16, 16, 48],
            enc_attention_channels=8,
            enc_res2net_scale=4,
            enc_se_channels=8,
        )
        encoder = Qwen3TTSSpeakerEncoder(encoder_config).eval()

        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for index, seconds in enumerate((0.5, 0.8)):
                samples = np.sin(
                    2 * np.pi * 440 * np.arange(int(24000 * seconds)) / 24000
                ).astype(np.float32)
                path = Path(temp) / f"ref{index}.wav"
                sf.write(path, samples, 24000)
                paths.append(str(path))

            records = [
                {"audio": path, "text": "sample", "audio_codes": [[0] * 16] * 3, "ref_audio": path}
                for path in paths
            ]
            dataset = TTSDataset(records, Processor(), config)
            loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
            batch = next(iter(Accelerator(cpu=True).prepare(loader)))
            ref_mels = batch["ref_mels"]
            self.assertIsInstance(ref_mels, list)
            self.assertNotEqual(ref_mels[0].shape[1], ref_mels[1].shape[1])

            with torch.inference_mode():
                actual = encode_reference_mels(encoder, ref_mels, torch.device("cpu"), torch.float32)
                expected = torch.cat([encoder(mel) for mel in ref_mels], dim=0)
            self.assertEqual(actual.shape, (2, 16))
            self.assertTrue(torch.allclose(actual, expected))

            same_reference = [dict(record, ref_audio=paths[0]) for record in records]
            same_dataset = TTSDataset(same_reference, Processor(), config)
            same_batch = next(iter(DataLoader(same_dataset, batch_size=2, collate_fn=same_dataset.collate_fn)))
            self.assertIsInstance(same_batch["ref_mels"], torch.Tensor)
            with torch.inference_mode():
                actual = encode_reference_mels(
                    encoder, same_batch["ref_mels"], torch.device("cpu"), torch.float32
                )
                expected = encoder(same_batch["ref_mels"])
            self.assertEqual(actual.shape, (2, 16))
            self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
