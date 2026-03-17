from pytapo import Tapo
import time
import math
import asyncio
from typing import Optional, Dict, Any
import core.config as config
from externals.speaker_tapo_cctv_handlers.http_audio_session import HttpAudioSession
from externals.speaker_tapo_cctv_handlers.resampling_audio import Resampling_Audio
from externals.speaker_tapo_cctv_handlers.hfix import Paketovadlo

class TapoCctvSpeakerClient:
    CHUNK_LEN = 320 * 6
    TAPO_AUDIO_PCKT_LEN = 1504
    def __init__(
        self,
        encrypt : bool = True
    ) -> None:
        self._paketovadlo = Paketovadlo()
        self._ip = config.Config.tapo_cctv.ip
        self._user = config.Config.tapo_cctv.user
        self._password = config.Config.tapo_cctv.password
        self._secret = config.Config.tapo_cctv.secret
        self._encrypt = encrypt

        self._tapo_ses: HttpAudioSession | None = None
        self._ses_id = -1
        self._timestamp = 0
        self._init_session()

    def _init_session(self) -> None:
        self._tapo_ses = HttpAudioSession(
            ip=self._ip,
            cloud_password=self._password,
            username=self._user,
            super_secret_key=self._secret,
        )
        self._ses_id = -1
        self._timestamp = 0

    async def half_duplex_start(self) -> None:

        if not self._tapo_ses._started:
            await self._tapo_ses.start()

        rt = self._tapo_ses.transceive_keepSession(
            data='{"type": "request","seq":1, "params": {"talk": {"mode": "half_duplex"}, "method": "get"}}',
            mimetype="application/json",
            encrypt=True,
        )
        async for h in rt:
            self._ses_id = h.session

    async def half_duplex_stop(self) -> None:
        rt = self._tapo_ses.transceive_keepSession(
            data='{"type":"request","seq":2,"params":{"stop":"null","method":"do"}}',
            mimetype="application/json",
            encrypt=True,
            session=self._ses_id,
        )
        async for h in rt:
            pass

    def process_audio_file(self, wav_filename) -> bytearray:
        wav = open(wav_filename, "rb")
        data = wav.read()
        print(data)
        data = [
            int.from_bytes(data[44 + i : 45 + i], byteorder="big", signed=True)
            for i in range(len(data) - 44)
        ]
        stream = []
        chunku = math.ceil(len(data) / TapoCctvSpeakerClient.CHUNK_LEN)
        self._timestamp = round(time.time() * 1000)
        for i in range(chunku):
            chunk = data[i * TapoCctvSpeakerClient.CHUNK_LEN : (i + 1) * (TapoCctvSpeakerClient.CHUNK_LEN) + 0]
            chunk = Resampling_Audio.b(chunk, len(chunk))
            _len = len(chunk)
            chunk = self._paketovadlo.k(chunk, _len, self._timestamp)
            stream += chunk
            self._timestamp += round(_len * 90000 / 8000)
        stream = [x if x > -1 else (256 + x) for x in stream]
        return bytearray(stream)

    async def _play_ts_file(self, ts_filename) -> None:
        await self.half_duplex_start()
        in_file = open(ts_filename, "rb")
        data = in_file.read()
        in_file.close()
        packetu = math.ceil(len(data) / TapoCctvSpeakerClient.TAPO_AUDIO_PCKT_LEN)
        for i in range(packetu):
            packet = data[
                i
                * TapoCctvSpeakerClient.TAPO_AUDIO_PCKT_LEN : (i + 1)
                * TapoCctvSpeakerClient.TAPO_AUDIO_PCKT_LEN
            ]
            time.sleep(
                (TapoCctvSpeakerClient.CHUNK_LEN) / (8000) / 2
            )
            await self._tapo_ses.transceive_audio(
                data=packet, encrypt=self._encrypt, session=self._ses_id
            )
        await self.half_duplex_stop()

    async def send_pcma_packet(self, data):
        rt = await self._tapo_ses.transceive_audio(
            data=data, encrypt=self._encrypt, session=self._ses_id
        )

    def process_audio_chunk(self, chunk):
        chunku = math.ceil(len(chunk) / TapoCctvSpeakerClient.CHUNK_LEN)
        if self._timestamp == 0:
            self._timestamp = round(time.time() * 1000)
        chunk = Resampling_Audio.b(chunk, len(chunk))
        _len = len(chunk)
        chunk = self._paketovadlo.k(chunk, _len, self._timestamp)
        self._timestamp += round(_len * 90000 / 8000)
        chunk = [x if x > -1 else (256 + x) for x in chunk]
        return bytearray(chunk)

    async def process_and_send_audio_chunk(self, data):
        data = self.process_audio_chunk(data)
        await self.send_pcma_packet(data)

    def wav2ts(self, wav_filename, out_filename):
        stream = self.process_audio_file(wav_filename)
        out = open(out_filename, "wb")
        out.write(stream)
        out.close()

    async def stream_wav(self, wav_filename) -> None:
        try:
            if self._tapo_ses is None:
                self._init_session()

            await self.half_duplex_start()
            if not self._tapo_ses._started:
                await self._tapo_ses.start()
            wav = open(wav_filename, "rb")
            data = wav.read()
            data = [
                int.from_bytes(data[44 + i : 45 + i], byteorder="big", signed=True)
                for i in range(len(data) - 44)
            ]

            chunku = math.ceil(len(data) / TapoCctvSpeakerClient.CHUNK_LEN)
            self._timestamp = round(time.time() * 1000)

            stream = []

            for i in range(chunku):
                chunk = data[i * TapoCctvSpeakerClient.CHUNK_LEN : (i + 1) * TapoCctvSpeakerClient.CHUNK_LEN]
                chunk = Resampling_Audio.b(chunk, len(chunk))
                _len = len(chunk)
                chunk = self._paketovadlo.k(chunk, _len, self._timestamp)
                chunk = [x if x > -1 else (256 + x) for x in chunk]
                stream += chunk
                self._timestamp += round(_len * 90000 / 8000)
                if len(stream) >= 1504:
                    chunk = bytes(stream[:1504])
                    stream = stream[1504:]
                    await self._tapo_ses.transceive_audio(
                        data=chunk, encrypt=self._encrypt, session=self._ses_id
                    )
                    time.sleep((TapoCctvSpeakerClient.CHUNK_LEN) / (8000) / 2)

            await self.half_duplex_stop()
        finally:
            try:
                if self._tapo_ses is not None:
                    await self._tapo_ses.stop()
            except Exception:
                pass
            self._tapo_ses = None

    def ts2camera(self, ts_filename) -> None:
        asyncio.run(self._play_ts_file(ts_filename))

    def streamovat(self, filename) -> None:
        self._init_session()
        asyncio.run(self.stream_wav(filename))


# if __name__ == "__main__":
    # talk = SpeakerTapoCctvClient(encrypt=True)
    # SpeakerTapoCctvClient.wav2ts("/home/inovasi/Documents/AlertEyeLevel/cv_project/audio/fokus.wav", "/home/inovasi/Documents/AlertEyeLevel/cv_project/audio/fokus.ts")
    # talk.ts2camera("fiala.ts")