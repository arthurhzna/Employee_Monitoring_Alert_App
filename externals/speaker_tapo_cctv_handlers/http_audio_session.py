import asyncio
import hashlib
import json
import logging
import random
import warnings
from asyncio import StreamReader, StreamWriter, Task, Queue
from json import JSONDecodeError
from typing import Optional, Mapping, Generator, MutableMapping
from pytapo.media_stream.response import HttpMediaResponse
from pytapo import HttpMediaSession
from pytapo.const import EncryptionMethod

class HttpAudioSession(HttpMediaSession):
    
    def __init__(
        self,
        ip: str,
        cloud_password: str,
        username: str = "admin",
        super_secret_key: str = "dadada",
        encryptionMethod: str = EncryptionMethod.SHA256,
    ):
        super().__init__(
            ip=ip,
            cloud_password=cloud_password,
            super_secret_key=super_secret_key,
            encryptionMethod=encryptionMethod,
            username=username,
        )
        
        
    async def transceive_audio(
        self,
        data: str,
        mimetype: str = "audio/mp2t",
        session: int = None,
        encrypt: bool = False,
        no_data_timeout=1.0,
    ) -> None:
        sequence = None
        queue = None

        if mimetype != "application/json" and session is None:
            raise ValueError("Non-JSON streams must always be bound to a session")

        if mimetype == "application/json":
            j = json.loads(data)
            if "type" in j and j["type"] == "request":
                sequence = random.randint(1000, 0x7FFF)
                j["seq"] = sequence
            data = json.dumps(j, separators=(",", ":"))

        if (
            (sequence is None)
            and (session is None)
            or (session is not None and session not in self._sessions)
        ):
            raise ValueError(
                "Data is not a request and no existing session has been found"
            )

        if session is not None:
            queue = self._sessions[session]
        if sequence is not None:
            queue = asyncio.Queue(128)
            self._sequence_numbers[sequence] = queue

        if type(data) == str:
            data = data.encode()

        headers = {
            b"Content-Type": mimetype.encode(),
        }

        if encrypt:
            data = self._aes.encrypt(data)
            headers[b"X-If-Encrypt"] = b"1"

        headers[b"Content-Length"] = str(len(data)).encode()

        if mimetype != "application/json":
            headers[b"X-If-Encrypt"] = str(
                int(encrypt)
            ).encode()  
            if session is not None:
                headers[b"X-Session-Id"] = str(
                    session
                ).encode()  

        if self.window_size is not None:
            headers[b"X-Data-Window-Size"] = str(self.window_size).encode()

        await self._send_http_request(b"--" + self.client_boundary, headers)

        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            self._writer.write(data[i : i + chunk_size])
            await self._writer.drain()

        self._writer.write(b"\r\n")
        await self._writer.drain()          
            
    async def transceive_keepSession(
        self,
        data: str,
        mimetype: str = "application/json",
        session: int = None,
        encrypt: bool = False,
        no_data_timeout=1.0,
    ) -> Generator[HttpMediaResponse, None, None]:
        sequence = None
        queue = None

        if mimetype != "application/json" and session is None:
            raise ValueError("Non-JSON streams must always be bound to a session")

        if mimetype == "application/json":
            j = json.loads(data)
            if "type" in j and j["type"] == "request":
                sequence = random.randint(1000, 0x7FFF)
                j["seq"] = sequence
            data = json.dumps(j, separators=(",", ":"))

        if (
            (sequence is None)
            and (session is None)
            or (session is not None and session not in self._sessions)
        ):
            raise ValueError(
                "Data is not a request and no existing session has been found"
            )

        if session is not None:
            queue = self._sessions[session]
        if sequence is not None:
            queue = asyncio.Queue(128)
            self._sequence_numbers[sequence] = queue

        if type(data) == str:
            data = data.encode()

        headers = {
            b"Content-Type": mimetype.encode(),
        }

        if encrypt:
            data = self._aes.encrypt(data)
            headers[b"X-If-Encrypt"] = b"1"

        headers[b"Content-Length"] = str(len(data)).encode()

        if mimetype != "application/json":
            headers[b"X-If-Encrypt"] = str(
                int(encrypt)
            ).encode()  
            if session is not None:
                headers[b"X-Session-Id"] = str(
                    session
                ).encode()  

        if self.window_size is not None:
            headers[b"X-Data-Window-Size"] = str(self.window_size).encode()

        await self._send_http_request(b"--" + self.client_boundary, headers)

        chunk_size = 4096
        for i in range(0, len(data), chunk_size):
            self._writer.write(data[i : i + chunk_size])
            await self._writer.drain()

        self._writer.write(b"\r\n")
        await self._writer.drain()
        try:
            while True:
                coro = queue.get()
                if no_data_timeout is not None:
                    try:
                        resp: HttpMediaResponse = await asyncio.wait_for(
                            coro, timeout=no_data_timeout
                        )
                    except asyncio.exceptions.TimeoutError:
                        print(
                            "Server did not send a new chunk in {} sec (sequence {}"
                            ", session {}), assuming the stream is over".format(
                                no_data_timeout, sequence, session
                            )
                        )
                        break
                else:
                    resp: HttpMediaResponse = await coro
                if resp.session is not None:
                    session = resp.session
                if resp.encrypted and isinstance(resp.plaintext, Exception):
                    raise resp.plaintext

                yield resp
                break

        finally:
            pass