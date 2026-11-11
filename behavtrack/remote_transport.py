"""HTTPS transport that authenticates a paired server before sending any secret."""
from __future__ import annotations
import hashlib
import http.client
import json
import re
import ssl
from urllib.parse import urlsplit

CHUNK = 2 * 1024 * 1024


def validate_pair(value):
    url = urlsplit(str(value.get("url", "")))
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment:
        raise ValueError("서버 주소는 https://주소:포트 형식이어야 합니다.")
    if not re.fullmatch(r"[a-f0-9]{64}", str(value.get("certificate_sha256", ""))):
        raise ValueError("서버 인증서 지문이 없습니다. 서버에서 연결 파일을 다시 발급하세요.")
    if not re.fullmatch(r"device_[a-f0-9]{12}", str(value.get("device", ""))) or not re.fullmatch(r"[A-Za-z0-9_-]{40,100}", str(value.get("token", ""))):
        raise ValueError("올바른 촬영 장치 연결 파일을 사용하세요.")
    return dict(value, url=value["url"].rstrip("/"))


class RemoteError(Exception):
    def __init__(self, status, body):
        self.status, self.body = status, body
        super().__init__(body.get("error", f"서버 응답 {status}"))


class PinnedConnection(http.client.HTTPSConnection):
    def __init__(self,*args,fingerprint,**kwargs):
        self.fingerprint=fingerprint
        super().__init__(*args,**kwargs)

    def connect(self):
        # http.client may reconnect automatically. Pin EVERY new socket, not
        # just the first request on a reused connection.
        super().connect()
        if hashlib.sha256(self.sock.getpeercert(binary_form=True)).hexdigest()!=self.fingerprint:
            self.close()
            raise ValueError("서버 인증서가 연결 파일과 다릅니다. 전송을 중단했습니다.")


class Transport:
    def __init__(self, pair, timeout=15, keep_alive=False):
        self.pair = validate_pair(pair)
        self.timeout = timeout
        self.keep_alive=keep_alive
        self.connection=None

    def close(self):
        if self.connection:self.connection.close();self.connection=None

    def request(self, path, value=None, raw=False):
        if not path.startswith("/v1/") or "\r" in path or "\n" in path:
            raise ValueError("지원하지 않는 서버 요청입니다.")
        parsed = urlsplit(self.pair["url"])
        # Trust is the exact certificate in the locally-issued pairing file.
        # Verify it BEFORE sending the bearer key, request path or video bytes.
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        connection = self.connection or PinnedConnection(parsed.hostname, parsed.port or 443, context=context,
                         timeout=self.timeout,fingerprint=self.pair['certificate_sha256'])
        self.connection=connection
        try:
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if value is not None else None
            connection.request("GET" if value is None else "POST", path, body=body,
                headers={"Authorization": "Bearer " + self.pair["token"], "X-Device": self.pair["device"],
                         "Content-Type": "application/octet-stream" if isinstance(value, bytes) else "application/json"})
            response = connection.getresponse()
            limit = 64*1024*1024 if raw else 8*1024*1024
            content = response.read(limit+1)
            if len(content)>limit:raise ValueError("서버 응답 크기를 확인하세요.")
            if response.status != 200:
                try:detail=json.loads(content)
                except ValueError:detail={"error":f"서버 응답 {response.status}"}
                raise RemoteError(response.status,detail)
            return (content, response.getheader("Content-Type", "application/octet-stream")) if raw else json.loads(content)
        except BaseException:
            self.close()
            raise
        finally:
            if not self.keep_alive:self.close()
