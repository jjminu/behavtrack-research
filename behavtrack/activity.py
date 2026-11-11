"""Process-released recording/training exclusion, including Windows CLI jobs."""
from __future__ import annotations
import functools
import os
import threading
from pathlib import Path

_mutex=threading.RLock()
_held={}


class Lease:
    def __init__(self,root,kind):
        self.path=Path(root).resolve()/".capture_training.lock"
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.closed=False
        with _mutex:
            if self.path in _held:
                item=_held[self.path]
                if kind==item["kind"]=="heavy":
                    item["count"]+=1
                    return
                raise ValueError("녹화와 학습·분석은 동시에 실행할 수 없습니다. 진행 중인 작업을 종료하세요.")
            f=self.path.open("a+b")
            f.seek(0,2)
            if f.tell()==0:
                f.write(b"0"); f.flush()
            f.seek(0)
            try:
                if os.name=="nt":
                    import msvcrt
                    msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError:
                f.close()
                raise ValueError("다른 BehavTrack 프로세스가 녹화 또는 학습 중입니다. 먼저 종료하세요.") from None
            _held[self.path]={"file":f,"kind":kind,"count":1}

    def close(self):
        with _mutex:
            if self.closed:
                return
            self.closed=True
            item=_held[self.path];item["count"]-=1
            if item["count"]:
                return
            item["file"].close()
            del _held[self.path]

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()


def heavy(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        root=Path(__file__).resolve().parents[1]
        if root.name in ("program","개발자료") and (root.parent/"program").is_dir() and (root.parent/"results").is_dir():
            root=root.parent
        with Lease(root/"results","heavy"):
            return function(*args,**kwargs)
    return wrapped
