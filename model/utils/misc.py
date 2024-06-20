from typing import Union, Optional

import os
import glob


def singleton(klass):
    # IDK how to type this decorator :(
    instances = {}
    def getinstance(*args, **kw):
        if klass not in instances:
            instances[klass] = klass(*args, **kw)
        return instances[klass]
    return getinstance


def latest_checkpoint(ckpt_path: str, hint: Optional[str] = None) -> Union[str, None]:
    ckpt_path = os.path.expanduser(ckpt_path)
    files = glob.glob(os.path.join(ckpt_path, '*.pt' if hint is None else f'{hint}*.pt'))
    if len(files) > 0:
        files = sorted(files, key=lambda t: os.stat(t).st_mtime)
        print(f'Found pretrained checkpoint {files[-1]}, loading')
        return files[-1]
    return None
