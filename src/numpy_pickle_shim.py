# -*- coding: utf-8 -*-
"""
兼容性垫片：让 numpy 2.x 序列化出的 pickle（内部引用 ``numpy._core``）能在
numpy < 2.x 环境下被 ``pickle.load`` 正常反序列化。

背景：``metadata.pkl`` 等文件当初在 numpy 2.x 环境保存，其数组重建路径指向
``numpy._core.multiarray``；而 numpy 1.x 中该包名为 ``numpy.core``。本模块在
import 时把 ``numpy._core`` 别名到 ``numpy.core``，从而无需改动数据文件即可加载。
在 numpy >= 2.0 下本模块为空操作（numpy._core 已存在）。

用法：在任何 ``pickle.load(metadata.pkl)`` 之前 ``import numpy_pickle_shim``。
"""

import sys
import numpy as np

if not hasattr(np, "_core"):
    # numpy < 2.0：建立 numpy._core -> numpy.core 别名
    np._core = np.core
    sys.modules.setdefault("numpy._core", np.core)
    for _sub in ("multiarray", "numeric", "umath", "fromnumeric",
                 "_multiarray_umath", "_dtype", "arrayprint", "_exceptions"):
        _sm = "numpy._core." + _sub
        if _sm not in sys.modules:
            try:
                sys.modules[_sm] = getattr(np.core, _sub)
            except AttributeError:
                pass
