# 第三方材料与依赖说明

本地157第一版候选；未公开发布。原创源码沿用根目录MIT许可证。Go2的XML/网格来自MuJoCo Menagerie，保留`third_party/unitree_go2/LICENSE`、README与SOURCE.json逐文件来源锁。上游材料不声明为项目原创。

源码Git树不收录37个wheel、Python解释器或虚拟环境。P2离线wheel仅为本地同版本验证缓存；将来若分发二进制附件，应以实际构建的完整随附许可为准。本清单记录实际锁定版本元数据，不能用简化表替代上游原文。

实际imageio-ffmpeg附带FFmpeg7.1，`-version`显示`--enable-gpl --enable-version3`，`-L`显示GPLv3-or-later；其Python包装层METADATA的BSD-2-Clause不能代表该二进制。该wheel及其它依赖wheel均未列入本次公开附件候选。实际输出证据在169本地记录留存。

|实际依赖|锁定版本|元数据许可（主许可文件见JSON）|
|---|---|---|
|absl-py|2.5.1|Apache-2.0|
|cloudpickle|3.1.2|BSD-3-Clause|
|contourpy|1.4.0|BSD-3-Clause|
|cycler|0.12.1|Copyright (c) 2015, matplotlib project         All rights reserved.                  Redistribution and use in source and binary forms, with or without         modification, are permitted provided that the f...|
|etils|1.14.0|License :: OSI Approved :: Apache Software License|
|Farama-Notifications|0.0.6|License :: OSI Approved :: MIT License|
|filelock|4.2.4|MIT|
|fonttools|4.67.0|MIT|
|fsspec|2026.9.0|BSD-3-Clause|
|glfw|2.11.0|MIT|
|gymnasium|1.4.0|MIT License|
|ImageIO|2.38.1|BSD-2-Clause|
|imageio-ffmpeg|0.6.0|BSD-2-Clause|
|Jinja2|3.1.6|License :: OSI Approved :: BSD License|
|kiwisolver|1.5.1|=========================          The Kiwi licensing terms         =========================         Kiwi is licensed under the terms of the Modified BSD License (also known as         New or Revised BS...|
|MarkupSafe|3.0.4|BSD-3-Clause|
|matplotlib|3.11.2|License agreement for matplotlib versions 1.3.0 and later          =========================================================                    1. This LICENSE AGREEMENT is between the Matplotlib Development...|
|mpmath|1.3.0|BSD|
|mujoco|3.16.0|Apache-2.0|
|networkx|3.7|BSD-3-Clause|
|numpy|2.5.4|BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0|
|packaging|26.3|Apache-2.0 OR BSD-2-Clause|
|pillow|12.3.0|MIT-CMU|
|psutil|7.2.2|BSD-3-Clause|
|pygame|2.6.1|LGPL|
|PyOpenGL|3.1.10|License :: OSI Approved :: BSD License|
|pyparsing|3.3.3|MIT|
|python-dateutil|3.0.0|Dual License|
|setuptools|84.2.1|MIT|
|six|1.17.0|MIT|
|stable_baselines3|2.9.0|MIT|
|sympy|1.14.0|BSD|
|torch|2.14.1|Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT|
|typing_extensions|4.16.0|PSF-2.0|
|tzdata|2026.5|Apache-2.0|
|websockets|17.3|BSD-3-Clause|
|zipp|4.1.2|MIT|

模型与实验数据为本项目训练/仿真产生的附件候选，来源和SHA见附件清单。历史源对话、机器清单、venv、pip缓存和原Git暂存区不纳入源码树；公开发布和附件许可确认仍在最终发布步骤执行。
