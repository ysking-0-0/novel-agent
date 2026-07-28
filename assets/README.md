# BGM 背景音乐目录

将背景音乐文件放在这里，命名为 `bgm.mp3`。

## 配置

`config.json` 中 `media.bgm_path` 默认指向 `./assets/bgm.mp3`。

- `bgm_volume`: BGM 音量（0.0-1.0，相对 TTS）。默认 0.15（15%）。
- BGM 会被循环/截断对齐视频时长，音量降低做背景。

## 注意

`assets/*.mp3` 等音频文件已在 .gitignore 中，不会被提交（体积大+版权）。请自行准备 BGM 文件放入此目录。
