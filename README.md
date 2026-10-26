# Guía de Trabajos IA (Redes Neuronales y Vibe Coding)

Monorepo con los 8 trabajos. Se desarrolla en el escritorio, se sube a GitHub y se clona en la notebook.

## Los trabajos

| # | Carpeta | Qué hace | Motor |
|---|---|---|---|
| 1 | `t1_cnn` | CNN: entrena con CIFAR-10 o tus imágenes, predice | PyTorch (CUDA) |
| 2 | `t2_rnn_audio_manuscrito` | Audio y manuscrito a texto + traducción | Whisper, TrOCR, Llama |
| 3 | `t3_gan` | GAN que genera imágenes + **benchmark CPU vs GPU** | PyTorch (CUDA) |
| 4 | `t4_transformers_sentimiento` | Sentimiento de textos (.txt/.csv) | Transformer HF |
| 5 | `t5_imagenes_animales` | Fotos de animales a tabla `animales` | PyTorch + Llama visión |
| 6 | `t6_audio_comentarios` | Audio de comentarios a tabla `comentarios` | Whisper + Llama |
| 7 | `t7_texto` | Texto de comentarios a tabla `comentarios` | Llama |
| 8 | `t8_pdf` | PDF a tabla `pdfs` (resumen) | PyMuPDF + Llama |

Nota sobre el trabajo 2: la guía dice RNN, pero Whisper y TrOCR son Transformers (los modelos que hoy hacen esto mejor). Confirmá con el profesor si lo acepta; si exige una RNN, se puede sumar un CRNN (CNN+LSTM) para el manuscrito.

## Primera vez en cada máquina

Windows: `.\setup.ps1`  ·  Linux: `bash setup.sh`. Después, con el entorno activado:

```
pip install -r requirements-torch.txt     # PyTorch con CUDA
pip install -r requirements-ml.txt        # transformers, audio, PDF
ollama pull llama3.1:8b                   # escritorio  (notebook: llama3.2:3b)
ollama pull gemma3:4b                     # modelo de visión (trabajo 5)
```

Editá `.env` con el modelo de esa máquina y verificá todo con `python scripts/check_env.py`.
La primera vez que se usa cada modelo de Hugging Face (Whisper ~1 GB, sentimiento ~1 GB, TrOCR ~1 GB) se descarga solo; hace falta internet.

## Ejecutar un trabajo

```
python run.py t3_gan          # abrí http://127.0.0.1:8000
```

Solo corre uno por vez en el puerto 8000 (o usá `--port 8001`).
