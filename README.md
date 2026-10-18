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

## Presentación: trabajo 3 (CUDA)

La pantalla muestra en vivo el uso de GPU, VRAM, temperatura y consumo. Guion sugerido:
1. Hardware: núcleos CUDA y SMs de la GPU.
2. Entrenar la GAN con MNIST: se ven las imágenes mejorar época a época mientras sube el uso de GPU.
3. Generar imágenes aleatorias (la misma semilla da las mismas imágenes).
4. Benchmark: imágenes por segundo en CPU contra GPU, y TFLOPS en FP32 contra los Tensor Cores (TF32/FP16/BF16).

## Flujo escritorio a notebook

```
git add . && git commit -m "avance" && git push     # escritorio
git pull                                            # notebook
```

El `.env`, el entorno virtual y `data/` no se suben. Los pesos entrenados chicos de `weights/` sí.
En la notebook (6 GB de VRAM) usá `llama3.2:3b` y batch más chico si hace falta.

## Estado

Los 8 trabajos están implementados. Probados con modelos simulados y datos sintéticos en CPU; la ejecución real con la GPU, Whisper, TrOCR y los modelos de Ollama se valida en cada máquina.
