**Posture Corrector** — это дополненная python-программа (https://github.com/TiffinTech/posture-corrector/blob/main/main.py) для анализа и улучшения осанки в реальном времени с использованием камеры и библиотеки [MediaPipe](https://google.github.io/mediapipe/). Программа отслеживает положение головы и плеч, оценивает осанку и отображает визуальную обратную связь на экране.  

## Особенности

- Реальное время: работает с веб-камерой.
- Автоматическая калибровка пользователя.
- Метрики осанки:
  - **Neck compression ratio** — втягивание шеи.
  - **Forward head ratio** — наклон головы вперед.
  - **Lateral head tilt** — наклон головы в стороны.
  - **Torso angle** — угол наклона корпуса.
- Визуализация скелета с помощью MediaPipe.
- Цветовая индикация состояния осанки:
  - Зеленый — хорошая осанка
  - Красный — плохая осанка

## Установка

1. Склонируйте репозиторий:  
```bash
git clone https://github.com/yourusername/posture-corrector.git
cd posture-corrector

Установите зависимости:

pip install opencv-python mediapipe numpy

Запустите скрипт:

python posture_corrector.py

Для выхода нажмите q в окне программы.
