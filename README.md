# Action Recognition with X3D

Проект по **распознаванию действий на видео** на основе архитектуры **X3D (PyTorchVideo)**.  
Включает полный пайплайн: **подготовка датасета → обучение → онлайн и офлайн инференс**.

---

##  Demo

Ниже показан пример работы модели на видео с наложенными предсказаниями действий:

![Action Recognition Demo](assets/demo.gif)

---

##  О проекте

- Модель: **X3D (spatiotemporal CNN)**
- Формат инференса: sliding window по видео
- Поддержка **CPU / GPU**
- Реализованы:
  - онлайн-инференс (real-time)
  - офлайн-инференс (video → video)
  - сглаживание предсказаний
  - контроль задержки реакции модели

Проект ориентирован на **реалистичное применение action recognition**, а не только офлайн-оценку.

---

## Закачка библиотек

```
pip install -r requirements.txt
```

##  Подготовка датасета

Код подготовки датасета:

```bash
python src/data/prepare_dataset.py
```

---

## Инференс

### Онлайн (real-time)
```bash
python inf/infer_online.py
```

### Офлайн (video → video)
```bash
python inf/infer_video.py
```


