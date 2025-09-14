# Car Condition Classifier

Мульти-лейбл классификатор на PyTorch для определения состояния автомобиля по фото: clean/dirty и intact/damaged.

## Быстрый старт
1. Установите зависимости: `pip install -r requirements.txt`
2. Положите изображения в `data/images/` и создайте `data/labels.csv` (см. labels_example.csv).
3. Тренировка:  
   `python src/train.py --data_csv data/labels.csv --images_dir data/images --output_dir models/` 
4. Демо:  
   `streamlit run app/app_streamlit.py`

## Лицензии
Используйте только разрешённые датасеты и замазывайте номера.

## Ограничения
Модель чувствительна к освещению и ракурсу. Используйте как подсказку, а не окончательное решение.
