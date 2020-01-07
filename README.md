#  Бэйслайн к задаче [RetailHero.ai/#2](https://retailhero.ai/c/recommender_system/overview) от [@geffy](https://github.com/geffy)

Репозиторий содержит:
* item-to-item модель (NMAP 0.1137, top5 на 09/01/2020)
* распиливание исходных данных на шарды
* вспомогательный переиспользуемый код 
* скрипты и для обучения кастомных эмбеддингов на pytorch 
* быстрый поиск соседей в связке с faiss
* кастомный docker-образ с поддержой pytorch 1.3 и faiss

Код написан так, что вполне успешно отрабатывает на машине с 8gb ram.

## Шаги по подготовке:

0. Скопировать данные в data/raw
```
cd {REPO_ROOT}
mkdir -p data/raw
cp /path/to/upacked/data/*.csv ./data/raw
cd src
```


1. Разделить исходные данные о покупках на 16 частей
```bash
python3 purchases_to_jrows.py
```


2. Подготовить train/valid данные в формате, максимально близком к формату `check_queries.tsv`
```bash
python3 train_valid_split.py
```

3. Обучить item-2-item модель:
```bash
python3 train_i2i_model.py
```

4. Скопировать артефакты в сабмит
```bash
cd {REPO_ROOT}
mkdir -p submit/solutions/assets
cp ./data/raw/products.csv submit/solutions/assets
cp ./tmp/implicit_cosine1/model.pkl submit/solutions/assets
```

5. Упаковать сабмит
```bash
cd submit
zip -r submit_title.zip solution/*
```

6. Profit!

## Результаты: 
```
Check: 0,1113
Public: 0,1137
```

Обучение кастомных эмбеддингов в текущем решении фактически не используется, их код  оставлен для экспериментов.
