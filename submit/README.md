## Основная директория для сбора и тестирования решения.

Комментарии к сборке кастомного образа можно найти в `custom_docker`.

### Для локального тестирования:
```bash
cd solution
docker run \
    -v `pwd`:/workspace \
    -v `realpath ../../src`:/workspace/src \
    -w /workspace \
    -p 8000:8000 \
    geffy/ds-base:retailhero \
    gunicorn --bind 0.0.0.0:8000 server:app
```

После этого, в соседнем терминале, нужно запустить прострел тестовыми запросами:
```bash
python3 run_queries.py http://localhost:8000/recommend data/check_queries.tsv
```

### Замечания
При каждом изменении кода локальный контейнер нужно оставливать (ctrl+c) и перезапускать.

Перед сабмитом важно не забыть убедиться, что в файле `solution/metadata.json` указан тот же образ, что использовался для локального тестирования.
