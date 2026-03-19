# AutoReport Chain (LangChain + ProxyAPI)

Учебный проект для ДЗ: цепочка из 4 этапов для подготовки отчета по CSV-данным.

## Что делает проект

Скрипт строит отчет в `markdown` по таблице `CSV` через 4 звена:

1. Анализ задачи
2. Подбор инструментов
3. Генерация результата (отчета)
4. Проверка / ревью

Промежуточные результаты сохраняются в папку `artifacts/`.

## Стек

- Python
- LangChain (`langchain-core`)
- OpenAI Python SDK (через OpenAI-совместимый endpoint ProxyAPI)
- pandas

## Подготовка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Заполни `OPENAI_API_KEY` в `.env`.

## Запуск

```bash
python3 script.py "Подготовь отчет по русскоязычному датасету продаж: ключевые метрики, сегменты по региону и категории, риски качества данных и рекомендации" --csv data/sample_sales.csv --out report.md
```

После запуска:

- итоговый отчет: `report.md`
- промежуточные артефакты: `artifacts/profile.json`, `artifacts/analysis.json`, `artifacts/tools.json`, `artifacts/draft_report.md`

## Пример результата

В репозитории уже есть пример сгенерированного отчета:

- [`report.md`](./report.md)

## Структура проекта

```text
.
├── data/
│   └── sample_sales.csv
├── artifacts/                 # создается после запуска
├── script.py
├── requirements.txt
├── .env.example
└── README.md
```

## Идея цепочки

- `analysis_chain`: из цели и профиля датасета формирует план анализа (JSON).
- `tools_chain`: подбирает библиотеки, расчеты и проверки (JSON).
- `generation_chain`: генерирует `markdown`-отчет.
- `review_chain`: проверяет структуру и фактологию отчета и улучшает финальную версию.
