# Diploma RL Trading Final

В папке находится итоговая версия дипломной работы по теме:

**«Применение методов обучения с подкреплением (Reinforcement Learning, RL) для прогнозирования и оптимизации биржевых котировок»**

## Основные файлы

- `diploma_rl_trading.docx` — итоговый файл в формате DOCX;
- `main.pdf` — итоговый PDF после сборки LaTeX-версии;
- `main.tex` — основной LaTeX-файл;
- `chapters/` — главы и служебные части документа;
- `figures/` — рисунки;
- `tables/` — таблицы;
- `REPORT.md` — отчет о внесенных правках и замечаниях для ручной проверки.

## Как открыть итоговый документ

Откройте файл `diploma_rl_trading.docx` в Microsoft Word, LibreOffice Writer или Pages.

## Как собрать PDF заново

На macOS или Linux при установленном TeX Live:

```bash
cd diploma_rl_trading_final
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

Повторный запуск нужен для корректного формирования оглавления и внутренних ссылок.

## Как пересобрать DOCX

Итоговый DOCX собран как визуально точная Word-версия финального PDF. Для пересборки:

```bash
cd diploma_rl_trading_final
gs -dNOPAUSE -dBATCH -sDEVICE=jpeg -dJPEGQ=90 -r150 -sOutputFile=docx_pages/page-%03d.jpg main.pdf
python3 build_docx_from_pdf_pages.py
```

Такой способ выбран для сохранения оформления, таблиц, рисунков, формул и нумерации страниц без искажений при конвертации.
