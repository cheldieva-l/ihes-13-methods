# IHES — реестр Molab-запусков

Последнее обновление: 2026-09-10, Москва.

Здесь хранится соответствие `профиль Molab → ноутбук → эксперимент`. Email, пароли,
токены и другие секреты не публикуются. Ссылку проверять под указанным профилем.

## Активные и последние запуски

| Профиль | Эксперимент | Ноутбук | Compute | Статус | Проверенный результат | Следующий шаг |
|---|---|---|---|---|---|---|
| Renuka | E001 T0 smoke p9 B2^14 | [nb_65WC1dSjdUsbupzQFHdWnZ](https://molab.marimo.io/notebooks/nb_65WC1dSjdUsbupzQFHdWnZ) | 4 CPU, 32 GiB, RTX Pro 6000 | завершён | GPU search 3.933 s; length 7; 18 moves; organizer valid; candidate full replay valid | E002 fast-20 |
| Renuka | E002 fast-20, frames 0,40, direct+reverse | [nb_65WC1dSjdUsbupzQFHdWnZ](https://molab.marimo.io/notebooks/nb_65WC1dSjdUsbupzQFHdWnZ) | RTX Pro 6000 | завершён; позже URL отдал 403 | 80 searches, 0 errors, selected 20/20 valid, 0 shorter, 616.875 s, commit `5e9b91f` | восстановить профиль/сессию и запустить p999 trace commit `27757d4` |
| Renuka | E005 p999 score-trace | [nb_65WC1dSjdUsbupzQFHdWnZ](https://molab.marimo.io/notebooks/nb_65WC1dSjdUsbupzQFHdWnZ) | ожидается RTX Pro 6000 | кодовая ячейка подготовлена, запуск не подтверждён; 09:37 URL 403 | артефактов расчёта пока нет | пользователь открывает Molab под Renuka; затем B2^6/10/14 |
| Renuka | запасной fork | [nb_yQ3kfqCGbJcZgWX863LkK9](https://molab.marimo.io/notebooks/nb_yQ3kfqCGbJcZgWX863LkK9) | не перепроверено | не запускать без назначения | — | резерв |
| Chandru | E001 импорт | [nb_Wm5mw3SEBwmeLf62a4Hjb6](https://molab.marimo.io/notebooks/nb_Wm5mw3SEBwmeLf62a4Hjb6) | выбран RTX Pro 6000 | launch 403: GPU quota занята | кодовая ячейка исправна; GPU run не состоялся | повторять только при свободной quota |
| Chandru | старый E001 | [nb_SW9e7mt1PG9jCLyRHA1Agg](https://molab.marimo.io/notebooks/nb_SW9e7mt1PG9jCLyRHA1Agg) | прежний sandbox | остановлен | — | не использовать |
| Chandru | старый beam 67M | [nb_E9bq8C6kvmGZwd6DouhfNv](https://molab.marimo.io/notebooks/nb_E9bq8C6kvmGZwd6DouhfNv) | прежний sandbox | остановлен | stale output | не использовать |
| Chandru | GPU-шаблон | [nb_jJxDboLKcKBAbNpommx1fJ](https://molab.marimo.io/notebooks/nb_jJxDboLKcKBAbNpommx1fJ) | RTX Pro 6000 | шаблон | — | использовать при свободной quota |
| Chandru | редактируемый GPU fork | [nb_dUmThXaJsiiMPDjt5qbjsU](https://molab.marimo.io/notebooks/nb_dUmThXaJsiiMPDjt5qbjsU) | не перепроверено | резерв | — | назначить T1 после 08:00 |

## Формат новой строки

```text
profile,experiment_id,notebook_url,git_sha,compute,started_at,finished_at,
status,puzzle_set,beam,model,valid,score_or_length,gpu_seconds,output,next_step
```

## Правила

- Сначала найти профиль в таблице, затем открывать ссылку.
- Если quota занята или версия выполняется, записать фактический статус; другие
  независимые эксперименты продолжать.
- Каждый вычислительный запуск должен иметь experiment id и git SHA.
- Результат считается полезным только после full replay.
- После создания/запуска нового ноутбука сразу добавить строку в этот файл и в `мне.md`.
