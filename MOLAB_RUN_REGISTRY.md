# IHES — реестр Molab-запусков

## Аудит 10 сентября, 15:35 МСК

| Профиль | Ноутбук | Наблюдение | Вывод |
|---|---|---|---|
| Liuda | [nb_fxcxVdJ1ucCsTU85XyfDZU](https://molab.marimo.io/notebooks/nb_fxcxVdJ1ucCsTU85XyfDZU) | В списке Running, при открытии Sandbox capacity is currently exhausted | Прогресс/experiment неизвестны; не перезапускать по этому сигналу |
| Renuka | [E009](https://molab.marimo.io/notebooks/nb_Yyuy9RzfsXUKp2CK12oxMq) | Read-only preview, прежние 2 строки p983/p984, live-таймера нет | Current state unknown; данные не являются свежим завершением |

Из этого аудита нельзя получить число реально вычисляющих GPU. На Liuda
ноутбук найден уже существующим; нового вычислительного кода не запускали.

## Последние известные запуски — обновление 10 сентября, 15:25 МСК

| Профиль / слот | Эксперимент | Ссылка | Последняя проверка | Текущий статус |
|---|---|---|---|---|
| Renuka / 1 | E003, fast-20, beam 2^16, frames 0/40 direct+reverse | [Открыть ноутбук](https://molab.marimo.io/notebooks/nb_65WC1dSjdUsbupzQFHdWnZ) | 12:14 МСК: выполнялся, первые три строки без сокращений | Не перепроверен; не утверждаем, что продолжает работать |
| Renuka / 2 | E009, fast-20, 48 frames, beam 2^14 | [Открыть ноутбук](https://molab.marimo.io/notebooks/nb_Yyuy9RzfsXUKp2CK12oxMq) | 12:14 МСК: выполнялся, первые две строки без сокращений | Не перепроверен; не утверждаем, что продолжает работать |

Открывать под Renuka. E010/E011 и обработчики ещё не запущены. Плановая ёмкость
обновлена до 14 уникальных профилей / 28 слотов; [реестр](ops/fleet.json).
Старые статусы ниже — история. [Текущий план](IHES_EXECUTION_PLAN.md).

Последнее обновление: 2026-09-10, Москва.

Здесь хранится соответствие `профиль Molab → ноутбук → эксперимент`. Email, пароли,
токены и другие секреты не публикуются. Ссылку проверять под указанным профилем.

Плановые профили: `Liuda`, `Chandru`, `Renuka`, `Satyanarayan`, `Vivek`; у каждого
заявлено по два одновременно работающих GPU-слота. Общего лимита часов нет; каждая
сессия живёт до 12 часов, после чего её можно немедленно заменить новой. Цель — держать
10/10 слотов занятыми полезными независимыми задачами круглосуточно. Двухчасовой полный
статус хранится в `MOLAB_GPU_CAPACITY.csv`.

## Активные и последние запуски

| Профиль | Эксперимент | Ноутбук | Compute | Статус | Проверенный результат | Следующий шаг |
|---|---|---|---|---|---|---|
| Renuka | E001 T0 smoke p9 B2^14 | [nb_65WC1dSjdUsbupzQFHdWnZ](https://molab.marimo.io/notebooks/nb_65WC1dSjdUsbupzQFHdWnZ) | 4 CPU, 32 GiB, RTX Pro 6000 | завершён | GPU search 3.933 s; length 7; 18 moves; organizer valid; candidate full replay valid | E002 fast-20 |
| Renuka | E002 fast-20, frames 0,40, direct+reverse | [nb_65WC1dSjdUsbupzQFHdWnZ](https://molab.marimo.io/notebooks/nb_65WC1dSjdUsbupzQFHdWnZ) | RTX Pro 6000 | завершён; позже URL отдал 403 | 80 searches, 0 errors, selected 20/20 valid, 0 shorter, 616.875 s, commit `5e9b91f` | восстановить профиль/сессию и запустить p999 trace commit `27757d4` |
| Renuka | E005 p999 score-trace | [nb_65WC1dSjdUsbupzQFHdWnZ](https://molab.marimo.io/notebooks/nb_65WC1dSjdUsbupzQFHdWnZ) | RTX Pro 6000 Blackwell | завершён 09:49; commit `27757d4` | incumbent 23 valid; B2^6/10/14 drop depth 3/4/5; 0 valid candidates; 12.081 s | Q3 root quota, затем limited lookahead rerank |
| Renuka | E006 p999 equal root quotas | [nb_65WC1dSjdUsbupzQFHdWnZ](https://molab.marimo.io/notebooks/nb_65WC1dSjdUsbupzQFHdWnZ) | RTX Pro 6000 Blackwell | завершён 09:57; commit `4251030`; tests 12/12 | B2^10/14 drop depth 3/4; root12 rank exceeds equal quota; 8.119 s | Q3 stop; E007 limited lookahead |
| Renuka | E007 p999 pool 8B + one-step backup | [nb_65WC1dSjdUsbupzQFHdWnZ](https://molab.marimo.io/notebooks/nb_65WC1dSjdUsbupzQFHdWnZ) | RTX Pro 6000 Blackwell | завершён 10:08; commits `10d2a35`,`9062aff`; tests 13/13 | pure drop d4; adaptive drop d5; rank 96,153 > B; 76.511 s | stop E007; U2/T1 |
| Renuka | E008 p999 suffix splice | [nb_65WC1dSjdUsbupzQFHdWnZ](https://molab.marimo.io/notebooks/nb_65WC1dSjdUsbupzQFHdWnZ) | RTX Pro 6000 Blackwell | завершён 11:46 | 10/10 searches, errors 0, valid shorter candidates 0, 43.506 s | stop E008 |
| Renuka | E003 fast-20 B2^16 | [nb_65WC1dSjdUsbupzQFHdWnZ](https://molab.marimo.io/notebooks/nb_65WC1dSjdUsbupzQFHdWnZ) | RTX Pro 6000 Blackwell | running с 11:47 | pending | проверить примерно в 12:30 |
| Renuka | E009 fast-20 all 48 frames B2^14 | [nb_Yyuy9RzfsXUKp2CK12oxMq](https://molab.marimo.io/notebooks/nb_Yyuy9RzfsXUKp2CK12oxMq) | RTX Pro 6000 Blackwell | running с 11:51 | pending | проверить по завершении |
| Liuda | E010 T1 PieceTransformer seed42 K23 | URL после запуска | RTX Pro 6000 | queued | restartable checkpoints each epoch; eval each 10 | запустить в slot 1 |
| Liuda | E011 T1 PieceTransformer seed42 K40 | URL после запуска | RTX Pro 6000 | queued | отличается от E010 только глубиной RW curriculum | запустить в slot 2 |
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

Molab launcher для E010/E011: `tools/molab_transformer_train.py`. Он закрепляет
training core commit `5a72174`, сохраняет checkpoints в `/marimo/ihes_runs/<run_id>`
каждую эпоху и автоматически продолжает с `_latest.pt` после замены 12-часовой сессии.

## Правила

- Сначала найти профиль в таблице, затем открывать ссылку.
- Если quota занята или версия выполняется, записать фактический статус; другие
  независимые эксперименты продолжать.
- Подтверждённо свободный слот не оставлять без работы: сразу брать следующий run из очереди.
- За 30–60 минут до 12-часового отключения сохранить checkpoint/output; после отключения
  немедленно запустить replacement.
- Каждый вычислительный запуск должен иметь experiment id и git SHA.
- Результат считается полезным только после full replay.
- После создания/запуска нового ноутбука сразу добавить строку в этот файл и в `мне.md`.
