# IHES 21840 — управление проектом

Последнее обновление: 2026-09-10, Москва.

## Definition of Done

До 22 сентября 2026 года существует CSV на 1003 строки, каждый путь независимо
проигран всеми 72 позициями до официального solved state, сумма длин ≤21840, score
пересчитан двумя независимыми реализациями, файл отправлен в соревнование и Kaggle
принял submission без invalid ids.

## Критический путь

```text
validator + baseline 21870 (DONE)
  -> GPU smoke E001 (DONE)
  -> T0 fast-20 curve + p999 score trace
  -> параллельные gates: S1/S2 | U2 universal solver | T1 Bellman Transformer
  -> paired selection-50
  -> дорогой inference только 1–2 победителей
  -> min merge valid-only
  -> двойной full replay + score
  -> Kaggle submission не позже 22 сентября
```

Любая задержка T0 curve, center coordinate или selection-50 сокращает финальный GPU
бюджет. Визуализации, новый UI и широкие рефакторинги не находятся на critical path.

## Правила управления

- WIP limit = 3: один inference, один training, один engineering/analysis.
- Каждый work item ≤5 часов Codex или ≤12 часов GPU; больший делится на checkpoints.
- У каждого эксперимента заранее есть гипотеза, baseline, бюджет, metric и stop rule.
- Все модели сравниваются paired на фиксированных 20, затем 50 кубиках.
- Плохой результат записывается так же тщательно, как хороший, чтобы не повторять.
- Каждые 24 часа пересчитывается critical path, GPU shortfall и вероятность цели.
- Любая новая идея получает быструю оценку: ожидаемый gain, confidence, GPU-ч,
  engineering-ч, dependency и самый дешёвый falsification test.

## Stage gates

| Gate | Дедлайн | Вход | Pass | Fail action |
|---|---|---|---|---|
| G0 Infrastructure | 10 Sep | E001 | p9 valid на GPU | исправлять только pipeline |
| G1 Search baseline | 11 Sep | E002–E004 + p999 trace | paired таблица и место потери пути | не запускать большой beam вслепую |
| G2 Cheap improvement | 12 Sep | S1 algebra/BFS rewrite | ≥1 valid shorter path | перейти к S2/MITM |
| G3 Center solver | 14 Sep | U2 coordinate/PDB | корректные transitions + первый full lift | оставить U2 как heuristic feature |
| G4 Model pilot | 15 Sep | T1 checkpoints | независимые wins или negative delta fast-20 | остановить/изменить loss, не масштабировать |
| G5 Selection | 17 Sep | все победители | selection-50 paired evidence | выбрать лучший risk-adjusted метод |
| G6 Production | 20 Sep | 1–2 метода | суммарный valid gain ≥30 | targeted tail + S2 reserve |
| G7 Freeze | 21 Sep | candidate CSVs | double replay, score ≤21840 | чинить только invalid/merge |
| G8 Submit | 22 Sep | final CSV | Kaggle accepted | использовать последний valid fallback |

## Rolling-wave план

| Дата | Главный deliverable | Параллельная ветка | Резервное решение |
|---|---|---|---|
| 10 Sep | E001 done; S1 cheap reduction; E002 ready | internet/code research | baseline 21870 сохранён |
| 11 Sep | T0 curve + p999 score trace | U2 coordinate start | direct/reverse identity only |
| 12 Sep | Q1/Q3 selector test; S1/S2 result | T1 data/loss pipeline | обычный global beam |
| 13–14 Sep | U2 center transition/PDB + first lift | T1 train 2k/4k | use U2 only as feature |
| 15–17 Sep | paired fast-20/selection-50; freeze winners | targeted S2 | best measured incumbent |
| 18–20 Sep | production beams/targeted long paths | final postprocess | stop weak branches |
| 21 Sep | merge, two validators, submission rehearsal | buffer | no new architecture |
| 22 Sep | final submit by midday Moscow | repair window | last full-valid file |

## Риск-реестр

Оценка exposure = probability × impact по шкале 1–5.

| Risk | P | I | Exposure | Ранний сигнал | Mitigation | Owner |
|---|---:|---:|---:|---|---|---|
| Center orientation ломает 3x3 lift | 4 | 5 | 20 | quotient solved, full replay false | U2 2048-fiber/PDB + kernel macros | Codex |
| GPU quota/профиль недоступен | 4 | 4 | 16 | 403/disconnect | две очереди, два профиля, Kaggle fallback | Codex + пользователь |
| Transformer не лучше MLP | 3 | 4 | 12 | no paired wins at 2k/4k | stop early; blend only if independent wins | Codex |
| Большой beam тратит часы без shorter paths | 3 | 4 | 12 | incumbent rank already far outside cutoff | score trace; Q1/Q3/BWAS before 2^21 | Codex |
| Статистический шум на 20 кубиках | 4 | 3 | 12 | many ties, wide CI | selection-50, sign/bootstrap + t-test | Codex |
| Ошибка path convention/merge | 2 | 5 | 10 | invalid replay | organizer calibration + double validator | Codex |
| Codex 5-hour cycles расходуются на ожидание | 3 | 4 | 12 | нет артефакта за цикл | resource ledger; CPU/U2 queue while GPU runs | Codex |
| Пользователь недоступен до 08:00 | 5 | 2 | 10 | нельзя сменить профиль | работать только в Renuka/локально; не блокироваться | Codex |

## Текущая доска (WIP=3)

| Статус | Work item | Owner | Stop/Done criterion |
|---|---|---|---|
| DONE | S001 axis/loop exact reductions baseline 21870 | Codex/local CPU | 1003/1003 valid; gain 0; не повторять |
| READY | S1 BFS-window/MITM rewrite baseline 21870 | Codex/local CPU | ≥1 full-replay-valid shorter path |
| READY | E002 T0 symmetry/reverse fast-20 B2^14 | Codex/Renuka GPU | 20-row paired result |
| READY | p999 score trace B2^6/10/14 | Codex/Renuka GPU | cutoff/rank loss depth known |
| BACKLOG | U2 center fiber/transitions | Codex/local CPU | reachable states + table |
| BACKLOG | T1 Bellman Transformer | Codex/second GPU | checkpoint paired test |
| BLOCKED UNTIL 08:00 | second Molab profile | пользователь | profile open + quota numbers |
| DONE | validator + baseline | Codex | 1003/1003, score 21870 |
| DONE | E001 | Codex/Renuka GPU | p9 length 7 valid, 3.933 s |

## Как добавляется новый метод

Записать одну строку в comparison journal:

```text
idea,mechanism,expected_gain,confidence,gpu_hours,engineering_hours,
dependency,smallest_test,stop_rule,result,next_decision
```

Приоритет = `(expected_gain × confidence × information_value) / scarce_resource_cost`.
Высокий upside не заменяет короткий falsification test. Каждый день искать минимум
один новый классический и один ML/search кандидат, но WIP не увеличивать.

## Задачи пользователю

1. С 08:00 быть готовой переключить Molab на конкретный профиль только после моего
   сигнала `!` и голосовой фразы.
2. Сообщить остаток GPU/quota и время reset для Renuka и Chandru без секретов.
3. Прислать скрин с точным временем, если Molab disconnect/403/лимит.
4. Читать публичные журналы и указывать, если приоритет кажется неверным; финальное
   техническое решение и ответственность за critical path остаются у Codex.

## Ссылки управления

- PMI critical path: https://www.pmi.org/-/media/pmi/documents/public/pdf/pmief/skills-for-life-english.pdf
- PMI resource leveling: https://www.pmi.org/learning/library/scheduling-resource-leveling-project-progression-8006
- WIP limits: https://www.atlassian.com/agile/kanban/wip-limits/
- NASA risk-informed planning/register: https://lmse.larc.nasa.gov/admin/public_docs/LMS_CP_8000.4_RevB_FINAL.pdf
