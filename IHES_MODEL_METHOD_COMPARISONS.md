# IHES — сравнение моделей и методов

Последнее обновление: 2026-09-10, Москва.

Цель этого журнала — выбрать методы, которые сокращают валидный submission 21870
до 21840 или ниже. Результат существует только после полного replay 72 позиций.

## Неизменяемые выборки

- `fast-20`: puzzle_id 983..1002.
- `selection-50`: puzzle_id 953..1002.
- Кубики меньше 50 не участвуют в выборе модели.
- Один и тот же incumbent, набор frames, лимит узлов и post-processing для paired run.
- Основные beam: `2^14`, `2^16`, `2^18`; `2^21` только победителям или диагностике p999.

## Статистика

Для каждого кубика считаем `delta = candidate_length - incumbent_length`; меньше нуля
лучше. Отчёт обязан содержать:

- valid/solved, сумму и среднее delta;
- wins/ties/losses и exact sign test по non-ties;
- paired t-test среднего delta (как просил пользователь);
- paired bootstrap 95% CI среднего и суммарного delta;
- median delta и worst regression;
- GPU-seconds, expanded nodes, nodes/s и peak VRAM.

Основное решение принимается по paired delta + bootstrap/sign test, потому что длины
дискретны и дают много ties. T-test сохраняется, но не используется один. При
множественных сравнениях p-value корректируются Holm method.

## Матрица результатов

| Run | Модель/метод | Выборка | Beam / nodes | Valid | Wins/Ties/Losses | Sum delta | Mean delta | Paired t p | Sign p | Bootstrap 95% CI | GPU time | Решение |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---|---:|---|
| E001 | T0 MLP `1778521793` smoke | p9 | 2^14 | 1/1 | — | 0 | 0 | — | — | — | 3.933 s | инфраструктура исправна |
| S001 | exact axis reduction + exact-state loop removal | все 1003 | CPU full replay | 1003/1003 | 0/1003/0 | 0 | 0 | — | — | — | <1 s CPU | дешёвые тождества исчерпаны; перейти к BFS-window |
| S002 | BFS-d5 (790,588 states) + window/2-step rewrite | все 1003 | CPU full replay | 1003/1003 | 0/1003/0 | 0 | 0 | — | — | — | 29.1 s CPU | incumbent уже насыщен BFS-d5; не повторять |
| E002 | T0 MLP + frames 0,40 × direct/reverse | fast-20 | 2^14; 80 searches | 20/20 selected; 0 search solutions | 0/20/0 | 0 | 0 | n/a, all ties | n/a, no non-ties | [0,0] | 616.875 s | эти frames/beam не масштабировать; сначала score-trace |
| E005 | T0 natural global beam score-trace | p999 | 2^6/2^10/2^14 | incumbent 23 valid; search 0/3 | — | — | — | — | — | — | 0.149/0.375/9.092 s | incumbent drop depth 3/4/5; global score ранжирует путь слишком низко |
| E003 | T0 MLP + symmetry/reverse | fast-20 | 2^16 | pending | pending | pending | pending | pending | pending | pending | pending | очередь |
| E010 | T1 PieceTransformerQ | fast-20 | 2^14 | pending | pending | pending | pending | pending | pending | pending | pending | после checkpoint |
| E011 | T1 + Bellman/Q consistency | fast-20 | 2^14 | pending | pending | pending | pending | pending | pending | pending | pending | после E010 |
| E012 | calibrated blend T0+T1 | fast-20 | 2^14 | pending | pending | pending | pending | pending | pending | pending | pending | после E010/E011 |

## Transformer: Bellman-вариант

Проверить известную удачную схему самосогласованности по 18 соседям:

```text
y(s) = clip(1 + min_a V_target(a(s)), 0, walk_depth(s))
L_V = huber(V_online(s), stop_gradient(y(s)))
L_Q = mean_a huber(Q_online(s,a), stop_gradient(V_target(a(s))))
L_consistency = huber(V_online(s), 1 + min_a Q_online(s,a))
L_total = L_V + alpha*L_Q + beta*L_consistency + exact_anchor_loss
```

`V_target` — замороженная копия, обновляемая раз в фиксированное число эпох. Обязательно
смешивать random-walk states с exact BFS/PDB anchors и состояниями лучших путей; иначе
самообучение может закрепить смещённую шкалу. Сравнить checkpoints каждые ~2000 эпох.

## Диагностика beam для p999

Сначала запускать `2^6`, `2^10`, `2^14`; затем `2^18` и `2^21`, только если лог
показывает, что расширение beam сохраняет полезные ветви. На каждой глубине сохранять
одну компактную строку:

```text
run,depth,generated,unique,kept,score_min,score_p01,score_p10,score_p50,
score_p90,score_cutoff,incumbent_score,incumbent_rank,incumbent_percentile,
incumbent_in_beam,center_distance,quotient_distance,elapsed,peak_vram
```

Дополнительно сохранять для каждой вершины известного incumbent-пути её 18 детей:
move, T0 score, T1 score, blend score, center-coordinate, rank среди детей и факт
прохождения cutoff. Это покажет, модель ошибается локально или ветвь теряется только
из-за глобальной конкуренции.

## Кандидаты изменения отбора beam

Каждый вариант сравнивается при одинаковом числе оценённых узлов.

| ID | Selector | Зачем | Короткий тест |
|---|---|---|---|
| Q0 | global top-B по T0 | baseline | p999 B2^6..2^14 |
| Q1 | `g + lambda*h`, lambda grid | не дать score игнорировать глубину | fast-20, lambda 0.2/0.5/0.8/1.0 |
| Q2 | interleave quota T0/T1/blend | сохранить ветви, нравящиеся разным моделям | p999 B2^10, затем fast-20 |
| Q3 | 18 root-stratified sub-beams | не позволить одному первому ходу занять весь beam | p999 B2^10/B2^14 |
| Q4 | quota по center-coordinate/quotient buckets | сохранить разные center lifts | после готовности U2 coordinate |
| Q5 | deterministic top + small Gumbel/stochastic tail | diverse N-best без полного random restart | 5 seeds на fast-20 B2^14 |
| Q6 | novelty penalty по близкому state hash | уменьшить почти одинаковые состояния | p999 B2^10, одинаковые nodes |
| Q7 | BWAS/batched best-first | learned h использовать вместе с OPEN и g | 5 кубиков длины 24 |

### Поиск от 18 соседей

Для каждого генератора `a` решаем состояние `a(start)` и добавляем `a` в начало
найденного пути. Глобально это эквивалентно первому расширению обычного beam, но 18
независимых поисков или фиксированные квоты обеспечивают directional diversity.
Принимать только итоговый самый короткий full-replay-valid путь. Сначала сравнить
общий бюджет узлов Q0 и Q3; 18-кратное увеличение compute без выигрыша не допускается.

## Правило выбора модели

Transformer, MLP и blend получают одинаковый beam и одинаковые states. Новый метод
переходит с fast-20 на selection-50, если он не ухудшает valid rate, имеет отрицательный
sum delta и либо bootstrap CI уже исключает ноль, либо сигнал достаточно велик для
расширения выборки. После selection-50 дорогой beam получает только 1–2 победителя.

## Очередь ближайших сравнений

1. Q3 root-stratified diagnostic/search при равном числе оценённых узлов; измерить ранг p999 внутри правильного root.
2. Limited oversample + one-step Bellman rerank: B2^14, candidate pool около 8B, потому что p999 depth-5 raw rank около 7.4B.
3. S2 targeted MITM/endgame только после отдельного дешёвого gate: S001 и S002 дали 0.
4. U2 2048-state center-coordinate table.
5. T1 checkpoints: обычный loss против Bellman/Q consistency; затем blend grid.

Примечание после E005: Q1 `g + lambda*h` не меняет порядок в текущем
layer-synchronous beam, потому что все кандидаты слоя имеют одинаковый `g`. Его
проверять только как часть best-first/BWAS, где OPEN содержит разные глубины.
