# Пять задач для проверки универсального ML-агента

В каждой папке готовый `task.md`, обучающая выборка, тест **без таргета**,
словарь признаков и шаблон ответа. Пути в условиях уже подходят локальному агенту.
Это пять дополнительных тренировочных задач, исходные public-датасеты сохранены.

| Условие | Тип / сложность | Train | Test | Основная метрика |
| --- | --- | ---: | ---: | --- |
| [Бетонные смеси](concrete_strength/task.md) | Регрессия, средняя; новые рецептуры | 832 | 198 | RMSE ↓ |
| [Банковские вклады](bank_subscription/task.md) | Бинарная классификация, средняя; вероятности и дисбаланс | 32974 | 8214 | ROC AUC ↑ |
| [Сорта фасоли](dry_bean/task.md) | Многоклассовая классификация, базовая | 10883 | 2728 | Macro F1 ↑ |
| [Велопрокат](bike_demand/task.md) | Регрессия во времени, высокая | 15211 | 2168 | RMSLE ↓ |
| [SMS-спам](sms_spam/task.md) | Текстовая классификация, высокая для табличного агента | 4459 | 1115 | F1 класса spam ↑ |

Сложность относительная: SMS проще решить подходящей текстовой моделью, но такая
задача обнаруживает ограничения агента, который рассматривает любой текст как категорию.
У бетона меньше строк, однако тест проверяет перенос на новые составы, а не узнавание рецептур.

Источники: [Concrete](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength),
[Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing),
[Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset),
[Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
[SMS Spam](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
Лицензия наборов — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Авторы, библиографические ссылки и преобразования данных сохранены отдельно от
условий в `evaluation/catalog.py` и `evaluation/private/<dataset>/manifest.json`.

Скрытые ответы, воспроизводимая сборка и команды проверки описаны в
[инструкции оценщика](../../evaluation/README.md).
