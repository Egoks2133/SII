import random
import matplotlib.pyplot as plt
from deap import base, creator, tools

# Данные: поля и культуры
# Для каждого поля (N полей) известна урожайность каждой из k культур
# и стоимость каждой культуры

N = 5  # количество полей
k = 4  # количество культур

# Стоимость каждой культуры (за единицу урожая)
culture_prices = [100, 150, 200, 250]  # для культур 0, 1, 2, 3

# Урожайность: matrix[N][k] - урожайность культуры k на поле i
# Строка - поле, столбец - культура
yield_matrix = [
    [10, 15, 20, 12],  # поле 0
    [14, 18, 16, 10],  # поле 1
    [12, 20, 18, 15],  # поле 2
    [16, 14, 22, 18],  # поле 3
    [18, 16, 20, 14],  # поле 4
]

# Целевая функция: максимизировать урожай, минимизировать стоимость
# Весовые коэффициенты для балансировки целей
YIELD_WEIGHT = 1.0
PRICE_WEIGHT = 0.01  # чем меньше, тем важнее урожай; чем больше, тем важнее стоимость


def evaluate(individual):
    """
    individual: список длины N, где каждый элемент - номер культуры (0..k-1)
    Задача: максимизировать урожай (суммарный по всем полям) и
            минимизировать стоимость (сумма цен выбранных культур)
    """
    total_yield = 0
    total_price = 0

    for field_idx, culture_idx in enumerate(individual):
        # Добавляем урожай с этого поля
        total_yield += yield_matrix[field_idx][culture_idx]
        # Добавляем стоимость культуры
        total_price += culture_prices[culture_idx]

    # Фитнес: максимизируем (урожай - цена * весовой коэффициент)
    # Чем больше значение, тем лучше
    fitness = total_yield - PRICE_WEIGHT * total_price

    return (fitness,)  # DEAP ожидает кортеж


# Настройка DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # максимизация
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, k - 1)  # случайная культура
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, n=N)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)

# Операторы скрещивания
crossovers = {
    "Одноточечное": tools.cxOnePoint,
    "Двухточечное": tools.cxTwoPoint,
    "Равномерное": lambda ind1, ind2: tools.cxUniform(ind1, ind2, indpb=0.5)
}


# Операторы мутации
def mut_uniform_int(individual, low=0, up=k - 1, indpb=0.1):
    """Равномерная мутация для целочисленных генов"""
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.randint(low, up)
    return (individual,)


mutations = {
    "Uniform 5%": lambda ind: mut_uniform_int(ind, indpb=0.05),
    "Uniform 10%": lambda ind: mut_uniform_int(ind, indpb=0.1),
    "Uniform 20%": lambda ind: mut_uniform_int(ind, indpb=0.2),
}

# Параметры ГА
POP_SIZE = 100
GENERATIONS = 100
CXPB = 0.8
MUTPB = 0.2

# Эксперименты с ГА
print("=" * 60)
print("ГЕНЕТИЧЕСКИЙ АЛГОРИТМ")
print("=" * 60)

results = {}
best_ga_solutions = {}

for cx_name, cx_op in crossovers.items():
    for mut_name, mut_op in mutations.items():

        print(f"\nТестирование: {cx_name} + {mut_name}")

        toolbox.register("mate", cx_op)
        toolbox.register("mutate", mut_op)

        # Инициализация популяции
        population = toolbox.population(n=POP_SIZE)

        # Оценка начальной популяции
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        best_history = []
        best_individual = None
        best_fitness_ga = float('-inf')

        # Эволюция
        for gen in range(GENERATIONS):
            # Селекция
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Скрещивание
            for i in range(1, len(offspring), 2):
                if random.random() < CXPB:
                    toolbox.mate(offspring[i - 1], offspring[i])
                    del offspring[i - 1].fitness.values
                    del offspring[i].fitness.values

            # Мутация
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Оценка новых особей
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)

            # Замена популяции
            population[:] = offspring

            # Сохранение лучшего результата
            gen_best = max(population, key=lambda ind: ind.fitness.values[0])
            if gen_best.fitness.values[0] > best_fitness_ga:
                best_fitness_ga = gen_best.fitness.values[0]
                best_individual = gen_best[:]

            best_history.append(best_fitness_ga)

        results[f"{cx_name} + {mut_name}"] = best_history
        best_ga_solutions[f"{cx_name} + {mut_name}"] = (best_individual, best_fitness_ga)

        print(f"  Лучший фитнес: {best_fitness_ga:.2f}")
        print(f"  Решение: {best_individual}")

# График сходимости
# График сходимости
plt.figure(figsize=(13, 8))
for label, history in results.items():
    plt.plot(history, label=label)

plt.xlabel("Поколение")
plt.ylabel("Fitness (чем больше, тем лучше)")
plt.title("Сравнение операторов ГА для задачи распределения культур")
plt.legend(fontsize=8, loc='lower right')
plt.grid(True)
plt.xlim(0, 30)

plt.show()

# Анализ лучшего найденного решения
print("\n" + "=" * 60)
print("АНАЛИЗ ЛУЧШЕГО РЕШЕНИЯ")
print("=" * 60)

best_config = max(best_ga_solutions.items(), key=lambda x: x[1][1])
print(f"Лучшая конфигурация: {best_config[0]}")
best_ind, best_fit = best_config[1]
print(f"Решение: {best_ind}")
print(f"Фитнес: {best_fit:.2f}")

# Детальный расчет
print("\nРаспределение по полям:")
total_yield = 0
total_price = 0
for i, culture in enumerate(best_ind):
    yield_val = yield_matrix[i][culture]
    price_val = culture_prices[culture]
    total_yield += yield_val
    total_price += price_val
    print(f"  Поле {i + 1}: Культура {culture + 1} -> урожай: {yield_val}, стоимость: {price_val}")

print(f"\nИТОГО:")
print(f"  Суммарный урожай: {total_yield}")
print(f"  Суммарная стоимость: {total_price}")
print(f"  Фитнес (урожай - {PRICE_WEIGHT}*стоимость): {total_yield - PRICE_WEIGHT * total_price:.2f}")