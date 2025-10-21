 
# Импортируем библиотеки
import pandas as pd

# Загружаем библиотеки для визуализации данных
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Загружаем библиотеку для расчёта коэффициента корреляции phi_k
try:
    import phik
except ModuleNotFoundError as e:
    get_ipython().system('pip install phik')
    import phik
    print("Error was:", e) 
# Это позволит установить необходимую библиотеку в случае её отсутствия
 
# **- Выведем информацию о датафреймах**
#**Создаем датафреймы из предоставленных таблиц**
# Записывем базовую часть в переменную 
base_url = "https://code.s3....."
# Создаем датафремы с использованием переменной
company_and_rounds = pd.read_csv(base_url + 'company_and_rounds.csv')
acquisition = pd.read_csv(base_url + 'acquisition.csv')
degrees = pd.read_csv(base_url + 'degrees.csv')
education = pd.read_csv(base_url + 'education.csv')
fund = pd.read_csv(base_url + 'fund.csv')
investment = pd.read_csv(base_url + 'investment.csv')
people = pd.read_csv(base_url + 'people.csv')

#**Приведем к единому стилю написания названий столбцов**

# Переименуем столбец company  ID
company_and_rounds = company_and_rounds.rename(columns={'company  ID': 'id'})

# Приведем столбцы в company_and_rounds к стилю snake case
company_and_rounds.columns = company_and_rounds.columns.str.replace('  ','_')

# Выведем названия столбцов после изменений
company_and_rounds.columns

# Создаем список названий файлов
filenames = ['company_and_rounds.csv', 'acquisition.csv', 'degrees.csv', 'education.csv', 'fund.csv', 'investment.csv', 'people.csv']
# Создаем список датафреймов
dfs = {file: pd.read_csv(base_url + file) for file in filenames}

#**Выведем информацию о датафреймах**
def lookup_datasets(dfs, filenames=None):
    """
    Выводит информацию о датасетах из списка

    Parameters:
        dfs (list of pd.DataFrame): Список датафреймов для обработки.
        filenames (list of str): Список названий файлов (имена для отображения).
    """
    if filenames is None:
        filenames = [f"df_{i + 1}" for i in range(len(dfs))]

    for df, df_name in zip(dfs, filenames):
        print('-'*10,' '*5, df_name, ' '*5, '-'*10)

        # Пропущенные значения
        missing_data = df.isna().mean()
        missing_data_result = missing_data.apply(lambda x: f'{x:.2%}' if x > 0 else "")
        missing_data_name = "Пропущено" if missing_data.sum() > 0 else ""
        missing_data_result.name = missing_data_name

        # Типы данных
        dtypes_result = df.dtypes
        dtypes_result.name = "Тип данных колонки"
        fewest_nans_row = df.iloc[1:-1].isna().sum(axis=1).idxmin()

        values_type = df.loc[fewest_nans_row].map(type).T
        values_type.name = "Тип значения"

        # Объединяем результаты и пример данных
        result = pd.concat([
            dtypes_result, # типы
            values_type, # типы значений
            missing_data_result, # пропущенные значения
            df.iloc[0, :], # первая строка
            df.loc[fewest_nans_row].T, # полная строка
            df.iloc[-1, :] # последняя строка
        ], axis=1)

        display(result)

        # Проверка на полные дубликаты
        duplicates = df.duplicated().mean()
        if duplicates > 0:
            print(f'Полных дубликатов: {duplicates:.2%}')

        print()

# Проводим автоматический осмотр данных
column_counts = lookup_datasets(dfs.values(), dfs.keys())
 
# ### 1.2. Смена типов данных и анализ пропусков
#
# Меняем тип данных в датафрейме company_and_rounds
company_and_rounds[['founded_at', 'closed_at', 'funded_at']] = company_and_rounds[['founded_at', 'closed_at', 'funded_at']].astype('datetime64[ns]')

# Для удобства анализа по годам. выделим из столбцов с датами года
company_and_rounds[['founded_year', 'closed_year', 'funded_year']] = company_and_rounds[['founded_at', 'closed_at', 'funded_at']].apply(lambda x: x.dt.year)

# Выведем несколько строк датафрейма с новыми столбцами
company_and_rounds.head()

# Меняем тип данных в датафрейме acquisition
acquisition['acquired_at'] = acquisition['acquired_at'].astype('datetime64[ns]')
# Выводим информацию о датафрейме с новым типом данных
acquisition.info()

# Меняем тип данных в датафрейме education
education['graduated_at'] = education['graduated_at'].astype('datetime64[ns]')
# Выводим информацию о датафрейме с новым типом данных
education.info()

# Меняем тип данных в датафрейме fund
fund['founded_at'] = fund['founded_at'].astype('datetime64[ns]')
# Выводим информацию о датафрейме с новым типом данных
fund.info()

# Убираем явные дубликаты в company_and_rounds
company_and_rounds.drop_duplicates()

# Убираем явные дубликаты в acquisition
acquisition.drop_duplicates()

# Убираем явные дубликаты в people
people.drop_duplicates()

# Убираем явные дубликаты в degrees
degrees.drop_duplicates()

# Убираем явные дубликаты в education
education.drop_duplicates()

# Убираем явные дубликаты в fund
fund.drop_duplicates()

# Убираем явные дубликаты в investment
investment.drop_duplicates()

# Посчитаем долю пропусков в столбце company_and_rounds
pd.DataFrame(round(company_and_rounds.isna().mean()*100, 2)).sort_values(by=0, ascending=False).style.background_gradient('coolwarm')

# Посчитаем долю пропусков в датафрейме acquisition
pd.DataFrame(round(acquisition.isna().mean()*100, 2)).sort_values(by=0, ascending=False).style.background_gradient('coolwarm')

# Посчитаем долю пропусков в датафрейме education
pd.DataFrame(round(education.isna().mean()*100, 2)).sort_values(by=0, ascending=False).style.background_gradient('coolwarm')

# Посчитаем долю пропусков в датафрейме people
pd.DataFrame(round(people.isna().mean()*100, 2)).sort_values(by=0, ascending=False).style.background_gradient('coolwarm')

# Посчитаем долю пропусков в датафрейме degrees
pd.DataFrame(round(degrees.isna().mean()*100, 2)).sort_values(by=0, ascending=False).style.background_gradient('coolwarm')

# Посчитаем долю пропусков в датафрейме fund
pd.DataFrame(round(fund.isna().mean()*100, 2)).sort_values(by=0, ascending=False).style.background_gradient('coolwarm')

# Посчитаем долю пропусков в датафрейме investment
pd.DataFrame(round(investment.isna().mean()*100, 2)).sort_values(by=0, ascending=False).style.background_gradient('coolwarm')


# ## Шаг 2. Предобработка данных, предварительное исследование

# ### 2.1. Раунды финансирования по годам
# 
# Выведем сводную таблицу по годам с типичным размером средств для одного раунда и общим количеством раундов в год
funding_years = company_and_rounds.groupby('funded_year').agg({'raised_amount':'median', 'funding_rounds':'sum'})
# Оставим года в которых количество раундов больше 50 
funding_years_50 = funding_years.loc[funding_years['funding_rounds'] >= 50]
display(funding_years_50)

# Построим график для визуализации финансирования по годам
funding_years_50['raised_amount'].plot.line()

plt.title('Типичный размер средств для одного раунда по годам')
plt.xlabel('')
plt.ylabel('Млн.руб.')
plt.figure(figsize=(12, 6))
plt.show()

# ### 2.2. Люди и их образование

# Соединим датафреймы people и education для анализа компаний
people_education = people.merge(education, left_on='id', right_on='person_id', how='left')
# Выведем несколько строк объединенного датафрейма
display(people_education.head(20))

# Посчитаем долю сотрудников без инфрмации об образовании по всем данным
round(1-people_education['instituition'].nunique() / len(people_education['instituition']),2)

# Посчитаем долю сотрудников без инфрмации об образовании по компаниям
people_education['education_share'] = round((1 - people_education.groupby('company_id')['instituition'].transform('nunique')
                                          / people_education.groupby('company_id')['instituition'].transform('size')),2)

# Посчитаем количество сотрудников по компаниям
people_education['people_count'] = people_education.groupby('company_id')['person_id'].nunique()


# Сгруппируем компании по количеству сотрудников
people_education['category'] = pd.cut(people_education['people_count'], bins=[0, 1, 15, 100, 250, 1000, 100000], labels=["Один сотрудник" ,"Микро", "Малая", "Средняя", "Крупная", "Очень крупная"])

# Построим график столбчатой диаграммы по категориям
people_education.groupby('category')['education_share'].mean().plot(kind='bar',
               title=f'Распределение доли сотрудников без информации об образовании',
               legend=True,
               ylabel='Доля сотрудников',
               xlabel='Категория компании',
               rot=0,
               figsize=(8, 4))
plt.grid()

# Выводим график
plt.show()

# Уберем буквы из id сотрудника в столбце 'object_id'
degrees['object_id'] = degrees['object_id'].str.replace('p:', '')

# Меняем тип данных
degrees['object_id'] = degrees['object_id'].astype('int64')

# Проверим корректность исправлений
degrees.head()
degrees.info()

# Название и тип данных скорректированы. Можно объединять таблицы
# Присоединим таблицу degrees
people_education_degrees = people_education.merge(degrees, left_on='person_id', right_on='object_id', how='left')
# Выведем несколько строк объединенного датафрейма
display(people_education_degrees.head(20))

# Выведем названия типа образования сотрудников
people_education_degrees['degree_type'].unique()

# Выведем названия специальностей сотрудников
people_education_degrees['subject'].unique()

# ### 2.3. Объединять или не объединять — вот в чём вопрос

# Выведем уникальные значения, чтобы оценить столбцы 'network_username'
company_and_rounds['network_username'].unique()
people['network_username'].unique()
fund['network_username'].unique()

# ### 2.4. Проблемный датасет и причина возникновения пропусков

# Разделим датафрейм на два. Один с с информацией о компаниях, второй с этапами финансирования.

company_and_rounds.head()

# Создадим датафрейм с информацией о компаниях
company = company_and_rounds.drop(columns=['funding_round_id', 'company_id', 'funded_at', 'funding_round_type', 'raised_amount',
                                                           'pre_money_valuation', 'participants', 'is_first_round',
                                                           'is_last_round', 'founded_year', 'closed_year', 'funded_year'])


# Удалим дубликаты в таблице company
company = company.drop_duplicates()

# Выведем пропуски в столбце id, если они есть, так как  пропуски помешают изменить тип столбца
company[company['id'].isna()]

# Удалим строку с пропуском в id
company = company.dropna(subset=['id'])

# Отсортируем по id
company = company.sort_values(by='id')

# Приведем столбец id к целочисленному типу
company['id'] = company['id'].astype('int64')
# Посмотрим на типы данных
company.info()

# Сбросим старый индекс
company.reset_index(drop=True)

# Выведем несколько строк итогового датасета
company.head()

# Сделаем тоже самое для нового датафрейма `rounds`

# Создадим датафрейм с информацией о раундах
rounds = company_and_rounds[['funding_round_id', 'company_id', 'funded_at', 'funding_round_type', 'raised_amount',
                                                           'pre_money_valuation', 'participants', 'is_first_round',
                                                           'is_last_round', 'founded_year', 'closed_year', 'funded_year']]

rounds.info()

# Удалим дубликаты в таблице rounds
rounds = rounds.drop_duplicates()

# Выведем пропуски в столбце id, если они есть, так как пропуски помешают изменить тип столбца
rounds[rounds['funding_round_id'].isna()]

# Удалим строку с пропусками в funding_round_id
rounds = rounds.dropna(subset=['funding_round_id'])

# Приведем столбцы с id к целочисленному типу
rounds['funding_round_id'] = rounds['funding_round_id'].astype('int64')
rounds['company_id'] = rounds['company_id'].astype('int64')

# Отсортируем по id
rounds = rounds.sort_values(by='funding_round_id')

# Сбросим старый индекс
rounds.reset_index(drop=True)
rounds.head()

# Проверим тип данных
rounds.info()

# ## Шаг 3. Исследовательский анализ объединённых таблиц
# 
# ### 3.1. Объединение данных

# Создадим датафрейм со значениями раундов финансирования и инвестирования больше нуля и статусом компании acquired
company_filtered = company[(company['funding_rounds']>0) | (company['investment_rounds']>0) | (company['status']=='acquired')]
company_filtered.info()
company_filtered.shape
 
# ### 3.2. Анализ выбросов

# Разброс данных по столбцу funding_total
company_filtered['funding_total'].describe()

# Отфильтруем значения funding_total равные 0
company_filtered = company_filtered[company_filtered['funding_total'] > 0]

# Разброс данных по столбцу funding_total со значениями больше 0
company_filtered['funding_total'].describe()

# Построим гистограмму, чтобы посмотреть как распределяются размеры финансирования
funding_log = np.log10(company_filtered['funding_total'] + 1)
plt.figure(figsize=(10, 6))
sns.histplot(funding_log, bins=100, kde=True, color='skyblue')
plt.title('Распределение размеров финансирования (логарифмическая шкала)')
plt.xlabel('Размеры финансирования (log10)')
plt.ylabel('Частота')
plt.show()


# Построим boxplot для оценки данных
plt.figure(figsize=(10, 2))
sns.boxplot(x=np.log10(company_filtered['funding_total'] + 1), color='skyblue')
plt.title('Boxplot финансирования (логарифмическая шкала)')
plt.xlabel('log10(funding_total)')
plt.grid()
plt.show()

# ### 3.3. Куплены забесплатно?

# Объединим датафреймы с информацией о покупке компаний и с информацией о финансировании, где общая сумма финансирования больше 0

company_amount = acquisition.merge(company_filtered, left_on='acquired_company_id', right_on='id', how='left')

# Выберем стартапы проданные за 0 или за 1 доллар
company_amount_zero = company_amount[(company_amount['price_amount'] == 0) | (company_amount['price_amount'] <=1)]

# Посмотрим информацию по получившемуся срезу
company_amount_zero.info()

# Посмотрим информацию по этим данным
company_amount_zero['funding_total'].describe()

# Построим гистограмму, чтобы посмотреть как распределяются размеры финансирования для дешевых стартапов
funding_log = np.log10(company_amount_zero['funding_total'] + 1)
plt.figure(figsize=(10, 6))
sns.histplot(funding_log, bins=100, kde=True, color='skyblue')
plt.title('Распределение размеров финансирования дешевых стартапов (логарифмическая шкала)')
plt.xlabel('Размеры финансирования (log10)')
plt.ylabel('Частота')
plt.show()

# Построим boxplot для оценки данных
plt.figure(figsize=(10, 2))
sns.boxplot(x=np.log10(company_amount_zero['funding_total'] + 1), color='skyblue')
plt.title('Boxplot финансирования дешевых стартапов (логарифмическая шкала)')
plt.xlabel('log10(funding_total)')
plt.grid()
plt.show()

# Рассчитаем аналитически верхнюю и нижнюю границу выбросов для столбца funding_total

# Посчитаем первый и третий квартили (25% и 75%)
Q1 = company_amount_zero['funding_total'].quantile(0.25)
Q3 = company_amount_zero['funding_total'].quantile(0.75) 
# Посчитаем межквартильный размах
IQR = Q3 - Q1 
# Посчитаем нижнюю и верхнюю границы выбросов
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

display(f"Нижняя граница: {lower_bound}")
display(f"Верхняя граница: {upper_bound}")

# Посчитаем каким процентилям соответствуют граицы выбросов
lower_percentile = (company_amount_zero['funding_total'] < lower_bound).mean()*100
upper_percentile = (company_amount_zero['funding_total'] > upper_bound).mean()*100

display(f"Нижняя граница выбросов - {lower_percentile:.2f}%")
display(f"Верхняя граница выбросов - {100 - upper_percentile:.2f}%")


# ### 3.4. Цены стартапов по категориям

# Для разбиения на категории сначала выведем размах данных по ценам стартапов
acquisition['price_amount'].describe()

# Создадим датафрейм без нулевых значений
acquisition_notnull = acquisition[(acquisition['price_amount']>0) & (acquisition['price_amount']<2.600000e+12)]

# Выведем размах данных по ценам стартапов без нулевых значений
acquisition_notnull['price_amount'].describe()

# Построим гистограмму, чтобы посмотреть как распределяются цены стартапов
funding_log = np.log10(acquisition_notnull['price_amount'] + 1)

plt.figure(figsize=(10, 6))
sns.histplot(funding_log, bins=100, kde=True, color='skyblue')
plt.title('Распределение размеров цен стартапов (логарифмическая шкала)')
plt.xlabel('Цены стартапов (log10)')
plt.ylabel('Частота')
plt.show()

# Построим boxplot для оценки данных
plt.figure(figsize=(10, 2))
sns.boxplot(x=np.log10(acquisition_notnull['price_amount'] + 1), color='skyblue')
plt.title('Boxplot стоимости стартапов (логарифмическая шкала)')
plt.xlabel('log10(funding_total)')
plt.grid()
plt.show()

# Разделим цены стартапов по категориям:

bins = [-float('inf'), 1_000_000, 10_000_000, 100_000_000, 10_000_000_000, float('inf')]
labels = ["Малые", "Средние", "Крупные", "Очень крупные", "Гиганты"]
acquisition_notnull = acquisition_notnull.copy()
acquisition_notnull['category_amount'] = pd.cut(acquisition_notnull['price_amount'], bins=bins, labels=labels)

# Выведем несколько строк датафрейма с новым столбцом категорий
acquisition_notnull.head()

# Посчитаем среднюю цену по категориям
category_amount_mean = acquisition_notnull.groupby('category_amount')['price_amount'].mean().sort_values(ascending=False)
category_amount_mean

# Посмотрим для наглядности цены на столбчатой диаграмме
category_amount_mean.plot.bar(legend=True,
                title='Средние цены на стартапы по категориям',  
                ylabel='Средние цены',  
                xlabel='',
                color='skyblue',
                edgecolor='black',                                                                   
                rot=0)
sns.set_style("whitegrid")
plt.figure(figsize=(16, 16))
plt.show()

# Разделим цены стартапов c нулевыми ценами по категориям:

bins = [-float('inf'), 1_000_000, 10_000_000, 100_000_000, 10_000_000_000, float('inf')]
labels = ["Малые", "Средние", "Крупные", "Очень крупные", "Гиганты"]
#acquisition = acquisition.copy()
acquisition['category_amount'] = pd.cut(acquisition['price_amount'], bins=bins, labels=labels)

# Посчитаем разброс цен как стандартное отклонение по категориям учитывая нулевые значения
category_amount_std = acquisition.groupby('category_amount')['price_amount'].std().sort_values(ascending=False)
category_amount_std

# Посмотрим для наглядности разброс цен на столбчатой диаграмме
category_amount_std.plot.bar(legend=True,
                title='Разброс цен на стартапы по категориям',  
                ylabel='Разброс цен',  
                xlabel='',
                color='skyblue',
                edgecolor='black',                                                                   
                rot=0)
sns.set_style("whitegrid")
plt.figure(figsize=(16, 16))
plt.show()

# ### 3.5. Сколько раундов продержится стартап перед покупкой

# Посмотрим на значения в стобце funding_rounds
company_filtered['funding_rounds'].describe()

# Посчитаем типичное значение числа раундов финансирования по статусам
company_filtered.groupby('status')['funding_rounds'].median()

# Выведем типичное значение числа раундов финансирования по статусам на диаграмме
company_filtered.groupby('status')['funding_rounds'].median().sort_values(ascending=False).plot.bar(legend=True,
                title='Среднее число раундов финансирования по статусам',  
                ylabel='Среднее число раундов',  
                xlabel='Статус компании',
                color='skyblue',
                edgecolor='black',                                                                   
                rot=0)
sns.set_style("whitegrid")
plt.figure(figsize=(16, 16))
plt.show()