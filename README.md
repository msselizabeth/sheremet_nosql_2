# HW 2

**Repo:** https://github.com/msselizabeth/sheremet_nosql_2

## Configuration

## Part 1

#### 1.2 Вибір інструментів

1. Чим Pinecone відрізняється від Qdrant і Chroma за моделлю розгортання, ліцензією і продуктивністю? У якому сценарії ви б обрали кожен із них?

- **Pinecone** — це SaaS-платформа, ідеальна для малого комерційного production або для автоматизації(як ми викоритсовуємо в поточній компанії). Ми зберігаємо попреднбо скорочені *support tickets* та шукаємо найбільш схожі за сенсом коли надходить новий—для пошуку інсайтів щодо root causes, або може були схожі кейси та зберегти час.
- **Qdrant** має відкритий код і підходить для *self-hosted* рішень, де потрібен повний контроль над даними, підвищена безпека та висока швидкість роботи. 
- **Chroma** — це найпростіша *in-memory* векторна база, яку раціонально обирати виключно для локальної розробки або швидкого прототипування(тестування ідей).

2. Чому для задачі пошуку по науковим текстам обрана модель `specter2_base`, а не універсальна `all-MiniLM-L6-v2`? Знайдіть картку моделі на HuggingFace і процитуйте, для яких задач вона навчена.

    Універсальні моделі навчаються на текстах загального призначення, а `specter2_base` створена спеціально для академічного контексту. Згідно з карткою моделі на HuggingFace, вона навчена для таких форматів задач: 
    - Classification
    - Regression
    - Proximity (Retrieval)
    - Adhoc Search
      
    *It builds on the work done in SciRepEval: A Multi-Format Benchmark for Scientific Document Representations and we evaluate the trained model on this benchmark as well.*

3. Що написано у картці моделі про рекомендовану метрику схожості? Чому це важливо при створенні індексу?

    Для моделей сімейства Sentence Transformers рекомендованою метрикою оцінки відстані є Cosine Similarity. Це критично важливо при створенні індексу в Pinecone: база даних повинна шукати найближчі вектори за тією ж математичною формулою, за якою модель оптимізували під час тренування, інакше результати пошуку будуть повністю нерелевантними.

#### 1.3 Отримання ембеддингів

Математично косинусна схожість між двома векторами $a$ та $b$ обчислюється як їхній скалярний добуток, поділений на добуток їхніх довжин:

$\text{Cosine Similarity} = \frac{a \cdot b}{||a|| ||b||}$

Нормалізація ембеддингів означає зведення векторів до одиничної довжини, тобто їхня норма стає рівною одиниці. Якщо підставити ці значення у формулу, знаменник перетворюється на одиницю, і рівняння скорочується:

$\text{Cosine Similarity} = \frac{a \cdot b}{1} = a \cdot b = \text{Dot Product}$

Тому для нормалізованих векторів ці дві метрики стають математично ідентичними. Вигідніше використовувати саме Dot Product для одиничних векторів, оскільки це економить обчислювальні ресурси бази даних на непотрібних операціях ділення.

## Part 2(Pinecone Screenshot)

![Pinecone Results](./Pincecone_Index.png)

## Part 3

#### Task 4 Аналіз результатів фільтрації:

При пошуку за запитом `"reinforcement learning"` результати кардинально відрізняються залежно від метаданих:

- Фільтр A ($>= 2019$, `cs.LG`): Видача фокусується на сучасних досягненнях у машинному навчанні, оскільки ми жорстко обмежили категорію `cs.LG` і взяли свіжі роки. Фільтр А не повернув результатів, оскільки з метою економії оперативної пам'яті датасет не перемішувався (читалися перші 10 000 рядків хронологічного файлу, що дало лише статті 2007-2008 років). Фільтр відпрацював коректно і відкинув нерелевантні дані.

- Фільтр B ($< 2015$): Видача показує фундаментальні, старіші підходи. Оскільки ми не обмежували категорію, пошуковик знайшов застосування концепцій навчання з підкріпленням у суміжних сферах: `physics.soc-ph`, `cs.MA`, які були популярні до буму глибокого навчання.

#### Theoretical Questions:

1. *Чи збігаються топ-5 для cosine і dot product і чому?*

    Так, збігаються. Оскільки ембеддинги нормалізовані, знаменник у формулі косинусної схожості зникає(довжини нормалызованих векторыв = 1) => математично скорочується до скалярного добутку(dot product).

    ```
    -----------------------------------
    Local Results: Top 5 Dot Product Results
    -----------------------------------
    Top_1: Capturing knots in polymers
    Year: 2007 | Category: cond-mat.soft
    Abstarct: This paper visualizes a knot reduction algorithm...

    Top_2: Symbolic sensors : one solution to the numerical-symbolic interface
    Year: 2007 | Category: physics.ins-det
    Abstarct: This paper introduces the concept of symbolic sensor as an extension of the
    smart sensor one. Then, the links between th...


    -----------------------------------
    Local Results: Top 5 Cosine Results
    -----------------------------------
    Top_1: Capturing knots in polymers
    Year: 2007 | Category: cond-mat.soft
    Abstarct: This paper visualizes a knot reduction algorithm...

    Top_2: Symbolic sensors : one solution to the numerical-symbolic interface
    Year: 2007 | Category: physics.ins-det
    Abstarct: This paper introduces the concept of symbolic sensor as an extension of the
    smart sensor one. Then, the links between th...
    
    ```
    
2. *Чи відрізняються результати для L2 і чому?*
    
    Топ-5 статей будуть ідентичними, але для L2 сортування йде за зростанням (шукаємо мінімум відстані). Для нормалізованих векторів ці метрики жорстко пов'язані: $L2^2 = 2 - 2 \times \text{Cosine}$. Тому мінімізація L2 дорівнює максимізації Cosine.

    ```
    -----------------------------------
    Local Results: Top 5 L2 Results
    -----------------------------------
    Top_1: Capturing knots in polymers
    Year: 2007 | Category: cond-mat.soft
    Abstarct: This paper visualizes a knot reduction algorithm...

    Top_2: Symbolic sensors : one solution to the numerical-symbolic interface
    Year: 2007 | Category: physics.ins-det
    Abstarct: This paper introduces the concept of symbolic sensor as an extension of the
    smart sensor one. Then, the links between th...
    ```

3. *Що сталося б, якби ембеддинги не були нормалізовані?*

    Топ-5 для кожної метрики був би різним. Dot Product залежав би від довжини вектора. Cosine ігнорував би довжину і рахував лише кут, а L2 вимірював би абсолютну відстань між точками.
