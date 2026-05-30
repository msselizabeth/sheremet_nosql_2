# HW 2

**Repo:** https://github.com/msselizabeth/sheremet_nosql_2

## Configuration

### 1. Environment Variables
    Create a `.env` file in the project root:

    ```env
    PINECONE_API_KEY=your_api_key_here
    ```

### 2. Docker Execution

Build and run the container in the background

    ```
    docker compose up -d --build

    # Access the container shell
    docker compose exec python_app bash
    
    ```

### 3. Pipeline Steps

    ```
    python scripts/01_prepare_data.py       # 1. Parse JSON and create Parquet subset
    python scripts/02_embed.py              # 2. Generate normalized embeddings (specter2_base)
    python scripts/03_load_to_pinecone.py   # 3. Create index and upsert batches to Pinecone
    python scripts/04_search.py             # 4. Semantic search and local metric calculations
    ```

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


### Part 4 Chunking

#### Terminal Output

```
docker exec -it sheremet_nosql_2 python scripts/05_chunking.py

No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
Chunking (fixed): 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:03<00:00,  8.04it/s]
Upserting: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.12s/it]
Chunking (semantic): 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:09<00:00,  3.26it/s]
Upserting: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.58it/s]

Top 1 | Paper: Absolute Calibration and Characterization of the Multiband Imaging
  Photometer for Spitzer. II. 70 micron Imaging | Chunk #3.0
Text: validates the MIPS 70 micron operating strategy, especially the use of frequent
stimulator flashes to track the changing responsivities of the Ge:Ga d...

Top 2 | Paper: Is Modified Gravity Required by Observations? An Empirical Consistency
  Test of Dark Energy Models | Chunk #3.0
Text: data already. We find w(grow) < -0.80 at 2 sigma. As an example, the best-fit
flat Dvali-Gabadadze-Porrati (DGP) model approximated by our parametriza...

Top 3 | Paper: Improved constraints on dark energy from Chandra X-ray observations of
  the largest relaxed galaxy clusters | Chunk #3.0
Text: constant paradigm. Our analysis includes conservative allowances for systematic
uncertainties. The small systematic scatter and tight constraints bode...

Top 4 | Paper: Multicolor observations of the afterglow of the short/hard GRB 050724 | Chunk #3.0
Text: compelling case for association between a short burst and a galaxy with old
stellar population. It thus plays a pivotal role in constraining progenito...

Top 5 | Paper: Dependence of CMI Growth Rates on Electron Velocity Distributions and
  Perturbation by Solitary Waves | Chunk #3.0
Text: result in a gain enhancement more than 40 dB depending on the convective growth
length within the structure. Similar enhancements may be caused by EMI...

Top 1 | Paper: The Kinematics of the Ultra-Faint Milky Way Satellites: Solving the
  Missing Satellite Problem | Chunk #1.0
Text: [slightly abridged]....

Top 2 | Paper: Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges | Chunk #0.0
Text: In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchanges. Main topic of our manuscript are critical phen...

Top 3 | Paper: The Boundary Conditions of the Heliosphere: Photoionization Models
  Constrained by Interstellar and In Situ Data | Chunk #1.0
Text: 23 - 0. 27 cm^-3, T = 6300 K, X(H^+) ~ 0. 2, and X(He^+) ~ 0. 4. These results appear to be robust since acceptable models are found for substantially...

Top 4 | Paper: Multicolor observations of the afterglow of the short/hard GRB 050724 | Chunk #0.0
Text: New information on short/hard gamma-ray bursts (GRBs) is being gathered thanks to the discovery of their optical and X-ray afterglows. However, some k...

Top 5 | Paper: A model for the Globular Cluster extreme anomalies | Chunk #0.0
Text: In spite of the efforts made in the latest years, still there is no comprehensive explanation for the chemical anomalies of globular cluster stars. Am...

Top 1 | Paper: Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges | Chunk #0.0
Text: In this thesis, we consider some spin effects in QCD and recurrence lattices
with multi-site exchanges. Main topic of our manuscript are critical phen...

Top 2 | Paper: Improved constraints on dark energy from Chandra X-ray observations of
  the largest relaxed galaxy clusters | Chunk #3.0
Text: constant paradigm. Our analysis includes conservative allowances for systematic
uncertainties. The small systematic scatter and tight constraints bode...

Top 3 | Paper: Absolute Calibration and Characterization of the Multiband Imaging
  Photometer for Spitzer. II. 70 micron Imaging | Chunk #3.0
Text: validates the MIPS 70 micron operating strategy, especially the use of frequent
stimulator flashes to track the changing responsivities of the Ge:Ga d...

Top 4 | Paper: Is Modified Gravity Required by Observations? An Empirical Consistency
  Test of Dark Energy Models | Chunk #3.0
Text: data already. We find w(grow) < -0.80 at 2 sigma. As an example, the best-fit
flat Dvali-Gabadadze-Porrati (DGP) model approximated by our parametriza...

Top 5 | Paper: The Origin of the Galaxy Mass-Metallicity Relation and Implications for
  Galactic Outflows | Chunk #3.0
Text: reflects the mass scale where MLF~1, rather than a characteristic wind speed.
The tight observed MZR scatter is ensured when t_d<1 dynamical time, whi...

Top 1 | Paper: The Kinematics of the Ultra-Faint Milky Way Satellites: Solving the
  Missing Satellite Problem | Chunk #1.0
Text: [slightly abridged]....

Top 2 | Paper: The Boundary Conditions of the Heliosphere: Photoionization Models
  Constrained by Interstellar and In Situ Data | Chunk #1.0
Text: 23 - 0. 27 cm^-3, T = 6300 K, X(H^+) ~ 0. 2, and X(He^+) ~ 0. 4. These results appear to be robust since acceptable models are found for substantially...

Top 3 | Paper: Multicolor observations of the afterglow of the short/hard GRB 050724 | Chunk #0.0
Text: New information on short/hard gamma-ray bursts (GRBs) is being gathered thanks to the discovery of their optical and X-ray afterglows. However, some k...

Top 4 | Paper: High energy afterglows and flares from Gamma-Ray Burst by Inverse
  Compton emission | Chunk #0.0
Text: We perform a detailed study of inverse Compton (IC) emission for a fireball undergoing external shock (ES) in either a uniform or a wind-like interste...

Top 5 | Paper: Spin Effects in Quantum Chromodynamics and Recurrence Lattices with
  Multi-Site Exchanges | Chunk #0.0
Text: In this thesis, we consider some spin effects in QCD and recurrence lattices with multi-site exchanges. Main topic of our manuscript are critical phen...

```

#### Theoretical Questions

*1. Яка стратегія дає більш осмислені чанки?*

Більш грунтовні результати дає **Semantic chunking**. Fixed-size нарізає текст *"наосліп"* за кількістю слів, що часто призводить до обриву логіки. Семантичний підхід розбиває текст на рівні цілих речень і об'єднує їх лише доти, поки вони зберігають високу косинусну схожість. Це гарантує, що один чанк містить одну завершену думку.

*2. Чи є випадки розрізаних речень і як це впливає на ембеддинги?*

При використанні `Fixed-size chunking` розрізані речення трапляються регулярно. Це критично погіршує якість ембеддингів: якщо алгоритм розділить складний термін навпіл, модель згенерує два окремих вектори, жоден з яких не міститиме повного змісту. В результаті такий чанк може просто не знайтися за релевантним запитом. У `Semantic chunking` таких випадків немає, оскільки спліттер базується на розділових знаках.

*3. Як розмір overlap впливає на кількість чанків і покриття тексту?*

Чим більший `overlap`, тим менший крок зсуву вікна (sliding window), відповідно, **більша загальна кількість чанків**, що збільшує витрати на зберігання у векторній БД. Проте великий overlap покращує покриття тексту: він гарантує, що важливий контекст, який знаходиться на межі двох фрагментів, не буде втрачено, і LLM отримає цілісну картину для формування відповіді.