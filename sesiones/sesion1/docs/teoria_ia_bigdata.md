# Sesión 1: Estadística Descriptiva, Análisis Exploratorio (EDA) y Calidad del Dato
## Fundamentos Matemáticos y Transferencia a Big Data & Machine Learning

> **Materia**: Estadística Descriptiva Univariante y Bivariante  
> **Programa**: Máster Universitario en Big Data & Data Science (MUBD) — LaSalle Campus Barcelona  
> **Conexión Tecnológica**: Data Profiling en Data Lakes, Algoritmos Probabilísticos de Streaming (HyperLogLog, t-digest), MLOps (Data Drift / PSI), e Imputación Formal de Datos Faltantes (Rubin).

---

## 1. Fundamentos Teóricos de la Sesión 1

### 1.1. Taxonomía de Variables y Escalas de Medida
El punto de partida de cualquier modelización estadística o pipeline de datos es la correcta clasificación de las covariables según la escala de medida (taxonomía de Stevens, 1946). La escala determina qué operaciones algebraicas y contrastes estadísticos son matemáticamente legítimos:

1. **Variables Cualitativas (Categóricas)**:
   - **Nominales**: Los valores representan categorías mutuamente excluyentes sin orden jerárquico natural (e.g., marcas de móviles `Brand`, tipo de contrato `Contract`, condición de extranjero `foreign`, sector de trabajo `job`).
     * *Operaciones válidas*: Conteo de frecuencias absolutas y relativas, moda, tablas de contingencia, test $\chi^2$ de independencia. La media o la varianza carecen de sentido matemático.
   - **Ordinales**: Existe un orden o jerarquía intrínseca, pero las distancias entre categorías contiguas no son constantes ni métricas (e.g., nivel de estudios, tramos de rating crediticio, escalas Likert).
     * *Operaciones válidas*: Medianas, percentiles, contrastes no paramétricos basados en rangos (Mann-Whitney U, Kruskal-Wallis) y correlación de rangos de Spearman ($\rho$).

2. **Variables Cuantitativas (Numéricas)**:
   - **Discretas**: Toman valores aislados, habitualmente en el conjunto de los enteros $\mathbb{N}_0$ (e.g., número de créditos abiertos `num.credits`, número de personas a cargo `num.people`, retraso de vuelos en minutos enteros).
   - **Continuas**: Pueden tomar cualquier valor dentro de un intervalo continuo de la recta real $\mathbb{R}$ (e.g., importe de factura `Bill`, saldo en cuenta, edad `age`, duración de llamadas `duration`).
     * *Operaciones válidas*: Media, desviación típica, momentos de orden superior (asimetría, curtosis), transformaciones matemáticas (logarítmicas, Box-Cox) y modelos de regresión lineal.

---

### 1.2. Medidas de Tendencia Central, Dispersión y Corrección por Insesgadez

#### A. Medidas de Centralidad
- **Media Aritmética ($\bar{x}$)**: Es el centro de gravedad de la distribución:
  $$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$
  *Propiedad*: Es el estimador insesgado de varianza mínima bajo el supuesto de normalidad, pero carece de resistencia ante valores atípicos (*outliers*). Su **punto de ruptura (*breakdown point*) es $0\%$**: una única observación con valor aberrante ($x_i \to \infty$) puede desplazar la media arbitrariamente lejos del cuerpo principal de los datos.
- **Mediana ($\tilde{x}$ o $Q_2$)**: Valor que divide a la muestra ordenada en dos partes iguales:
  $$\tilde{x} = \begin{cases} x_{\left(\frac{n+1}{2}\right)} & \text{si } n \text{ es impar} \\ \frac{1}{2}\left( x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)} \right) & \text{si } n \text{ es par} \end{cases}$$
  *Propiedad*: Es el estimador que minimiza la suma de desviaciones absolutas ($\sum |x_i - c|$). Su punto de ruptura es del **$50\%$**, siendo la medida de elección en distribuciones asimétricas (rentas, precios, importes de crédito).
- **Media Recortada (*Trimmed Mean*)**: Promedio tras excluir una proporción $\alpha$ (típicamente 5% o 10%) de las observaciones más pequeñas y más grandes. Combina la eficiencia de la media con la robustez de la mediana.

#### B. Medidas de Dispersión y Corrección de Bessel
- **Varianza Muestral ($s^2$)**:
  $$s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2$$
  *La corrección de Bessel ($n-1$)*: Si dividiéramos por $n$, el estimador subestimaría sistemáticamente la varianza poblacional $\sigma^2$, porque los datos muestrales están naturalmente más cercanos a su propia media muestral $\bar{x}$ que a la verdadera media poblacional $\mu$. La esperanza matemática demuestra que:
  $$E\left[ \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2 \right] = \frac{n-1}{n} \sigma^2$$
  Dividir por los **grados de libertad ($n-1$)** restaura la propiedad de **insesgadez**: $E[s^2] = \sigma^2$.
- **Desviación Típica ($s$)**: Raíz cuadrada de la varianza; preserva las unidades originales de la variable.
- **Rango Intercuartílico ($\mathrm{RIC}$)**:
  $$\mathrm{RIC} = Q_3 - Q_1$$
  Representa la amplitud del 50% central de la muestra. Insensible a colas y valores extremos.
- **Desviación Absoluta Mediana ($\mathrm{MAD}$)**:
  $$\mathrm{MAD} = \mathrm{median}\left( |x_i - \tilde{x}| \right)$$
  Para una distribución normal $\mathcal{N}(\mu, \sigma^2)$, se cumple que $\sigma \approx 1{,}4826 \cdot \mathrm{MAD}$. Es el estimador de dispersión no paramétrico más robusto disponible en estadística.

---

### 1.3. Asimetría, Curtosis y Reglas de Detección de Outliers
- **Momento Centrado de Orden $k$**: $m_k = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^k$.
- **Coeficiente de Asimetría de Fisher-Pearson ($g_1$)**:
  $$g_1 = \frac{m_3}{s^3}$$
  * $g_1 = 0$: Distribución simétrica (e.g., normal).
  * $g_1 > 0$: Asimetría positiva (cola larga a la derecha, media $>$ mediana; habitual en ingresos, saldos bancarios y facturación).
  * $g_1 < 0$: Asimetría negativa (cola larga a la izquierda, media $<$ mediana).
- **Curtosis y Exceso de Curtosis ($g_2$)**:
  $$g_2 = \frac{m_4}{s^4} - 3$$
  * $g_2 = 0$: Mesocúrtica (comportamiento en colas idéntico a la distribución normal).
  * $g_2 > 0$: Leptocúrtica (colas pesadas, mayor probabilidad de eventos extremos o *cisnes negros*).
  * $g_2 < 0$: Platicúrtica (colas ligeras, distribución más aplanada).
- **Criterio de Tukey para Detección de Valores Atípicos**:
  $$\text{Límite Inferior} = Q_1 - 1{,}5 \cdot \mathrm{RIC}, \qquad \text{Límite Superior} = Q_3 + 1{,}5 \cdot \mathrm{RIC}$$
  Puntos fuera de este intervalo se representan como puntos aislados en los diagramas de caja (*boxplots*).

---

### 1.4. Relaciones Bivariantes: Pearson vs. Spearman
- **Coeficiente de Correlación Lineal de Pearson ($r$)**:
  $$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2 \sum_{i=1}^n (y_i - \bar{y})^2}} \in [-1, 1]$$
  *Supuestos*: Exige variables cuantitativas continuas, relación lineal y ausencia de outliers severos. Un valor $r = 0$ indica ausencia de relación **lineal**, pero no descarta relaciones cuadráticas o circulares deterministas ($Y = X^2$).
- **Coeficiente de Correlación de Rangos de Spearman ($\rho$)**:
  Calcula el coeficiente de Pearson sobre los rangos ordenados de las observaciones $\mathrm{rg}(x_i), \mathrm{rg}(y_i)$:
  $$\rho = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}, \quad \text{donde } d_i = \mathrm{rg}(x_i) - \mathrm{rg}(y_i)$$
  *Propiedad*: Evalúa la **monotonicidad** de la relación (si cuando $X$ crece, $Y$ siempre crece, aunque sea de forma exponencial o logarítmica). Es robusto ante transformaciones monótonas y outliers.

---

### 1.5. Tablas de Contingencia, Perfiles Condicionales y Paradoja de Simpson
Al cruzar dos variables cualitativas (e.g., tipo de contrato `Contract` vs. uso `Use`), una tabla de frecuencias absolutas es insuficiente. Deben examinarse las distribuciones de probabilidad condicionada:
1. **Proporciones Marginales Globales**: Divididas por el tamaño muestral total $N$.
2. **Proporciones por Fila**: $P(Y = y_j \mid X = x_i) = \frac{n_{ij}}{n_{i\cdot}}$. Muestran la distribución de la respuesta dentro de cada categoría de $X$.
3. **Proporciones por Columna**: $P(X = x_i \mid Y = y_j) = \frac{n_{ij}}{n_{\cdot j}}$. Muestran la composición del predictor dentro de cada clase de $Y$.
4. **Diagramas de Mosaico (*Mosaic Plots*)**: El ancho de cada columna refleja la probabilidad marginal del predictor $P(X)$, mientras que la altura de las celdas representa la probabilidad condicional $P(Y \mid X)$. El área resultante es exactamente proporcional a la frecuencia conjunta observada.

> [!CAUTION]
> **La Paradoja de Simpson (Yule-Simpson)**:
> Una asociación positiva (o negativa) observada en toda la población puede invertirse o desaparecer cuando los datos se estratifican por una tercera variable de confusión (*confounder*). Esto ocurre cuando dicha variable latente está fuertemente correlacionada tanto con la covariable independiente como con la dependiente. Consecuencia en ciencia de datos: **la correlación observacional no prueba causalidad; agregar datos sin controlar factores de confusión puede llevar a decisiones empresariales y clínicas catastróficas**.

---

## 2. Aplicación Directa a Big Data e Inteligencia Artificial

### 2.1. Data Profiling en Data Lakes y Algoritmos Probabilísticos de Streaming (*Data Sketching*)
En plataformas de Big Data distribuido (Apache Spark, Trino, Databricks, BigQuery) que procesan tablas de miles de millones de registros (petabytes), las operaciones de ordenación requeridas para calcular percentiles o medianas exactas colapsan la memoria de los nodos (*Out-Of-Memory errors*). Para resolverlo, la industria implementa **estructuras de datos probabilísticas (Data Sketches)** que operan en una única pasada (*single-pass stream*):

1. **HyperLogLog (HLL) — Conteo de Elementos Distintos (`COUNT(DISTINCT)`)**:
   - *Problema*: Contar cuántos usuarios únicos visitaron una plataforma web sin almacenar millones de UUIDs en memoria.
   - *Fundamento Estadístico*: Aplica una función hash uniforme de 64 bits a cada dato entrante. La posición del primer bit con valor `1` en el hash sigue una distribución geométrica:
     $$P(\text{primer 1 en posición } k) = 2^{-k}$$
     Registrar el máximo valor de $k$ observado en $m = 2^b$ registros independientes (*stochastic averaging*) permite estimar la cardinalidad $N$ mediante la media armónica.
   - *Ventaja Big Data*: Ofrece un error relativo de $\approx 1{,}04 / \sqrt{m}$ (típicamente $< 1\%$) utilizando únicamente **1,5 KB de memoria**, independientemente de si hay mil o mil millones de usuarios.
2. **t-digest — Cuantiles y Latencias Extremas (p95, p99, p99.9)**:
   - *Problema*: Calcular SLAs de latencia de endpoints de inferencia de modelos de lenguaje (LLMs) sobre petabytes de logs sin ordenar los tiempos de respuesta.
   - *Fundamento Estadístico*: Agrupa las muestras en clústeres unidimensionales con capacidad elástica: los clústeres son extremadamente densos en las colas (cerca de 0 y 1) y más anchos en el centro.
   - *Ventaja Big Data*: Permite calcular percentiles extremos con altísima resolución y es **completamente paralelizable** (los digests calculados en diferentes workers de Spark se fusionan mediante adición asociativa sin pérdida de precisión).
3. **Count-Min Sketch — Frecuencias de Elementos en Streaming**:
   - Matriz 2D de contadores con $d$ funciones hash independientes. Permite identificar elementos ultra-frecuentes (*heavy hitters*) y palabras clave en NLP a escala masiva consumiendo memoria sublineal $O(\frac{1}{\epsilon} \ln \frac{1}{\delta})$.

---

### 2.2. Detección de Data Drift y Covariate Shift en MLOps
Un modelo de Machine Learning entrenado con datos históricos sufre degradación de rendimiento en producción cuando cambia el entorno operativo. La estadística descriptiva bivariante proporciona las métricas para monitorizar este desvío:

- **Covariate Shift**: Se produce cuando cambia la distribución marginal de entrada $P(X_{\text{prod}}) \neq P(X_{\text{train}})$, aunque la función $P(Y \mid X)$ se mantenga constante.
- **Monitorización Estadística**:
  1. *Test de Kolmogorov-Smirnov (K-S)*: Contraste no paramétrico que calcula la distancia suprema entre la función de distribución acumulada empírica de entrenamiento ($F_1(x)$) y la de producción ($F_2(x)$):
     $$D = \sup_x |F_1(x) - F_2(x)|$$
     Si el p-valor asociado es inferior al umbral fijado, se dispara automáticamente un reentrenamiento del modelo.
  2. *Population Stability Index (PSI)*: Basado en la divergencia de Kullback-Leibler, discretiza la feature en 10 cuantiles y evalúa:
     $$\mathrm{PSI} = \sum_{b=1}^{10} \left( \% \text{Actual}_b - \% \text{Esperado}_b \right) \cdot \ln\left( \frac{\% \text{Actual}_b}{\% \text{Esperado}_b} \right)$$
     * $\mathrm{PSI} < 0{,}10$: Estabilidad total; el modelo sigue siendo válido.
     * $0{,}10 \le \mathrm{PSI} < 0{,}25$: Deriva moderada; requiere supervisión.
     * $\mathrm{PSI} \ge 0{,}25$: Desplazamiento severo (*significant drift*); el modelo debe suspenderse o reentrenarse.

---

### 2.3. Preprocesamiento e Imputación Formal de Datos Ausentes
En entornos industriales, la pérdida de datos debe modelarse formalmente bajo la taxonomía de Donald Rubin (1976):
- **MCAR (Missing Completely at Random)**: La probabilidad de que falte un dato es independiente de cualquier variable observada o no observada. La eliminación de casos (*listwise deletion*) es matemáticamente insesgada, aunque reduce la potencia muestral.
- **MAR (Missing at Random)**: La ausencia depende de variables observadas (e.g., los hombres omiten reportar sus ingresos con mayor frecuencia, pero dentro del estrato "hombre", la falta es aleatoria). 
  * *Solución en ML*: Imputación multivariante basada en árboles (MissForest) o imputación por cadenas de ecuaciones (MICE). Imputar la media muestral en MAR introduce un sesgo artificial severo y colapsa la varianza.
- **MNAR (Missing Not at Random)**: La probabilidad de falta depende del propio valor ausente (e.g., clientes con deudas impagadas no responden encuestas de ingresos).
  * *Solución en ML*: La eliminación introduce sesgo de selección severo. Debe conservarse el patrón de ausencia creando un predictor binario explícito (*Missingness Indicator*: $I_X = 1$ si falta $X$, $0$ si está presente) antes de imputar un valor constante o recurrir a modelos de censura de Heckman.

### 2.4. Escalado Estadístico Robusto para Redes Neuronales y Algoritmos de Distancias
Algoritmos de Machine Learning basados en cálculo de distancias euclídeas (k-Means, k-NN, SVM) o en descenso de gradiente (redes neuronales densas):
- `StandardScaler` ($\frac{x - \bar{x}}{s}$): Colapsa cuando los datos presentan colas pesadas o outliers extremos, comprimiendo el 99% de los datos normales en un rango diminuto cercano a cero.
- `RobustScaler`: Escala los datos utilizando la mediana y el rango intercuartílico:
  $$x_{\text{robust}} = \frac{x - \tilde{x}}{\mathrm{RIC}}$$
  Inmuniza al modelo frente a outliers extremos, preservando el gradiente numérico durante el entrenamiento en Deep Learning.
