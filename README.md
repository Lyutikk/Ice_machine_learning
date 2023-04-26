[![Typing SVG](https://readme-typing-svg.herokuapp.com?color=%2336BCF7&lines=Machine+Learning+and+Arctic+ice)](https://git.io/typing-svg)

<h1 align="center"><i>Выделение продолжительности безледного периода с помощью методов машинного обучения по данным о сплоченности морского льда, в прибрежной зоне морей российской Арктики</i></h1>

<p align="center"><img src="https://img3.akspic.ru/crops/4/4/5/5/0/105544/105544-morskoj_led-gora-led-lednikovoe_ozero-ajsberg-1920x1080.jpg" width="1000" height="600"></p>

  <h2 align="center">Сбор и аналитика данных</h2>
    <h3>Спутниковые данные NSIDC, OSISAF и JAXA</h3>
      <p>В работе были использованы сеточные наборы данных о концентрации морского льда из различных источников: 
        <ul>
          <li><a href="https://osi-saf.eumetsat.int/">OSISAF</a></li> 
          <li><a href="https://nsidc.org/home">NSIDC</a></li> 
          <li><a href="https://global.jaxa.jp/">JAXA</a></li>
        </ul>
      </p> 
      <p>Метаинформация об использованных массивах данных:</p>
      <p align="center"><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/metainfo.png" width="800" height="300"></p>
      <p>Система спутниковых данных об океанических и морских льдах – OSISAF 
        (Ocean and Sea Ice Satellite Application Facility) – продукт спутниковых измерений концентрации морского льда Европейской организации спутниковой 
        метеорологии (EUMETSAT). Аналогичный продукт представлен так же Национальным центром снега и льда США – NSIDC (National Snow and Ice Data Center), 
        и Японским агентством аэрокосмических исследований (Japan Aerospace Exploration Agency – JAXA). 
      </p>
      <p>Для визуализации и анализа данных из источников использовалась программа <b>Panoply</b>, специально разработанная в NASA для работы с форматом <b>NetCDF</b></p>
      <p align="center"><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/panoply.png" width="800" height="600"></p>
      <p>Из неструктурированных данных с набором 4 623 820 записей был составлен датасет, содержащий данные концентрации морского льда со спутниковых снимков:</p>
      <p align="center"><img src="https://raw.githubusercontent.com/Lyutikk/Ice_machine_learning/master/input_data/dataset.png" width="550" height="200"></p>
      <p>Все преобразования и работа с датафреймом производились с помощью <b>pandas и numpy</b>. Отфильтрованы строки со значением Nan, отобраны прибрежные пиксели методом 
          <b>np.argsort()</b></p>
      <p align="center"><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/points.png" width="550" height="400"></p>
      <p>Проанализировав пространственно-временные особенности данных, было предложено выделять безледный период посредством бинарной классификации. Мат основа постановки задачи:</p>
      <p><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/f(1).png" width="230" height="70"></p>
      <p><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/f(2).png" width="100" height="30"></p>
      <p>Размеченный таргет:</p>
      <p align="center"><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/target.png" width="550" height="200"></p>
  <h2 align="center">Анализ распределения данных, подбор алгоритма</h2>
    <h3>Нетривиальные случаи</h3>
      <p>Чтобы избежать зависимости результатов характеристик безледного периода от абсолютных значений концентрации морского льда (часто зашумлённых и неадекватно 
        завышенных), был предложен оригинальный метод на основе «скользящего окна» (Rolling Window Approach, RWA), основанный на анализе формы годовой кривой 
        концентрации морского льда. Замечено, что началу безлёдного периода предшествует резкое падение значений концентрации морского льда, а замерзание акватории 
        сопровождается резким подъёмом. Однако, существуют случаи, не поддающиеся описанию и решению вышеизложенными методами. К таким относятся довольно старые 
        данные, полученные в 70-80е года – для них характерны частые выбросы и «недотягивание» графика концентрации морского льда ниже порогового значения.</p>
      <p><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/point_200_098.png" width="550" height="400"></p>
      <p>С поставленной задачей классификации хорошо справился алгоритм глубокого машинного обучения <b>nn MLP classifier библиотеки scikit-learn</b>
        (имеет структуру многослойного перцептрона).</p>
      <p><b>Параметры модели:</b></p>
      <p>
        <ul>
          <li>Функция активации - logistic</li>
          <li>Число скрытых слоев - 100</li>
          <li>random_state - 1</li>
          <li>max_iter - 300</li>
        </ul>
      </p>
      <p>В ходе анализа метрик качества и распределения ошибок модели машинного обучения, мы определили оптимальное количество скрытых слоев нейронов (160), 
      дающих нам наилучший результат качества при тестировании.</p>
      <p><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/model_error_distribution.png" width="550" height="450"></p>
      <p>В ходе работы над алгоритмом машинного обучения возникали некоторые вопросы несоответствия распределения данных. Эти неординарные случаи значительно 
         спутывали фактическую картину для алгоритма и мешали его корректной работе.</p>
      <p>Такие несоответствия возникали лишь в нескольких географических областях, группируясь небольшими группами. Проанализировав их распределение относительно 
         географического положения, можно сделать вывод о том, что смазанная картина возникает в узких проливах и заливах.</p>
      <p>Вероятно некорректность этих данных вызвана достаточно грубым разрешением спутниковых снимков для слишком изрезанной береговой полосы, а также ошибками 
         измерений.</p>
      <p><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/distribution_wrong_data.png" width="550" height="450"></p>
      <p>По многочисленным источникам судовых наблюдений в областях несоответствий, по сравнению со спутниковыми снимками показатель продолжительности безледного 
         периода значительно выше, что говорит об существенных ошибках.</p>
      <p>При этом сами графики концентрации морского льда в этих точках имеют довольно разбросанную картину, существенно не дотягивая до порогового значения, и 
         соответственно неверно определяется безледный период.</p>
      <p>Для минимизации влияния таких данных на работу алгоритма, было предложено использовать сглаживание и нормализацию с помощью метода <b>MinMaxScaller</b> библиотеки 
         scikit-learn на этапе форматирования данных на три датасета для обучения.</p>
      <p>Также был написан скрипт, проверяющий строки датафрейма, которые содержат данные о концентрации на их валидность.</p>
      <p><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/wrong_plot.png" width="550" height="450"></p>
 <h2 align="center">Тестирование</h2>
    <h3>Метрики</h3>
      <p>Первым критерием качества, который мы использовали в задаче бинарной классификации - является Accuracy – доля объектов, для которых мы правильно предсказали 
        класс. Если мы рассмотрим долю правильно предсказанных положительных объектов среди всех объектов, предсказанных положительным классом, то мы получим метрику - 
        точность (precision). Если же мы рассмотрим долю правильно найденных положительных объектов среди всех объектов положительного класса, то мы получим метрику - 
        полнота (recall).</p>
      <p>Для полноты оценки был выбран комплексный показатель, сочетающий в себе предыдущие метрики - <b>F1-measure</b>.</p>
      <p><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/f1_measure.png" width="400" height="100></p>
      <p>Деление на train, test и validation производилось в соотношении 40:20:20 с исключением влияния географической привязки, информация о которой выносилась отдельно
         в датасет метаинформации.</p>
      <p>По результатам тестирования многослойного перцептрона в работе бинарной классификации безледного периода по данным спутниковых снимков NSIDC, OSISAF и JAXA 
         были полученны следующие результаты: по достаточно зашумленным данным, алгоритм смог достаточно точно выделить продолжительность безледного периода.</p>
      <p><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/500_point_test.png" width="550" height="450"></p>
      <p><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/400_point_test.png" width="550" height="450"></p>
      <p>Как видно, величина F1-меры в ходе теста составила примерно 0.99-0.98, что является достаточно хорошим показателем, без признаков переобучения.</p>
      <p>Просматривается достаточно сильное зашумление данных, значительно портящих общую картину распределения признаков. Помимо этого заметно, что график достаточно 
         разбросан опускается ниже порогового значения 15% кратковременно но часто в летне-весенние месяцы. Отметим, что показатель F1-меры для такого случая 
         составляет примерно 0.98, что не на много ниже предыдущего.</p>
      <p>Для человеческого глаза не составит труда выделить период в таких неординарных ситуациях, и догадаться, что безледный период находится в этих промежутках 
         выбросов. Однако для машинного обучения иногда это становится довольно трудной задачей. Особенно это важно учитывать при многочисленности данных, 
         имеющих разное распределение и свойства.</p>
      <p>Что до машинного обучения, такие случаи для него не распознаются корректно:</p>
      <p><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/x_val%20-%200%20rows%20plot%20in%20predicted%20y%20val.png" width="550" height="450"></p>
      <p>Как видим, алгоритм не смог корректно классифицировать периоды, и присвоил целому году значение 0 (льда нет в течении всего года). Однако, если взглянуть на 
         график концентрации можно предположить продолжительность безледного периода, в среднем 50 - 70 дней (где то между 225 и 275 днями).</p>
      <p>Для этого было предложено применить растягивание по оси, для того чтобы стало возможным определить период, свободный ото льда</p>
      <p><img src="https://github.com/Lyutikk/Ice_machine_learning/blob/master/input_data/stretching.jpg" width="550" height="450"></p>



   
