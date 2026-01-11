# MLOps
Проект, выполненный в рамках задания по курсу MLOps

В рамках проекта предполагается создать рекомендательную систему, которая по введённому пользователем названию произведения и автору должна предложить другие произведения, которые могут быть ему интересны.

В качестве датасета используется GoodReads Best Books датасет (https://www.kaggle.com/datasets/thedevastator/comprehensive-overview-of-52478-goodreads-best-b/data) в котором содержится множество книг с их подробной инофрмацие о каждой из них, а также Goodreads Book Datasets With User Rating 2M (https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m/data), в котором есть таблицы с различными книгами и их характеристиками, а также таблицы с отзывами пользователей о различных книгах. Из обоих датасетов формируется один, содержащий подробные качественные описания книг и рейтинги пользователей, поставленные этим книгам.

Предположим, что сейчас рекомендации строятся просто на основе среднего рейтинга книг: пользователю рекомендуется книга с наивысшим рейтингом среди тех, что он ещё не читал. Задача: улучшить рекомендации путём персонализации рекомендаций для пользователей на основе их истории. Для этого предполагается построить информативные эмбединги книг, после чего, используя эти эмбединги и поставленные соответствующим книгам рейтинги, делать персонализированные рекомендации книг. В силу того, что в таблице рейтингов пользователей нет информации о времени, а также нет возможности получения новых рейтингов для проведения A/B-теста для сравнения подходов, оценка качества происходит путём сравнения корелляции между вектором предсказаний рейтингов книг, которых нет в историии пользователя, и известными истинными оценками.

Для построения информативных векторных эмбедингов использовался предобученный энкодер трансформера и последующая нейросеть для проекции в меньшую размерность, обученную на TripletLoss по косинусному сходству. Далее предсказание рейтинга книги на основе истории пользователя происходит уже без нейросетевых методов, обучаемым там является только коэффициент, отвечающий за баланс между учитыванием персональных предпочтений пользователя и среднего рейтинга книги.

В отличие от первоначального задания (ветка ml_ops), где проект собирался через Makefile, тут (ветка ml_ops_2) в соответствие с заданием происходит логирование данных с помощью dvc и экспериментов c помощью mlflow. Также реализован контейнер для офлайн инференса и онлайн сервис.

Шаги для вопроизведения результатов:

1-2) Настройка окружения и данных:
git clone git@github.com:Buyan-Kirill/MLOps.git
cd MLOps
python3 -m venv venv (если ещё нет)
source venv/bin/activate
pip install -r requirements.txt
dvc pull (для скачивания нужны ключи, так как данные хранятся в облаке в Yandex Object Cloud. Файл config.local с ключами необходимо будет положить в .dvc папку. При необходимости получения ключей можно написать мне в телеграмм: @BuyanKirill)
В итоге появятся все необходимые файлы с кодом и данными

Можно запустить пайплайн заново, для этого нужно выполнить следующие шаги:
mkdir -p logs/ processed_data/ outputs/
dvc repro (в таком случае в папке mlruns появится информация о текущем запуске)

3) Оффлайн-инференс:
docker build -t ml-app:v1 .
* Первый вариант (подаём только название, автора и поставленный вами рейтинг). Работает только если книга была в датасете (поскольку для построения нового эмбединга требуется ещё описание и жанр):
echo 'title,author,rating' > docker_task/test_on_books_in_dataset.csv
echo '"To Kill a Mockingbird","Harper Lee",5' >> docker_task/test_on_books_in_dataset.csv
docker run --rm \
  -v "$(pwd)/.dvc/config.local:/app/.dvc/config.local" \
  -v "$(pwd):/io" \
  ml-app:v1 \
  python src/predict.py \
  --input_path /io/docker_task/test_on_books_in_dataset.csv \
  --output_path /io/docker_task/result.csv
  
* Второй вариант (название, автор, рейтинг, а также описание и жанры). Для случая, когда книги не нашлось в датасете и нужно построить новый эмбединг для книги а не просто сопоставить с существующим:
echo 'title,author,rating,description,genres' > docker_task/test_on_books_outside_dataset.csv
echo '"Test Book","Unknown",5,"A book about docker tests","Tech"' >> docker_task/test_on_books_outside_dataset.csv
Запустите контейнер:
Мы пробрасываем локальный конфиг .dvc/config.local внутрь контейнера для авторизации.bash
docker run --rm \
  -v "$(pwd)/.dvc/config.local:/app/.dvc/config.local" \
  -v "$(pwd):/io" \
  ml-app:v1 \
  python src/predict.py \
  --input_path /io/docker_task/test_on_books_outside_dataset.csv \
  --output_path /io/docker_task/result.csv
В обоих случаях в папке docker_task появится result.csv с рекомендациями.

4) Онлайн сервис:
python3 torch_server/change_model_extension.py (посокльку изначально модель сохраняется в формате safetensor)
mkdir -p model_store
EMB_PATH=outputs/book_encoder_contrastive_256/book_embeddings_contrastive_256.npy (итоговые эмбеддинги)

Сборка .mar файла:
torch-model-archiver --model-name book-recs \
  --version 1.0 \
  --serialized-file outputs/book_encoder_contrastive_256/model.pt \
  --handler src/torchserve_handler.py \
  --extra-files "configs/default.yaml,outputs/book_encoder_contrastive_256/config.json,src/encoder.py,src/utils.py,src/recommender.py,src/cold_start.py,processed_data/books_meta_multimodal.csv,$EMB_PATH" \
  --export-path model_store --force
docker build -t recs-service:v1 -f Dockerfile_torchserve .

Запуск сервера:
docker run --rm -d \
  -p 8080:8080 \
  -p 8081:8081 \
  --name recs \
  -v "$(pwd)/torch_server/config.properties:/home/model-server/config.properties" \
  recs-service:v1 \
  torchserve --start --model-store model-store --ts-config config.properties --ncs
  
Отправка запросов (примеры аналогичны примерам выше):
curl -X POST http://localhost:8080/predictions/recs \
  -H "Content-Type: application/json" \
  -d @torch_server/cold_start_test.json
