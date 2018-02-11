Данная нейронная сеть была создана по книге:

```
Make Your Own Neural Network (Создаем нейронную сеть)
Автор: Tariq Rashid (Тарик Рашид)
ISBN: 978-5-9909445-7-2, 978-1530826605
```

Со своей стороны я добавил несколько новых фич, например, возможность сохранять и загружать уже обученную модель и провел рефакторинг кода, но общие принципы остались те же.

Тестирование нейронной сети осуществлялось на наборе рукописных цифр MNIST и находится здесь [mnist_classification.ipynb](https://github.com/AndrewLrrr/neural-network/blob/master/mnist/mnist_classification.ipynb).

Чтобы загрузить предобученную модель, необходимо выполнить следующие команды в директории где находится файл `neural_network.py`:

```
mkdir .storage
cp best_model.backup .storage/best_model
```

После этого можно работать с этой моделью по аналогии с тем как это делается в блокноте [mnist_classification.ipynb](https://github.com/AndrewLrrr/neural-network/blob/master/mnist/mnist_classification.ipynb).
