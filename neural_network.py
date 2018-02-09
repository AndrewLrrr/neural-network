import numpy as np

"""
Данная нейросеть представляет собой многослойный перцептрон и состоит из трёх слоёв (входной (сенсорный), скрытый и 
выходной). Нейроны каждого слоя соединены по принципу "каждый с каждым". 
Пример схемы нейронной сети: http://robocraft.ru/files/neuronet/neuronet.png
"""


class NeuralNetwork:
    def __init__(self, input_nodes=3, hidden_nodes=3, output_nodes=3, rate=0.3):
        """
        :param int input_nodes: количество узлов во входном слое
        :param int hidden_nodes: количество узлов в скрытом слое
        :param int output_nodes: количество узлов в выходном слое
        :param float rate: коэфициент обучения
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.rate = rate
        self.w_i_h = None  # весовые коэффициенты между входным и скрытым слоем
        self.w_h_o = None  # весовые коэффициенты между скрытым и выходным слоем
        self.__init_weights()

    def train(self):
        """Тренировка нейронной сети - уточнение весовых коэффициентов
        """
        pass

    def query(self, input_list):
        """Опрос нейронной сети - получение значений сигналов выходных узлов
        :param list|tuple input_list: входные данные
        :return: выходные данные
        """
        # Преобразуем входные данные в двумерный массив [1, 2, 3, 4] -> array([[1], [2], [3], [4]])
        inputs = np.array(input_list, ndmin=2).T

        # Расчитаем входящие сигналы для скрытого слоя
        h_inputs = np.dot(self.w_i_h, inputs)

        # Расчитаем исходящие сигналы для скрытого слоя
        h_outputs = self.__activation_function(h_inputs)

        # Расчитаем входящие сигналы для выходного слоя
        o_inputs = np.dot(self.w_h_o, h_outputs)

        # Расчитаем исходящие сигналы для выходного слоя
        o_outputs = self.__activation_function(o_inputs)

        return o_outputs

    def __init_weights(self):
        """Инициализация случайных весов используя "улучшенный" вариант инициализации весовых коэфициентов. 
           Весовые коэфициенты выбираются из нормального распределения центром в нуле и со стандартным отклонением, 
           величина которого обратно пропорциональна квадратному корню из количества входящих связей на узел.
        """
        self.w_i_h = np.random.normal(0.0, np.sqrt(self.hidden_nodes), (self.hidden_nodes, self.input_nodes))
        self.w_h_o = np.random.normal(0.0, np.sqrt(self.output_nodes), (self.output_nodes, self.hidden_nodes))

    @staticmethod
    def __activation_function(s):
        """Функция активации нейронной сети
        :param np.array s: двумерный массив входящих сигналов сети
        :return: двумерный массив сглаженных комбинированных сигналов
        """
        return 1.0 / (1.0 + np.exp(-s))  # в качастве функции активации будет выступать сигмойда


if __name__ == '__main__':
    nn = NeuralNetwork()
    print(nn.query([1.0, 2, -1.5]))
