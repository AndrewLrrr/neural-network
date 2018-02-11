"""
Данная нейросеть представляет собой многослойный перцептрон и состоит из трёх слоёв (входной (сенсорный), скрытый и
выходной). Нейроны каждого слоя соединены по принципу "каждый с каждым".
Пример схемы нейронной сети: http://robocraft.ru/files/neuronet/neuronet.png
"""

import os
import pickle
import errno

import numpy as np


STORAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.storage'))


class NeuralNetwork:
    def __init__(self, input_nodes=3, hidden_nodes=3, output_nodes=3, rate=0.3, load_from=None):
        """
        :param int input_nodes: количество узлов во входном слое
        :param int hidden_nodes: количество узлов в скрытом слое
        :param int output_nodes: количество узлов в выходном слое
        :param float rate: коэфициент обучения
        :param str load_from: загрузить данные из предварительно обученной модели
        """
        if load_from is not None:
            if not self.load(load_from):
                raise ValueError('Model with name `{}` not found'.format(load_from))
        else:
            self.input_nodes = input_nodes
            self.hidden_nodes = hidden_nodes
            self.output_nodes = output_nodes
            self.rate = rate
            self.w_i_h = None  # весовые коэффициенты между входным и скрытым слоем
            self.w_h_o = None  # весовые коэффициенты между скрытым и выходным слоем
            self.__init_weights()

    def train(self, input_list, target_list):
        """Тренировка нейронной сети - уточнение весовых коэффициентов
        :param iterable input_list: входные данные
        :param iterable target_list: целевые значения
        """
        # Преобразуем входные данные в двумерный массив [1, 2, 3, 4] -> array([[1], [2], [3], [4]])
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # Расчитаем входящие сигналы для скрытого слоя
        h_inputs = np.dot(self.w_i_h, inputs)

        # Расчитаем исходящие сигналы для скрытого слоя
        h_outputs = self.__activation_function(h_inputs)

        # Расчитаем входящие сигналы для выходного слоя
        o_inputs = np.dot(self.w_h_o, h_outputs)

        # Расчитаем исходящие сигналы для выходного слоя
        o_outputs = self.__activation_function(o_inputs)

        # Выходная ошибка сети = целевое значение - фактическое значение
        o_errors = targets - o_outputs

        # Ошибки скрытого слоя - это ошибки выходного слоя сети,
        # распределенные пропорционально весовым коэфициентам связей
        # и рекомбинированные на скрытых узлах
        h_errors = np.dot(self.w_h_o.T, o_errors)

        # Обновим весовые по следующей формуле:
        # alpha * e * sigmoid(x) * (1 - sigmoid(x)) * o, где
        # alpha - коэфициент обучения,
        # e - выходная ошибка,
        # sigmoid(x) * (1 - sigmoid(x)) - производная от функции активации (сигмойды в нашем случае),
        # o - выходной сигнал предыдущего слоя.

        # Обновим весовые коэфициенты между скрытым и выходным слоем сети
        self.w_h_o += self.rate * np.dot((o_errors * o_outputs * (1 - o_outputs)), h_outputs.T)

        # Обновим весовые коэфициенты между входным и скрытым слоем сети
        self.w_i_h += self.rate * np.dot((h_errors * h_outputs * (1 - h_outputs)), inputs.T)

    def predict(self, input_list):
        """Опрос нейронной сети - получение значений сигналов выходных узлов
        :param iterable input_list: входные данные
        :return numpy.array: выходные данные
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

    def save(self, key):
        """Сохраняет обученную модель
        :param str key: имя модели
        """
        try:
            os.makedirs(STORAGE_PATH)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        file_path = os.path.join(STORAGE_PATH, key)
        value = {
            'input_nodes': self.input_nodes,
            'hidden_nodes': self.hidden_nodes,
            'output_nodes': self.output_nodes,
            'rate': self.rate,
            'w_i_h': self.w_i_h,
            'w_h_o': self.w_h_o,
        }
        with open(file_path, mode='wb') as fn:
            pickle.dump(value, fn, protocol=2)

    def load(self, key):
        """Загружает обученную модель
        :param str key: имя модели
        """
        file_path = os.path.join(STORAGE_PATH, key)
        if os.path.isfile(file_path):
            with open(file_path, mode='rb') as fn:
                value = pickle.load(fn)
            self.input_nodes = value['input_nodes']
            self.hidden_nodes = value['hidden_nodes']
            self.output_nodes = value['output_nodes']
            self.rate = value['rate']
            self.w_i_h = value['w_i_h']
            self.w_h_o = value['w_h_o']
            return True
        else:
            return False

    def __init_weights(self):
        """Инициализация случайных весов используя "улучшенный" вариант инициализации весовых коэфициентов. 
           Весовые коэфициенты выбираются из нормального распределения центром в нуле и со стандартным отклонением, 
           величина которого обратно пропорциональна квадратному корню из количества входящих связей на узел.
        """
        self.w_i_h = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.w_h_o = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

    @staticmethod
    def __activation_function(s):
        """Функция активации нейронной сети
        :param iterable s: двумерный массив входящих сигналов сети
        :return numpy.array: двумерный массив сглаженных комбинированных сигналов
        """
        return 1.0 / (1.0 + np.exp(-s))  # в качастве функции активации будет выступать сигмойда
