import configparser
import os
import json


def loader(path):
    config = configparser.ConfigParser()
    with open(path) as fp:
        config.read_file(fp)
    return config


def saver(path, config):
    with open(path, 'w') as fp:
        config.write(fp)


class Config(object):

    def __init__(self, namespace: list = []):
        self.config = None
        self.namespace = namespace or []

    def tag_abs(self, tag):
        return '/'.join([*self.namespace, tag])

    def load(self, path=None):
        _path = path if path else self.path
        if _path:
            self.path = _path
            self.config = loader(_path)
            self.parameter = Parameter(self.config, self)
            self.data = Data(self.config, self)
            self.runtime = Runtime(self.config, self)
            self.log = Log(self.config, self)
            if not self.config.has_section('PARAMETER'):
                self.config.add_section('PARAMETER')
                self.save()
        return self

    # def save_as(self, path):
    #     if self.config:
    #         saver(self.config, path)
    #     return self

    def save(self):
        if self.config:
            saver(self.path, self.config)
        return self

    # def data(self, tag: str) -> str:
    #     return os.path.join(self.config['PATH']['DATA'], os.path.join(*self.namespace, tag))
    #
    # def runtime(self, tag: str) -> str:
    #     return os.path.join(self.config['PATH']['RUNTIME'], os.path.join(*self.namespace, tag))

    def cast(self, sub_namespace: str=None):
        real_namespace = self.namespace.copy()
        if sub_namespace is not None:
            real_namespace.append(sub_namespace)
        return Config(real_namespace).load(self.path)


class Parameter(object):
    def __init__(self, config: configparser.ConfigParser, upper: Config):
        self.__config = config
        self.__upper = upper

    def tag_abs(self, tag: str):
        return '.'.join([*self.__upper.namespace, tag])

    def put(self, tag: str, value):
        self.__config['PARAMETER'][self.tag_abs(tag)] = json.dumps(value)
        self.__upper.save()
        return

    def get(self, tag: str):
        if self.tag_abs(tag) not in self.__config['PARAMETER']:
            return None
        return json.loads(self.__config['PARAMETER'][self.tag_abs(tag)])


class Data(object):
    def __init__(self, config: configparser.ConfigParser, upper: Config):
        self.__config = config
        self.__upper = upper

    def path(self, tag: str):
        _path = os.path.join(self.__config['PATH']['DATA'], os.path.join(*self.__upper.namespace, tag))
        dir_name = os.path.dirname(_path)
        os.makedirs(dir_name, exist_ok=True)
        return _path


class Runtime(object):
    def __init__(self, config: configparser.ConfigParser, upper: Config):
        self.__config = config
        self.__upper = upper

    def path(self, tag: str):
        _path = os.path.join(self.__config['PATH']['RUNTIME'], os.path.join(*self.__upper.namespace, tag))
        dir_name = os.path.dirname(_path)
        os.makedirs(dir_name, exist_ok=True)
        return _path


class Log(object):
    def __init__(self, config: configparser.ConfigParser, upper: Config):
        self.__config = config
        self.__upper = upper

    def path(self, tag: str):
        _path = os.path.join(self.__config['PATH']['LOGS'], os.path.join(*self.__upper.namespace, tag))
        dir_name = os.path.dirname(_path)
        os.makedirs(dir_name, exist_ok=True)
        return _path


def put_all_parameter(config: Config, paramter_map: dict):
    for key, value in paramter_map.items():
        config.parameter.put(key, value)



