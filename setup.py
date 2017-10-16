import os
from utils.ConfigHelper import Config

cwd = os.path.split(os.path.realpath(__file__))[0]
config = Config().load(os.path.join(cwd, 'config.ini'))


