import os
import pandas as pd
dir_path = os.path.dirname(os.path.realpath(__file__))

rating_train = os.path.join(dir_path, 'data\ml-100k/ub.base')
rating_test  = os.path.join(dir_path, 'data\ml-100k/ub.test')
