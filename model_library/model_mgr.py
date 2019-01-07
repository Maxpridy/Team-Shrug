# from model_library.baseline import Baseline
# from model_library.umsoImg_v1 import UmsoImg_v1
from model_library.umsoImg_v2 import UmsoImg_v2
# from model_library.umsoImg_v3 import UmsoImg_v3
# from model_library.umsoImg_v4 import UmsoImg_v4


def get_model(model_name='baseline'):
    # if model_name.lower() == 'baseline':
    #     return Baseline
    # elif model_name.lower() == 'umsoimg_v1':
    #     return UmsoImg_v1
    if model_name.lower() == 'umsoimg_v2':
        return UmsoImg_v2
    # elif model_name.lower() == 'umsoimg_v3':
    #     return UmsoImg_v3
    # elif model_name.lower() == 'umsoimg_v4':
    #     return UmsoImg_v4