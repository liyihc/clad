from pydantic import BaseModel
from typing import Dict, Type
from .cnn_model import *
from .gru_model import *

models: Dict[str, Type[BaseModel]] = {
    "cnn-all": CNN_All_Model,
    "cnn-gm": CNN_GM_Model,
    "cnn-cll": CNN_CLL_Model,
    "cnn": CNN_Model,
    "gru-all": GRU_ALL_Model,
    "gru-gm": GRU_GM_Model,
    "gru-cll": GRU_CLL_Model,
    "gru": GRU_AE_Model
}
