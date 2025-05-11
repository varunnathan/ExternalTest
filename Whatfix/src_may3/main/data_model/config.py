from pydantic import BaseModel, Extra
from pydantic_settings import BaseSettings
import typing
from main.common.constants import PROJECT_ROOT_DIR
import os
import yaml

service_config_object: "ServiceConfig" = None
service_settings_object: "ServiceSettings" = None


class ServiceSettings(BaseSettings):
    ENVIRONMENT: str = "e2e"

    class Config:
        extra = Extra.forbid


class RootConfig(BaseModel):
    __service_settings__ = ServiceSettings()


class ModelConfig(RootConfig):
    model_name: str
    max_seq_len: int
    ctx_dim: typing.Dict[str, int]


class DataConfig(RootConfig):
    data_dir_prefix: typing.Dict[str, str]
    special_tokens: typing.Dict[str, str]
    max_num_values_cat_slot: int


class TrainingConfig(RootConfig):
    output_dir: typing.Dict[str, str]
    eval_strategy: str
    save_strategy: str
    eval_steps: int
    save_steps: int
    logging_steps: int
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    seed: int
    max_n_epochs: int


class InferenceConfig(RootConfig):
    prob_threshold: typing.Dict[str, float]


class ServiceConfig(RootConfig):
    base_model_config: ModelConfig
    data_config: DataConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig

def load_service_config(path=None) -> typing.Tuple[ServiceConfig, ServiceSettings]:
    global service_config_object, service_settings_object
    if not service_config_object:
        service_settings_object = ServiceSettings()

    if not service_config_object:
        if not path:
            path = os.path.join(str(PROJECT_ROOT_DIR.parent),
                                f"config/config-{service_settings_object.ENVIRONMENT}.yaml")
        with open(path) as f:
            t = yaml.load(f, Loader=yaml.FullLoader)
        t = ServiceConfig(**t)
        service_config_object = t
    return service_config_object, service_settings_object
