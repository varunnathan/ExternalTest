import argparse, time
from main.train.data_handler import DataHandler
from main.train.utils import log, ModuleType
from main.train.trainer import train_helper
from main.data_model.config import load_service_config
from main.data_model.initialize import initialize_paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", choices=[m.value for m in ModuleType], required=True)
    ap.add_argument("--splits", default='[("train", "train"), ("dev", "validation")]', help="splits for data preparation")
    args = ap.parse_args()
    args.splits = eval(args.splits)

    log("Load app config")
    app_config, _ = load_service_config()

    log("Load path config")
    path_config = initialize_paths()
    
    log("Init Data Handler")
    data_handler = DataHandler(data_config=app_config.data_config, model_config=app_config.base_model_config,
                               path_config=path_config)

    log("Data preparation begins...")
    start = time.time()
    data_handler.prepare_data(splits=args.splits, module=args.module)
    log(f"Time taken for data preparation: {time.time() - start}")

    log("Training")
    train_helper(module=args.module, data_config=app_config.data_config,
                 model_config=app_config.base_model_config,
                 training_config=app_config.training_config,
                 path_config=path_config)
    