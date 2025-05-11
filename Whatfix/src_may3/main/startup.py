import argparse, os, time, torch
from main.train.data_handler import DataHandler
from main.train.utils import log, ModuleType
from main.train.trainer import train_helper
from main.infer.inference import prediction_helper, run_inference
from main.data_model.config import load_service_config
from main.data_model.initialize import initialize_paths
from main.evaluate.evaluate_sgd import main_evaluate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", choices=[m.value for m in ModuleType], required=True)
    ap.add_argument("--splits", default='[("train", "train"), ("dev", "validation")]', help="splits for data preparation")
    ap.add_argument("--train", action="store_true", help="train the model")
    ap.add_argument("--predict", action="store_true", help="get predictions from individual models")
    ap.add_argument("--inference", action="store_true", help="get predictions on the test set")
    ap.add_argument("--evaluate", action="store_true", help="evaluate the models")
    ap.add_argument("--n_steps", type=int, default=150, help="number of steps used for training")
    ap.add_argument("--sampling", action="store_true", help="sample from the dataset")
    ap.add_argument("--n_samples", type=int, default=1000, help="number of samples to sample")
    args = ap.parse_args()
    args.splits = eval(args.splits)

    log("Load app config")
    app_config, _ = load_service_config()

    log("Load path config")
    path_config = initialize_paths()
    
    log("Init Data Handler")
    data_handler = DataHandler(data_config=app_config.data_config, model_config=app_config.base_model_config,
                               path_config=path_config)

    if args.train:
        log("Data preparation begins...")
        start = time.time()
        data_handler.prepare_data(splits=args.splits, module=args.module, is_train=True)
        log(f"Time taken for data preparation: {time.time() - start}")

        log("Training")
        train_helper(module=args.module, data_config=app_config.data_config,
                     model_config=app_config.base_model_config,
                     training_config=app_config.training_config,
                     path_config=path_config)
    
    if args.predict:
        log("Data preparation begins for getting predictions from individual models...")
        start = time.time()
        data_handler.prepare_data(module=args.module, is_train=False, test_split_tag="test")
        log(f"Time taken for data preparation: {time.time() - start}")

        log("Get Predictions")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prediction_helper(module=args.module, data_config=app_config.data_config,
                          model_config=app_config.base_model_config,
                          training_config=app_config.training_config,
                          path_config=path_config, device=device,
                          n_steps=args.n_steps, sampling=args.sampling,
                          n_samples=args.n_samples)
    
    if args.inference:
        log("Running inference on the test set...")
        start = time.time()
        run_inference(n_steps=args.n_steps, schema=data_handler.schemas, path_config=path_config,
                      model_config=app_config.base_model_config, test_split_tag="test",
                      inference_config=app_config.inference_config)
        log(f"Time taken for inference: {time.time() - start}")
    
    if args.evaluate:
        log("Evaluating the models...")
        start = time.time()
        main_evaluate(ref_dir=os.path.join(str(path_config.RAW_DATA_DIR), "test"),
                      pred_dir=str(path_config.DLG_LEVEL_PREDICTION_DIR),
                      schemas=data_handler.schemas)
        log(f"Time taken for evaluation: {time.time() - start}")
