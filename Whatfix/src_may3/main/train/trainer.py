import os
import time
import torch
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer
from main.train.utils import log, ModuleType
from main.train.model_handler import (build_tokenizer, IntentHead, CatSlotHead,
                                      FreeSlotHead, ReqSlotHead, InXferHead,
                                      CrossXferHead)
from main.train.data_collator import collate_intent, XferCollator, CatCollator
from main.data_model.config import DataConfig, ModelConfig, TrainingConfig
from main.data_model.initialize import PathConfig


def train_helper(module: ModuleType, data_config: DataConfig, model_config: ModelConfig,
                 path_config: PathConfig, training_config: TrainingConfig):
    start = time.time()

    log("Load Train and Validation datasets")
    start_module = time.time()
    data_dir = path_config.PREPARED_DATA_DIR
    data_dir_prefix = data_config.data_dir_prefix[module]
    train = load_from_disk(data_dir / data_dir_prefix.format("train"))
    val = load_from_disk(data_dir / data_dir_prefix.format("validation"))
    log(f"Time taken for loading dataset: {time.time() - start_module}")
    log("-" * 100)

    log("Load Tokenizer")
    start_module = time.time()
    tok = build_tokenizer(model_config)
    log(f"Time taken for loading tokenizer: {time.time() - start_module}")
    log("-" * 100)

    log("Init Model class and Data Collator")
    start_module = time.time()
    if module == ModuleType.INTENT:
        model = IntentHead(tok=tok, model_config=model_config)
        collator = collate_intent
    elif module == ModuleType.CAT_SLOT:
        model = CatSlotHead(model_config=model_config)
        collator = CatCollator(pad_id=tok.pad_token_id)
    elif module == ModuleType.FREE_SLOT:
        model = FreeSlotHead(model_config=model_config)
        collator = XferCollator(pad_id=tok.pad_token_id, module=module)
    elif module == ModuleType.REQUESTED_SLOT:
        model = ReqSlotHead(model_config=model_config)
        collator = XferCollator(pad_id=tok.pad_token_id)
    elif module == ModuleType.IN_DOMAIN_SLOT:
        model = InXferHead(model_config=model_config)
        collator = XferCollator(pad_id=tok.pad_token_id)
    else:
        model = CrossXferHead(model_config=model_config)
        collator = XferCollator(pad_id=tok.pad_token_id)
    log(f"Time taken for loading model class and collator: {time.time() - start_module}")
    log("-" * 100)
    
    log("Define Training arguments")
    start_module = time.time()
    out_dir = os.path.join(str(path_config.RAW_DATA_DIR), training_config.output_dir[module])
    training_args = TrainingArguments(
        out_dir,
        learning_rate=training_config.learning_rate,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        num_train_epochs=training_config.max_n_epochs,
        eval_strategy=training_config.eval_strategy,
        save_strategy=training_config.save_strategy,
        eval_steps=training_config.eval_steps,
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=False,
        seed=training_config.seed,
        save_safetensors=False)
    log(f"Time taken for defining training args: {time.time() - start_module}")
    log("-" * 100)
    
    log("Define Trainer")
    start_module = time.time()
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train,
                      eval_dataset=val,
                      data_collator=collator,
                      tokenizer=tok)
    log(f"Time taken for init trainer: {time.time() - start_module}")
    log("-" * 100)
    
    log("Begin Training")
    log("X" * 100)
    start_module = time.time()
    trainer.train()
    log(f"Time taken for training: {time.time() - start_module}")
    log("X" * 100)

    log(f"Total Time taken: {time.time() - start}")
