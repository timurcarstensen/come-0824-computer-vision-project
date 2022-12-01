# third party imports
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# local imports (i.e. our own code)
# noinspection PyUnresolvedReferences
import utils.utils
from utils.datasets import PretrainDataset
from src.modules.pl_original_models.lit_detection import LitDetectionModule

if __name__ == "__main__":
    # defining callbacks
    # 1. checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="pretrain_loss",
        filename="detection-{epoch:02d}-{pretrain_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    # 2. learning rate monitor callback
    lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=True)
    batch_size = 16
    # defining the model
    detection_model = LitDetectionModule(
        pretrain_set=PretrainDataset(split_file=["train.txt"]),
        batch_size=batch_size,
        num_points=4,
    )

    # try:
    #   detection_model.load_state_dict(torch.load("model_weights/fh02.pth"))
    #   print("Model loaded")
    # except Exception:
    #    print("Model not found")

    # TODO: uncomment this to use pretrained network
    """
    # get state dict of detection model
    state_dict = detection_model.state_dict()
    print(state_dict.keys())

    model_weights = torch.load("model_weights/wR2.pth")
    # print(model_weights.keys())

    # some keys in the model_weights dictionary start with "module.wR2.module.", some start only with "module.". Remove these parts
    model_weights = {k.replace("module.wR2.module.", ""): v for k, v in model_weights.items()}
    model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}

    # if there is a relu in our model, we have to rename some model_weight keys
    # rename keys 'classifier.1.weight', 'classifier.1.bias', 'classifier.2.weight', 'classifier.2.bias' to 'classifier.2.weight', 'classifier.2.bias', 'classifier.3.weight', 'classifier.3.bias'
    help_a = model_weights["classifier.2.weight"]
    help_b = model_weights["classifier.2.bias"]
    model_weights["classifier.2.weight"] = model_weights["classifier.1.weight"]
    model_weights["classifier.2.bias"] = model_weights["classifier.1.bias"]
    model_weights["classifier.4.weight"] = help_a
    model_weights["classifier.4.bias"] = help_b
    del model_weights["classifier.1.weight"]
    del model_weights["classifier.1.bias"]

    # remove key "sas" from model_weights
    print(model_weights.keys())

    # print all keys which are in the state_dict but not in the model_weights
    print("Keys in state_dict but not in model_weights:")
    for key in state_dict.keys():
        if key not in model_weights.keys():
            print(key)

    print("\nKeys in model_weights but not in state_dict:")
    for key in model_weights.keys():
        if key not in state_dict.keys():
            print(key)

    detection_model.load_state_dict(model_weights)
    """
    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=300,
        callbacks=[checkpoint_callback, lr_logger],
        logger=WandbLogger(
            entity="mtp-ai-board-game-engine",
            project="cv-project",
            group=f"no_pretraining_relu_adam_batch={batch_size}",
            log_model=True,
        ),
        auto_scale_batch_size=True,
        auto_lr_find=True,
        accelerator="gpu",
        devices=[1, 2, 5, 6, 7],
    )
    trainer.fit(model=detection_model)
