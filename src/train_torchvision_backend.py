# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# local imports (i.e. our own code)
# noinspection PyUnresolvedReferences
import utilities.setup_utils
from src.modules.torchvision_backend.recognition_module import RecognitionModule


if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        monitor="pretrain_loss",
        filename="detection-{epoch:02d}-{pretrain_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    # 2. learning rate monitor callback
    lr_logger = LearningRateMonitor(logging_interval="step", log_momentum=True)

    # defining the model
    recognition_module = RecognitionModule(
        batch_size=32,
        pretrained_model_path="resnet_backend.ckpt",
        crop_size=(128, 128),
        fine_tuning=True,
    )

    # print(detection_model)

    trainer = pl.Trainer(
        # fast_dev_run=True,
        max_epochs=100,
        precision=16,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, lr_logger],
        # limit_train_batches=0.001,
        # limit_test_batches=0.05,
        # limit_val_batches=0.001,
        log_every_n_steps=1,
        logger=WandbLogger(
            entity="mtp-ai-board-game-engine",
            project="cv-project",
            name="fine-tuning-crop-size-112",
            group="training-resnet-backbone-conv-transformer",
            log_model="all",
        ),
        # auto_scale_batch_size=True,
        # auto_lr_find=True,
        accelerator="gpu",
        devices=[0, 1],
    )

    trainer.fit(model=recognition_module)
