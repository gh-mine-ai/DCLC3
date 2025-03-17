from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.segmentation import MeanIoU
from transformers import Mask2FormerForUniversalSegmentation


class BuildingSegModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        mask_threshold: int = 0.2,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # net: torch.nn.Module = DeepLabV3Plus_MiT(num_classes=1)
        # self.net = DeepLabV3Plus_MiT(num_classes=1)
        # self.net = net
        # load Mask2Former fine-tuned on Cityscapes semantic segmentation
        # we only need one class
        id2label =  {1: "building"}
        label2id = {label: id for id, label in id2label.items()}
        num_labels = len(id2label)
        self.net = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-small-cityscapes-semantic",
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            num_queries=1,
        )
        
        self.mask_threshold = mask_threshold
        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # metric objects for calculating and averaging IoU across batches
        self.train_iou = MeanIoU(num_classes=1)
        self.val_iou = MeanIoU(num_classes=1)
        self.test_iou = MeanIoU(num_classes=1)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_iou_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_iou.reset()
        self.val_iou_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        # Change the data to fit the huggingface input
        # monkey patching some code
        data_dict = {
            "pixel_values": batch[0],
            "pixel_mask": batch[1].squeeze(1),
        }
        outputs = self.net(**data_dict)
        # class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Scale back to preprocessed image size - (384, 384) for all models
        logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(256, 256), mode="bilinear", align_corners=False
        )

        # # Remove the null class `[..., :-1]`
        # masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        # masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]
        # segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

        # # resize the logits to be of the input shape of 256 x 256
        # logits = torch.nn.functional.interpolate(
        #                     segmentation, size=(256,256), mode="bilinear", align_corners=False
        #         )

        # logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = (torch.sigmoid(logits) > self.mask_threshold)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_iou(preds.int(), targets.int())
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_iou(preds.int(), targets.int())
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/iou", self.val_iou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        iou = self.val_iou.compute()  # get current val iou
        self.val_iou_best(iou)  # update best so far val iou
        # log `val_iou_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/iou_best", self.val_iou_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass
        # loss, preds, targets = self.model_step(batch)

        # # update and log metrics
        # self.test_loss(loss)
        # self.test_acc(preds, targets)
        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/iou",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = BuildingSegModule(None, None, None, None)
