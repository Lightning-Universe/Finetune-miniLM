#! pip install git+https://github.com/Lightning-AI/Finetune-miniLM
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/train.csv --create-dirs -o ${HOME}/data/yelpreviewfull/train.csv -C -
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/test.csv --create-dirs -o ${HOME}/data/yelpreviewfull/test.csv -C -
import lightning as L
import torch
from transformers import *
from finetune_minilm import *


class EmbeddingSimilarity(L.LightningModule):
    """
    Loosely based on "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks": https://arxiv.org/abs/1908.10084

    Finetunes a Bert-based model (from HF) for classification by minimizing the cosine similarity loss of text pairs.
    """

    def __init__(self, module: TextEmbedder):
        super().__init__()
        self.module = module

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self.module(x)
        loss = pairwise_cosine_embedding_loss(embeddings, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self.module(x)
        loss = pairwise_cosine_embedding_loss(embeddings, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=0.001)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=5, num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]


class Finetune(L.LightningWork):
    def __init__(self, *args, tb_drive, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorboard_drive = tb_drive

    def run(self):
        warn_if_drive_not_empty(self.tensorboard_drive)

        L.seed_everything(777, workers=True)
        tokenizer = self.configure_tokenizer()
        train_dataloader = self.configure_data("~/data/yelpreviewfull/train.csv", tokenizer)
        val_dataloader = self.configure_data("~/data/yelpreviewfull/test.csv", tokenizer)
        lightning_module = self.configure_module()

        trainer = L.Trainer(
            max_epochs=5,
            limit_train_batches=100,
            limit_val_batches=100,
            strategy="ddp_find_unused_parameters_false",
            precision=16,
            accelerator="auto",
            devices="auto",
            callbacks=self.configure_callbacks(),
            logger=DriveTensorBoardLogger(save_dir=".", drive=self.tensorboard_drive),
            log_every_n_steps=5,
        )
        trainer.fit(lightning_module, train_dataloader, val_dataloader)

    def configure_module(self) -> L.LightningModule:
        # https://github.com/microsoft/unilm/tree/master/minilm#english-pre-trained-models. 33M parameters
        module = TextEmbedder(backbone="microsoft/MiniLM-L12-H384-uncased")
        return EmbeddingSimilarity(module)

    def configure_tokenizer(self):
        return AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")

    def configure_data(self, path: str, tokenizer) -> torch.utils.data.DataLoader:
        return TokenizedDataloader(dataset=TextDataset(csv_file=path), batch_size=16, shuffle=True, tokenizer=tokenizer)

    def configure_callbacks(self):
        early_stopping = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, verbose=True, mode="min")
        checkpoints = L.pytorch.callbacks.ModelCheckpoint(save_top_k=3, monitor="val_loss", mode="min")
        return [early_stopping, checkpoints]


app = L.LightningApp(TrainerWithTensorboard(Finetune, L.CloudCompute("gpu-fast", disk_size=50)))
