#! pip install git+https://github.com/Lightning-AI/Finetune-miniLM
#! mkdir -p ${HOME}/data/yelpreviewfull
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/train.csv -o ${HOME}/data/yelpreviewfull/train.csv
#! curl https://s3.amazonaws.com/pl-flash-data/lai-llm/lai-text-classification/datasets/Yelp/datasets/YelpReviewFull/yelp_review_full_csv/test.csv -o ${HOME}/data/yelpreviewfull/test.csv
import lightning as L
import torch
from transformers import *
from sentence_transformers.models import Pooling

from finetune_minilm import *


class Embedding(L.LightningModule):
    """
    Loosely based on "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks": https://arxiv.org/abs/1908.10084

    Finetunes a Bert-based model (from HF) for classification by minimizing the cosine similarity loss of text pairs.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.pooling = Pooling(module.config.hidden_size)

    def forward(self, batch):
        output_states = self.module(**batch)
        output_tokens = output_states.last_hidden_state
        batch.update({"token_embeddings": output_tokens})
        return self.pooling(batch)["sentence_embedding"]

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        loss = pairwise_cosine_embedding_loss(embeddings, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        loss = pairwise_cosine_embedding_loss(embeddings, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=0.001)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=5, num_training_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]


class FinetuneEmbedding(L.LightningWork):
    def __init__(self, *args, tb_drive, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensorboard_drive = tb_drive

    def run(self):
        tokenizer = self.configure_tokenizer()
        train_dataloader = self.configure_data("~/data/yelpreviewfull/train.csv", tokenizer)
        val_dataloader = self.configure_data("~/data/yelpreviewfull/test.csv", tokenizer)
        lightning_module = Embedding(module=self.configure_module())

        trainer = L.Trainer(
            max_epochs=5,
            limit_train_batches=100,
            limit_val_batches=100,
            # strategy="ddp",  # FIXME
            precision=16,
            accelerator="auto",
            devices="auto",
            callbacks=self.configure_callbacks(),
            logger=False,  # FIXME DriveTensorBoardLogger(save_dir=".", drive=self.tensorboard_drive),
            log_every_n_steps=5,
        )
        trainer.fit(lightning_module, train_dataloader, val_dataloader)

    def configure_module(self):
        # https://github.com/microsoft/unilm/tree/master/minilm#english-pre-trained-models. 33M parameters
        return AutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased", output_hidden_states=True)

    def configure_tokenizer(self):
        return AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")

    def configure_data(self, path: str, tokenizer) -> torch.utils.data.DataLoader:
        # FIXME: batch size
        return TokenizedDataloader(dataset=TextDataset(csv_file=path), batch_size=8, tokenizer=tokenizer)

    def configure_callbacks(self):
        early_stopping = L.pytorch.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, verbose=True, mode="min")
        checkpoints = L.pytorch.callbacks.ModelCheckpoint(save_top_k=3, monitor="val_loss", mode="min")
        return [early_stopping, checkpoints]


# FIXME
# app = L.LightningApp(TrainerWithTensorboard(FinetuneEmbedding, L.CloudCompute("gpu-fast", disk_size=50)))
e = FinetuneEmbedding(tb_drive="foo")
e.run()
