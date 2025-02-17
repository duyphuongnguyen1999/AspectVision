from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments, pipeline


class AspectExtractionModel:
    def __init__(self, model_name="distilbert/distilbert-base-uncased", num_labels=3):
        """
        Initialize Aspect Term Extraction Model

        Args:
            model_name: Pre-trained model name
            num_labels: Number of labels (O, B-Term, I-Term))
        """

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def train(
        self, train_dataloader, val_dataloader, num_epochs=100, learning_rate=2e-5
    ):
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=learning_rate,
            per_device_train_batch_size=16,
            per_gpu_eval_batch_size=16,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataloader,
            eval_dataset=val_dataloader,
        )

        trainer.train()

    def predict(self, text):
        nlp_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
        )
        return nlp_pipeline(text)

    def save_pretrain(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_pretrained(self, load_path):
        self.model = AutoModelForTokenClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)


# ====================== MAIN FUNCTION ======================


def main():
    """
    Quick test AspectExtractionModel
    """
    model_path = "models/ate_model"
    ate_model = AspectExtractionModel()
    ate_model.load_pretrained(model_path)

    test_sentence = "The battery life is great but the screen quality is poor."
    aspects = ate_model.predict(test_sentence)

    print("Predicted Aspect Terms:")
    for aspect in aspects:
        print(
            f"Word: {aspect['word']}, Score: {aspect['score']:.2f}, Label: {aspect['entity_group']}"
        )


if __name__ == "__main__":
    main()
