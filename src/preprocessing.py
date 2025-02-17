from transformers import AutoTokenizer


class Preprocessor:
    def __init__(self, model_name="distilbert/distilbert-base-uncased"):
        """
        Initialize tokenizer for Preprocessor

        Args:
            model_name (str, optional): Pre-trained model name. Defaults to "distilbert/distilbert-base-uncased".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align_lables(self, examples):
        """
        Tokenize the text and align labels to match BERT's tokens.
        """
        tokenized_input = []
        labels = []

        for tokens, tags in zip(examples["Tokens"], examples["Tags"]):
            bert_tokens = []
            bert_tags = []

            for token, tag in zip(tokens, tags):
                # Tokenize each word into sub-tokens
                sub_tokens = self.tokenizer.tokenize(token)
                bert_tokens.extend(sub_tokens)

                # Repeat the tag for each sub-token
                bert_tags.extend([int(tag)] * len(sub_tokens))

            # Convert the tokenized tokens into ids
            bert_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
            tokenized_input.append(bert_ids)
            labels.append(bert_tags)

        return {"input_ids": tokenized_input, "labels": labels}


if __name__ == "__main__":
    """
    Test the Preprocessor module.
    """
    preprocessor = Preprocessor()

    # Sample input for testing
    example = {
        "Tokens": ["But", "the", "staff", "was", "so", "horrible", "to", "us", "."],
        "Tags": ["0", "0", "1", "0", "0", "0", "0", "0", "0"],
    }

    preprocessed = preprocessor.tokenize_and_align_lables(example)

    print("Original Example:\n", example)
    print("Processed Example:\n", preprocessed)
