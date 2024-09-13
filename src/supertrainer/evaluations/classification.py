import numpy as np
from sklearn.metrics import precision_recall_fscore_support

MAPPING_EN_TO_ID = {
    "positive": "positif",
    "negative": "negatif",
    "neutral": "netral",
}
MAPPING_ID_TO_EN = {v: k for k, v in MAPPING_EN_TO_ID.items()}


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# class SentAntBenchmark:
#     def __init__(
#         self,
#         dataset_path: str,
#         model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
#         max_length: int = 512,
#         stride: int = 256,
#     ):
#         self.dataset_path = dataset_path
#         self.model_name = model_name
#         self.max_length = max_length
#         self.stride = stride
#         self.dataset = None
#         self.model = None
#         self.tokenizer = None
#         self.config = None

#     def load_dataset(self):
#         self.dataset = Dataset.load_from_disk(self.dataset_path)
#         print(f"Dataset loaded: {self.dataset}")

#     def load_model_and_tokenizer(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
#         self.config = AutoConfig.from_pretrained(self.model_name)
#         self.model.eval()
#         print(f"Model and tokenizer loaded: {self.model_name}")

#     def sliding_window_classification(self, text: str, entity: str | None = None):
#         if entity is not None:
#             text = f"{text}: {entity}"  # Change template as needed

#         tokens = self.tokenizer.encode(text, add_special_tokens=False)

#         # Initialize the result arrays
#         all_logits = []
#         all_attention_masks = []

#         for i in range(0, len(tokens), self.stride):
#             chunk = tokens[i : i + self.max_length - 2]
#             input_ids = [self.tokenizer.cls_token_id] + chunk + [self.tokenizer.sep_token_id]
#             attention_mask = [1] * len(input_ids)

#             padding_length = self.max_length - len(input_ids)
#             input_ids += [self.tokenizer.pad_token_id] * padding_length
#             attention_mask += [0] * padding_length

#             input_ids = torch.tensor([input_ids])
#             attention_mask = torch.tensor([attention_mask])

#             with torch.no_grad():
#                 outputs = self.model(input_ids, attention_mask=attention_mask)

#             all_logits.append(outputs.logits)
#             all_attention_masks.append(attention_mask)

#         all_logits = torch.cat(all_logits, dim=0)
#         all_attention_masks = torch.cat(all_attention_masks, dim=0)

#         probs = F.softmax(all_logits, dim=-1)

#         weights = all_attention_masks.float().sum(dim=1) / all_attention_masks.float().sum()

#         weighted_probs = (probs * weights.unsqueeze(-1)).sum(dim=0)

#         predicted_class = torch.argmax(weighted_probs).item()

#         predicted_class = self.config.id2label[predicted_class]
#         predicted_class = MAPPING_EN_TO_ID[predicted_class]

#         return predicted_class, weighted_probs.tolist()

#     def classify_dataset(self):
#         if self.dataset is None:
#             raise ValueError("Dataset not loaded. Call load_dataset() first.")
#         if self.model is None or self.tokenizer is None:
#             raise ValueError(
#                 "Model or tokenizer not loaded. Call load_model_and_tokenizer() first."
#             )

#         self.predictions = []
#         for example in self.dataset:
#             text = example["content"]
#             predicted_class, probabilities = self.sliding_window_classification(
#                 text, entity=example.get("entity", None)
#             )
#             self.predictions.append(
#                 {
#                     "text": text,
#                     "predicted_class": predicted_class,
#                     "probabilities": probabilities,
#                 }
#             )

#         return self.predictions

#     def evaluate(self, label_field="label"):
#         if self.dataset is None:
#             raise ValueError("Dataset not loaded. Call load_dataset() first.")
#         if self.predictions is None:
#             raise ValueError("No predictions available. Call classify_dataset() first.")

#         true_labels = [example[label_field] for example in self.dataset]
#         predicted_labels = [pred["predicted_class"] for pred in self.predictions]

#         precision, recall, f1, _ = precision_recall_fscore_support(
#             true_labels, predicted_labels, average="weighted"
#         )
#         accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)

#         self.results = {
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "accuracy": accuracy,
#         }

#         print(f"Precision: {precision:.4f}")
#         print(f"Recall: {recall:.4f}")
#         print(f"F1 Score: {f1:.4f}")
#         print(f"Accuracy: {accuracy:.4f}")

#         return self.results

#     def save_results(self, save_path: str):
#         """Save the benchmark results to a file."""
#         if self.results is None:
#             print("No results available to save. Run evaluate() first.")
#             return

#         with open(save_path, "w") as f:
#             json.dump(self.results, f, indent=2)
#         print(f"Results saved to {save_path}")

#     def save_predictions(self, save_path: str):
#         """Save the model predictions to a file."""
#         if self.predictions is None:
#             print("No predictions available to save. Run classify_dataset() first.")
#             return

#         serializable_predictions = []
#         for pred in self.predictions:
#             serializable_pred = pred.copy()
#             serializable_pred["probabilities"] = [float(p) for p in pred["probabilities"]]
#             serializable_predictions.append(serializable_pred)

#         with open(save_path, "w") as f:
#             json.dump(serializable_predictions, f, indent=2)
#         print(f"Predictions saved to {save_path}")


# class SentAntBenchmarkOutlines:
#     def __init__(
#         self,
#         dataset_path: str,
#         model_name: str,
#         classes: list[str] | None = None,
#         device: str = "cuda",
#     ):
#         self.dataset_path = dataset_path
#         self.model_name = model_name
#         self.dataset = None
#         self._model = None
#         self._tokenizer = None
#         self.config = None
#         self.classes = classes or ["positif", "netral", "negatif"]
#         self.device = device

#     @property
#     def model(self) -> "outlines.models.transformers":
#         if self._model is None:
#             self._model = outlines.models.transformers(self.model_name, device=self.device)
#             generator = outlines.generate.choice(self._model, self.classes)
#             self._model = generator
#         return self._model

#     @property
#     def tokenizer(self) -> "AutoTokenizer":
#         if self._tokenizer is None:
#             self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         return self._tokenizer

#     def prepare_dataset(self) -> "DatasetDict":
#         logger.debug("Preparing dataset")
#         dataset_dict = load_dataset(
#             self.dataset_path,
#         )
#         logger.debug(f"Dataset loaded: {dataset_dict}")

#         if isinstance(dataset_dict, DatasetDict) and len(list(dataset_dict.keys())) < 2:
#             logger.debug("Splitting dataset")
#             dataset = self.split_dataset(dataset_dict)
#             logger.debug(f"Dataset split: {dataset}")
#         else:
#             dataset = dataset_dict

#         logger.debug("Formatting dataset for aspect sentiment analysis")
#         formatted_dataset = self.format_for_aspect_sentiment_analysis(dataset)
#         logger.debug(f"Dataset formatted: {formatted_dataset}")

#         logger.debug("Applying formatting prompts")
#         dataset = formatted_dataset.map(self.formatting_prompts_func, batched=True)
#         self.test_tokenization(dataset)

#         # Print some examples from the dataset to inspect the tokenizer's output
#         print("*** Example from the dataset ***")
#         for i in range(5):
#             print(f"Example {i+1}:")
#             self.print_text_after_substring(dataset["train"][i]["text"], "[/Judul]")
#             print("-" * 20)

#         logger.debug("Dataset preparation completed")

#         return dataset

#     def formatting_prompts_func(self, examples) -> dict[str, str]:
#         instructions = examples["instruction"]
#         inputs = examples["input"]
#         outputs = examples["output"]
#         texts = []
#         for instruction, input, output in zip(instructions, inputs, outputs):
#             chat = [
#                 {"role": "system", "content": f"{instruction}"},
#                 {"role": "user", "content": f"{input}"},
#                 # {"role": "assistant", "content": f"{output}"},
#             ]
#             text = self.tokenizer.apply_chat_template(
#                 chat, tokenize=False, add_generation_prompt=True
#             )
#             texts.append(text)
#         return {
#             "text": texts,
#         }

#     @staticmethod
#     def format_for_aspect_sentiment_analysis(dataset: DatasetDict) -> DatasetDict:
#         logger.debug("Formatting dataset for aspect sentiment analysis")

#         def format_example(example):
#             instruction = f"Tentukan sentimen (positif, netral, atau negatif) pada teks berikut dari sudut pandang {example['Entity']}"
#             input_text = f"\n[Judul]: {example['Title']}\n[/Judul]\n[Konten]: {example['Content']}\n[/Konten]"
#             # TODO: THIS IS HORRIBLE. I NEED TO SEPARATE DATASET INTO ONE CLASS AAAA
#             output_text = example["Entity Sentiment"]
#             return {
#                 "instruction": instruction,
#                 "input": input_text,
#                 "output": output_text,
#             }

#         formatted_dataset = dataset.map(format_example, batched=False)
#         logger.debug("Removing unnecessary columns")
#         formatted_dataset = DatasetDict(
#             {
#                 split: ds.remove_columns(
#                     [
#                         col
#                         for col in ds.column_names
#                         if col not in ["instruction", "input", "output", "label"]
#                     ]
#                 )
#                 for split, ds in formatted_dataset.items()
#             }
#         )
#         logger.debug("Dataset formatting for aspect sentiment analysis completed")
#         return formatted_dataset

#     def test_tokenization(self, dataset):
#         for i in range(5):
#             text = dataset["train"][i]["text"]
#             print(f"Original Text: {text}")
#             tokens = self.tokenizer.encode(text)
#             print(f"Tokenized: {tokens}")
#             detokenized_text = self.tokenizer.decode(tokens)
#             print(f"Detokenized Text: {detokenized_text}")
#             print("-" * 50)

#     def split_dataset(self, dataset_dict):
#         logger.debug("Splitting dataset into train, test, and validation sets")
#         train_test_split = dataset_dict["train"].train_test_split(test_size=0.2)
#         test_valid_split = train_test_split["test"].train_test_split(test_size=0.5)

#         if self.config.testing:
#             split_dataset = DatasetDict(
#                 {
#                     "train": train_test_split["train"]
#                     .shuffle(seed=42)
#                     .select(range(10)),  # Reduce train set size
#                     "test": test_valid_split["test"]
#                     .shuffle(seed=42)
#                     .select(range(5)),  # Reduce test set size
#                     "validation": Dataset.from_list(
#                         test_valid_split["train"]
#                         .shuffle(seed=42)
#                         .select(range(5))  # Reduce validation set size
#                     ),
#                 }
#             )
#         else:
#             logger.debug("Split dataset enter testing mode")
#             split_dataset = DatasetDict(
#                 {
#                     "train": train_test_split["train"].shuffle(seed=42),
#                     "test": test_valid_split["test"].shuffle(seed=42),
#                     "validation": Dataset.from_list(test_valid_split["train"].shuffle(seed=42)),
#                 }
#             )
#         logger.debug(f"Dataset split completed: {split_dataset}")

#         return split_dataset

#     def load_dataset(self):
#         logger.info("Loading dataset")
#         self.dataset = self.prepare_dataset()

#     def classify_dataset(self):
#         if self.dataset is None:
#             raise ValueError("Dataset not loaded. Call load_dataset() first.")

#         if isinstance(self.dataset, DatasetDict):
#             if "test" in self.dataset:
#                 dataset = self.dataset["test"]
#             else:
#                 raise ValueError("Test set not found in dataset")
#         else:
#             dataset = self.dataset

#         self.predictions = []
#         for example in dataset:
#             text = example["text"]

#             predicted_class = self.model(dedent_strip_limit_newlines(text))

#             self.predictions.append(
#                 {
#                     "text": text,
#                     "predicted_class": predicted_class,
#                 }
#             )

#         return self.predictions

#     def evaluate(self, label_field="label"):
#         if self.dataset is None:
#             raise ValueError("Dataset not loaded. Call load_dataset() first.")
#         if self.predictions is None:
#             raise ValueError("No predictions available. Call classify_dataset() first.")

#         if isinstance(self.dataset, DatasetDict):
#             if "test" in self.dataset:
#                 dataset = self.dataset["test"]
#             else:
#                 raise ValueError("Test set not found in dataset")
#         else:
#             dataset = self.dataset

#         true_labels = [example[label_field] for example in dataset]
#         predicted_labels = [pred["predicted_class"] for pred in self.predictions]

#         precision, recall, f1, _ = precision_recall_fscore_support(
#             true_labels, predicted_labels, average="weighted"
#         )
#         accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)

#         self.results = {
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "accuracy": accuracy,
#         }

#         print(f"Precision: {precision:.4f}")
#         print(f"Recall: {recall:.4f}")
#         print(f"F1 Score: {f1:.4f}")
#         print(f"Accuracy: {accuracy:.4f}")

#         return self.results

#     def save_results(self, save_path: str):
#         """Save the benchmark results to a file."""
#         if self.results is None:
#             print("No results available to save. Run evaluate() first.")
#             return

#         with open(save_path, "w") as f:
#             json.dump(self.results, f, indent=2)
#         print(f"Results saved to {save_path}")

#     def save_predictions(self, save_path: str):
#         """Save the model predictions to a file."""
#         if self.predictions is None:
#             print("No predictions available to save. Run classify_dataset() first.")
#             return

#         serializable_predictions = []
#         for pred in self.predictions:
#             serializable_pred = pred.copy()
#             serializable_predictions.append(serializable_pred)

#         with open(save_path, "w") as f:
#             json.dump(serializable_predictions, f, indent=2)
#         print(f"Predictions saved to {save_path}")

#     @staticmethod
#     def print_text_after_substring(text: str, substring: str):
#         """Prints the text after the first occurrence of the substring."""
#         index = text.find(substring)
#         if index != -1:
#             print(text[index + len(substring) :])
#         else:
#             print("Substring not found.")


# def main():
#     # benchmark = SentAntBenchmark(
#     #     "my_dataset/huggingface_dataset",
#     #     "cardiffnlp/twitter-xlm-roberta-base-sentiment",
#     # )
#     # benchmark.load_dataset()
#     # benchmark.load_model_and_tokenizer()
#     # benchmark.classify_dataset()
#     # benchmark.evaluate(label_field="sentiment")
#     # benchmark.save_results("results.json")
#     # benchmark.save_predictions("predictions.json")
#     benchmark = SentAntBenchmarkOutlines(
#         "masa-research/news_absa_v2_annotated_with_token_usage",
#         "Qwen/Qwen2-0.5B-Instruct",
#     )
#     benchmark.load_dataset()
#     benchmark.classify_dataset()
#     benchmark.evaluate(label_field="output")
#     benchmark.save_results("results_outlines-instruct.json")
#     benchmark.save_predictions("predictions_outlines-instruct.json")


# if __name__ == "__main__":
#     main()
