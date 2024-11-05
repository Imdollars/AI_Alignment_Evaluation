from abc import ABC, abstractmethod
import numpy as np
import torch
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
from datasets import load_dataset


class AI_Alignment_Evaluation(ABC):
    def __init__(self):
        self.all_hidden_state = []
        self.all_prob_vectors = []
        self.all_entropies = []

    @abstractmethod
    def pad_token_setting(self):
        pass

    @abstractmethod
    def input_generate(self):
        pass

    @abstractmethod
    def detect(self, generate_token_num=10):
        pass

    @abstractmethod
    def visualize(self):
        pass


class AIAlignmentEvaluation_from_pretrained(AI_Alignment_Evaluation):
    def __init__(self, model_id):
        super().__init__()
        self.all_hidden_states = []
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.inputs = self.tokenizer("", return_tensors="pt")

    def pad_token_setting(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def input_generate(self, use_datasets=False, dataset_name="web_nlg", input_text="The key to life is"):
        if use_datasets:
            dataset = load_dataset(dataset_name)
            text_samples = dataset['test']['text'][:10]
            self.inputs = self.tokenizer(text_samples, return_tensors='pt', padding=True, truncation=True)
        else:
            self.inputs: BatchEncoding = self.tokenizer(input_text, return_tensors="pt")

    def detect(self, generate_token_num=10):
        generated_ids = self.inputs.input_ids.clone()
        with torch.no_grad():
            for _ in range(generate_token_num):  # Control the number of generated tokens
                outputs = self.model(generated_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                next_token_logits = outputs.logits[:, -1, :]

                # 1. Record the embedding vectors of each layer
                token_hidden_states = [hs[0, -1, :].cpu().numpy() for hs in hidden_states]
                self.all_hidden_states.append(token_hidden_states)

                # 2. Calculate probability vectors and entropy
                probabilities = torch.softmax(next_token_logits, dim=-1).cpu().numpy()
                self.all_prob_vectors.append(probabilities[0])  # Record the probability vector for each layer

                # Calculate entropy for each layer
                layer_entropies = []
                for hidden_state in hidden_states:
                    layer_probabilities = torch.softmax(hidden_state[0, -1, :], dim=-1).cpu().numpy()
                    entropy = -np.sum(layer_probabilities * np.log(layer_probabilities + 1e-12))
                    layer_entropies.append(entropy)
                self.all_entropies.append(layer_entropies)

                # 3. Sample the next token and update the generated IDs
                next_token_id = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

    def visualize(self):
        # Convert data format for plotting
        all_entropies = np.array(self.all_entropies).T  # Transpose to get [num_layers, num_generated_tokens]

        # Check and print entropy data for each layer
        print("All Entropies Shape:", all_entropies.shape)  # Should be (num_layers, num_generated_tokens)
        for i, entropy in enumerate(all_entropies):
            print(f"Layer {i} Entropy:", entropy)

        # Visualization: entropy change across generated token positions for each layer
        plt.figure(figsize=(12, 6))
        for i in range(all_entropies.shape[0]):  # Iterate over each layer
            plt.plot(all_entropies[i], label=f"Layer {i}")
        plt.xlabel("Generated Token Position")
        plt.ylabel("Entropy")
        plt.title("Entropy for Each Layer Across Generated Tokens")
        plt.legend(loc="upper right")
        plt.savefig("entropy_per_layer.png", format="png")  # Save entropy variation plot
        plt.show()

    def evaluate(self):
        self.pad_token_setting()
        self.input_generate()
        self.detect()
        self.visualize()


if __name__ == '__main__':
    test = AIAlignmentEvaluation_from_pretrained(model_id="gpt2")
    test.evaluate()
