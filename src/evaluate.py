    from time import time
    from datasets import load_dataset

class PerformanceBenchmark:
    """
    A class to benchmark the performance of a model on a given dataset.
    
    Attributes:
    -----------
    model : transformers.PreTrainedModel
        The model to be benchmarked.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer associated with the model.
    dataset : datasets.Dataset
        The dataset on which the model's performance will be evaluated.
    """
    
    def __init__(self, model, tokenizer, dataset):
        """
        Initializes the PerformanceBenchmark with the provided model, tokenizer, and dataset.
        
        Parameters:
        -----------
        model : transformers.PreTrainedModel
            The model to be benchmarked.
        tokenizer : transformers.PreTrainedTokenizer
            The tokenizer for encoding the inputs for the model.
        dataset : datasets.Dataset
            The dataset on which the model's performance will be evaluated.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset

    def compute_parameters(self):
        """
        Computes the total number of parameters and the number of trainable parameters.
        
        Returns:
        --------
        dict :
            A dictionary containing:
            - `total_params`: The total number of parameters in the model.
            - `trainable_params`: The number of trainable parameters in the model.
        """
        total_params = sum(p.numel() for p in self.model.parameters())  # Total parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)  # Trainable parameters
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params
        }

    def compute_size(self):
        """
        Computes the size of the model in terms of the number of parameters 
        and memory usage in megabytes (MB).

        Returns:
        --------
        dict :
            A dictionary containing the number of parameters (`num_params`) and 
            the model size in MB (`model_size_mb`).
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = sum(p.element_size() * p.nelement() for p in self.model.parameters()) / (1024**2)
        
        return {"num_params": num_params, "model_size_mb": model_size_mb}

    def time_pipeline(self):
        """
        Measures the total time and average time taken by the model to process 
        the dataset.
        
        This method will use the tokenizer to encode the inputs before passing them 
        to the model.

        Returns:
        --------
        dict :
            A dictionary containing the total processing time in seconds (`total_time_sec`) 
            and the average time per example (`avg_time_per_example_sec`).
        """
        start_time = time.time()
        
        for example in self.dataset:
            inputs = example['conversations']
            # Tokenize the input
            tokenized_input = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)
            _ = self.model.generate(**tokenized_input, max_new_tokens=10)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_example = total_time / len(self.dataset) if len(self.dataset) > 0 else float('inf')
        
        return {"total_time_sec": total_time, "avg_time_per_example_sec": avg_time_per_example}

    def compute_latency(self):
        """
        Computes the average latency of the model, defined as the time taken 
        to process a single example from the dataset.

        Returns:
        --------
        dict :
            A dictionary containing the average latency in seconds (`avg_latency_sec`).
        """
        latencies = []
        
        for example in self.dataset:
            inputs = example['conversations']
            # Tokenize the input
            tokenized_input = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)
            
            start_time = time.time()
            _ = self.model.generate(**tokenized_input, max_new_tokens=10)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        avg_latency = sum(latencies) / len(latencies) if len(latencies) > 0 else float('inf')
        return {"avg_latency_sec": avg_latency}

    def compute_throughput(self):
        """
        Computes the throughput of the model, defined as the number of examples 
        processed per second.

        Returns:
        --------
        dict :
            A dictionary containing the throughput in examples per second (`throughput_examples_per_sec`).
        """
        start_time = time.time()
        
        for example in self.dataset:
            inputs = example['conversations']
            # Tokenize the input
            tokenized_input = self.tokenizer(inputs, return_tensors="pt").to(self.model.device)
            _ = self.model.generate(**tokenized_input, max_new_tokens=10)
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = len(self.dataset) / total_time if total_time > 0 else 0
        
        return {"throughput_examples_per_sec": throughput}
    

    def run_benchmark(self):
        """
        Runs all the benchmark metrics (size, time, latency, throughput, and FLOPs) 
        and returns the results.

        Returns:
        --------
        dict :
            A dictionary containing all the computed metrics for the model. 
            Includes size, parameters, time, latency, throughput, and FLOPs estimates.
        """
        metrics = {}
        metrics['Size'] = self.compute_size()
        metrics['Parameters'] = self.compute_parameters()
        metrics['Time'] = self.time_pipeline()
        metrics['Latency'] = self.compute_latency()
        metrics['Throughput'] = self.compute_throughput()
        return metrics



# Create the main code to run the benchmarking 

if __name__=="__main__":

    # Get model, tokenizer and dataset 

    # TODO


    # Instantiate the PerformanceBenchmark class with the model, tokenizer, and test dataset
    benchmark = PerformanceBenchmark(model, tokenizer, dataset['test'])

    # Run the benchmark to compute performance metrics
    results = benchmark.run_benchmark()

    # Display the benchmark results
    print(results)