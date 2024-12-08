import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import json
import os
import time
from tqdm.auto import tqdm
import ipywidgets as widgets
from IPython.display import display
import re 

class ModelModifier:
    def __init__(self, model_name=None, top_percent=50, batch_size=1):
        self.model_name = model_name
        self.top_percent = top_percent
        self.batch_size = batch_size
        
        if model_name:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float32, 
                    low_cpu_mem_usage=True, 
                    trust_remote_code=True, 
                    device_map="auto"
                )
            except KeyError as e:
                print(f"Error loading model: {e}")
                print("Attempting to load with custom configuration...")
                config = AutoConfig.from_pretrained(model_name)
                config.rope_scaling = {"type": "linear", "factor": 1.0}
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto"
                )
            
            if not hasattr(self.model.config, 'rope_scaling'):
                self.model.config.rope_scaling = {'type': 'linear'}
            elif not isinstance(self.model.config.rope_scaling, dict):
                self.model.config.rope_scaling = {'type': 'linear'}
            elif 'type' not in self.model.config.rope_scaling:
                self.model.config.rope_scaling['type'] = 'linear'
        else:
            self.model = None

        self.layer_snr = {}
        self.layer_types = []

    def get_weight_types(self):
        weight_types = set()
        for name, module in self.model.named_modules():
            parts = name.split('.')
            if any(hasattr(module, attr) for attr in ['weight', 'bias','inv_freq']):
                layer_index = next((i for i, part in enumerate(parts) if part.isdigit()), -1)
                weight_type = '.'.join(parts[layer_index + 1:]) if layer_index != -1 else name
                weight_types.add(weight_type)
        return list(weight_types)

    
    def sort_weight_types(self, weight_types):
        categories = {}
        for wt in weight_types:
            category = wt.split('.')[0]
            categories.setdefault(category, []).append(wt)
        sorted_categories = {k: sorted(v) for k, v in sorted(categories.items(), key=lambda item: item[0])}
        sorted_weight_types = [wt for sublist in sorted_categories.values() for wt in sublist]
        return sorted_weight_types

    def calculate_snr_for_layer(self, layer_type):
        layers = [(name, module) for name, module in self.model.named_modules() if layer_type in name and hasattr(module, 'weight')]
        num_batches = (len(layers) + self.batch_size - 1) // self.batch_size

        with tqdm(total=num_batches, unit='batch', desc=f'Calculating SNR for {layer_type}') as progress_bar:
            for i in range(0, len(layers), self.batch_size):
                batch_layers = layers[i:i + self.batch_size]
                for name, module in batch_layers:
                    try:
                        # Keep computation on GPU but use efficient memory management
                        weights = module.weight.detach()
                        if weights.ndim < 2:
                            weights = weights.unsqueeze(0)
                            
                        # Use torch.cuda.empty_cache() before heavy computation
                        torch.cuda.empty_cache()
                        
                        # Compute SVD values directly on GPU
                        S = torch.linalg.svdvals(weights)
                        max_singular_value = S[0]
                        sigma_estimated = self.estimate_sigma_with_full_iqr(S)
                        n, m = weights.shape[-2:]
                        mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)
                        signal = S[S > mp_threshold].sum()
                        noise = S[S <= mp_threshold].sum()
                        snr = signal / noise if noise != 0 else float('inf')
                        snr_ratio = snr / max_singular_value
                        self.layer_snr[name] = {'type': layer_type, 'snr': float(snr_ratio)}
                        
                        # Clean up memory immediately
                        del weights, S
                        torch.cuda.empty_cache()
                        
                    except RuntimeError as e:
                        print(f"\nWarning: Error processing layer {name}: {str(e)}")
                        try:
                            # Fallback to CPU only if GPU fails
                            weights = module.weight.detach().cpu()
                            S = torch.linalg.svdvals(weights)
                            max_singular_value = S[0]
                            sigma_estimated = self.estimate_sigma_with_full_iqr(S)
                            n, m = weights.shape[-2:]
                            mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)
                            signal = S[S > mp_threshold].sum()
                            noise = S[S <= mp_threshold].sum()
                            snr = signal / noise if noise != 0 else float('inf')
                            snr_ratio = snr / max_singular_value
                            self.layer_snr[name] = {'type': layer_type, 'snr': float(snr_ratio)}
                            
                            del weights, S
                            torch.cuda.empty_cache()
                        except Exception as e2:
                            print(f"\nWarning: Could not process layer {name} even with CPU fallback: {str(e2)}")
                            continue
                            
                progress_bar.update(1)

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta)) ** 2)
        return threshold

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349
        return sigma_estimated

    def assess_layers_snr(self, selected_weight_types):
        total_layers = sum(1 for name, module in self.model.named_modules() if any(layer_type in name for layer_type in selected_weight_types) and hasattr(module, 'weight'))
        start_time = time.time()

        with tqdm(total=len(selected_weight_types), unit='type', desc='Calculating SNR for types') as progress_bar:
            for layer_type in selected_weight_types:
                self.calculate_snr_for_layer(layer_type)
                progress_bar.update(1)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken: {total_time:.2f} seconds")

    def save_snr_to_json(self):
        model_name_slug = self.model_name.replace('/', '-').replace('_', '-')
        directory = 'model_snr_results'
        filename = os.path.join(directory, f'snr_results_{model_name_slug}.json')
        
        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        serializable_data = {}
        for layer_name, info in self.layer_snr.items():
            snr_value = info['snr'].item() if isinstance(info['snr'], torch.Tensor) else info['snr']
            layer_type = str(info['type'])
            serializable_data[layer_name] = {'snr': snr_value, 'type': layer_type}
        
        with open(filename, 'w') as file:
            json.dump(serializable_data, file, indent=4)
        
        print(f"Results saved to {filename}")
        self.save_top_snr_ratios_to_json(filename)
        self.generate_unfrozen_params_yaml(filename)

    def generate_unfrozen_params_yaml(self, json_filename, top_percent=None):
        top_percent = top_percent if top_percent is not None else self.top_percent
        with open(json_filename, 'r') as file:
            snr_data = json.load(file)
        unfrozen_parameters = {}
        for layer_name, info in snr_data.items():
            layer_type = info['type']
            if layer_type not in unfrozen_parameters:
                unfrozen_parameters[layer_type] = []
            unfrozen_parameters[layer_type].append((layer_name, info['snr']))
        top_layers_by_type = {}
        for layer_type, layers in unfrozen_parameters.items():
            layers_sorted = sorted(layers, key=lambda x: x[1], reverse=True)
            num_top_layers = int(len(layers) * top_percent / 100)
            top_layers_by_type[layer_type] = [layer[0] for layer in layers_sorted[:num_top_layers]]
        # Modify the yaml_filename to include the input json name and top_percent
        json_file_base = os.path.splitext(os.path.basename(json_filename))[0]
        yaml_filename = f"{json_file_base}_unfrozenparameters_{top_percent}percent.yaml"
        with open(yaml_filename, 'w') as file:
            file.write("unfrozen_parameters:\n")
            file.write("- ^lm_head.weight$\n")
            file.write("- ^model.embed_tokens.weight$\n")
            for layer_type, layer_names in top_layers_by_type.items():
                file.write(f"# {layer_type} layers\n")
                for layer_name in layer_names:
                    file.write(f"- {layer_name}\n")
        print(f"Top {top_percent}% SNR layers saved to {yaml_filename}")

    def save_top_snr_ratios_to_json(self, json_filename, filename=None):
        with open(json_filename, 'r') as file:
            snr_data = json.load(file)
        all_snr_layers = {}
        for layer_name, info in snr_data.items():
            layer_type = info['type']
            if layer_type not in all_snr_layers:
                all_snr_layers[layer_type] = []
            all_snr_layers[layer_type].append((layer_name, info['snr']))
        for layer_type, layers in all_snr_layers.items():
            layers_sorted = sorted(layers, key=lambda x: x[1], reverse=True)
            all_snr_layers[layer_type] = {layer[0]: layer[1] for layer in layers_sorted}

        json_file_base = os.path.splitext(os.path.basename(json_filename))[0]
        filename = f"{json_file_base}_sorted.json" if filename is None else filename

        with open(filename, 'w') as file:
            json.dump(all_snr_layers, file, indent=4)
        print(f"All SNR layers sorted and saved to {filename}")

    def visualize_snr_distribution(self):
        """Visualize the SNR distribution of layers"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare data for plotting
            types_snr = {}
            for layer_name, info in self.layer_snr.items():
                layer_type = info['type']
                if layer_type not in types_snr:
                    types_snr[layer_type] = []
                types_snr[layer_type].append(info['snr'])
            
            # Create plot
            plt.figure(figsize=(12, 6))
            for layer_type, snr_values in types_snr.items():
                sns.kdeplot(data=snr_values, label=layer_type)
            
            plt.title('SNR Distribution by Layer Type')
            plt.xlabel('SNR Ratio')
            plt.ylabel('Density')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Please install matplotlib and seaborn to visualize results")


def analyze_model(model_name, top_percent=50, batch_size=1, weight_to_snr=None):
    """
    Notebook-friendly function to analyze a model's SNR
    """
    print(f"Analyzing model: {model_name}")
    print(f"Top percent: {top_percent}")
    print(f"Batch size: {batch_size}")
    
    # Check for existing SNR results file
    model_name_slug = model_name.replace('/', '-').replace('_', '-')
    snr_file_path = os.path.join('model_snr_results', f'snr_results_{model_name_slug}.json')

    if os.path.exists(snr_file_path):
        print(f"Found existing SNR results file for {model_name}")
        modifier = ModelModifier(top_percent=top_percent)
        modifier.generate_unfrozen_params_yaml(snr_file_path, top_percent)
    else:
        print(f"No existing SNR results file found for {model_name}. Proceeding with SNR calculation.")
        modifier = ModelModifier(model_name=model_name, batch_size=batch_size)
        
        # Get weight types
        all_weight_types = modifier.get_weight_types()
        print(f"Found {len(all_weight_types)} weight types")
        
        # Check if weight_to_snr is provided
        if weight_to_snr:
            if isinstance(weight_to_snr, str):
                selected_weight_types = [weight_to_snr] if weight_to_snr in all_weight_types else []
            else:
                selected_weight_types = [wt for wt in weight_to_snr if wt in all_weight_types]
        else:
            selected_weight_types = all_weight_types
        
        if selected_weight_types:
            modifier.assess_layers_snr(selected_weight_types)
            modifier.save_snr_to_json()
            
            # Visualize results
            modifier.visualize_snr_distribution()
            print("Finished SNR scanning and data saved.")
        else:
            print("No valid weight types selected. Please provide valid weight types.")
            
    return modifier

import re 
def get_spectrum(model, top_percent=50, batch_size=1, weight_to_snr=None):
    """
    Analyze model and apply Spectrum freezing/unfreezing based on SNR analysis
    Returns the modified model with frozen/unfrozen parameters
    
    Args:
        model: The pre-loaded model object
        top_percent: Percentage of top SNR layers to unfreeze
        batch_size: Batch size for analysis
        weight_to_snr: List of weight types to analyze
    """
    import copy
    
    # Get model name from config
    model_name = model.config._name_or_path
    modifier = ModelModifier(model_name=model_name, top_percent=top_percent, batch_size=batch_size)
    
    # Instead of creating a new model, use the existing one
    modifier.model = model
    
    if weight_to_snr:
        modifier.assess_layers_snr(weight_to_snr)
        modifier.save_snr_to_json()
    
    # Get model name slug for file paths
    model_name_slug = model_name.replace('/', '-').replace('_', '-')
    yaml_file = f"snr_results_{model_name_slug}_unfrozenparameters_{top_percent}percent.yaml"
    
    # Print total parameters before freezing
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameter Analysis:")
    print(f"Total Parameters: {total_params:,}")
    
    try:
        with open(yaml_file, "r") as fin:
            yaml_parameters = fin.read()

        # Extract unfrozen parameters
        unfrozen_parameters = []
        for line in yaml_parameters.splitlines():
            if line.startswith("- "):
                unfrozen_parameters.append(line.split("- ")[1])

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze Spectrum parameters and count
        unfrozen_count = 0
        for name, param in model.named_parameters():
            if any(re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters):
                param.requires_grad = True
                unfrozen_count += param.numel()
                
        print(f"\nSpectrum Freezing Results:")
        print(f"Total Parameters (unchanged): {total_params:,}")
        print(f"├── Frozen (non-trainable): {(total_params - unfrozen_count):,} ({100 * (total_params - unfrozen_count) / total_params:.2f}%)")
        print(f"└── Unfrozen (trainable): {unfrozen_count:,} ({100 * unfrozen_count / total_params:.2f}%)")
        
        print(f"\nUnfrozen Layers (trainable during fine-tuning):")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"- {name}: {param.numel():,} parameters")
                
    except FileNotFoundError:
        print(f"Warning: YAML file {yaml_file} not found. Model parameters remain unchanged.")
    except Exception as e:
        print(f"Warning: Error applying Spectrum: {str(e)}. Model parameters remain unchanged.")
    
    return model