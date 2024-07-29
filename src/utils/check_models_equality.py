import torch
from transformers import OPTForCausalLM

def check_models_equality(subset: str, model1_path: str, model2_path: str) -> None:
    """
    Check if two models have identical weights.

    Args:
        subset (str): The subset name for which the models are being compared.
        model1_path (str): Path to the first model.
        model2_path (str): Path to the second model.
    """
    try:
        model1 = OPTForCausalLM.from_pretrained(model1_path)
        model2 = OPTForCausalLM.from_pretrained(model2_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    weights_equal = all(torch.equal(state_dict1[key], state_dict2[key]) for key in state_dict1)

    if weights_equal:
        print(f'The models for subset "{subset}" have identical weights.')
    else:
        print(f'The models for subset "{subset}" have different weights.')

        
if __name__ == '__main__':
    subset = 'aspirin_0.4'
    model1_path = f'/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13_my/src/checkpoints/fine_tuned/OPT_1.2B_ep_1_all_rand_finetune_all_canon_{subset}_sm_1000K_4.00E-05/checkpoint-3900'
    model2_path = f'/mnt/2tb/chem/hasmik/GDB_Generation_hf_project/Molecular_Generation_with_GDB13_my/src/checkpoints/fine_tuned/OPT_1.2B_ep_1_all_rand_finetune_all_rand_{subset}_sm_1000K_4.00E-05/checkpoint-3900'
    check_models_equality(subset=subset, model1_path=model1_path, model2_path=model2_path)

