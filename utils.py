import os, torch, random
import numpy as np


def count_params(model, all=False):
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model_weights(model, filename, verbose=1, cp_folder=""):
    if verbose:
        print(f"\n >>> Saving model to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def decode_output(output, offset, text):
    output = output.argmax(2).flatten().tolist()
    encoded_decode = dict()
    idx = 0
    while idx < len(output):
        if output[idx] != 0:
            idx2 = idx
            temp_entity = []
            predicted_class = output[idx2]
            while output[idx2] == predicted_class:
                temp_entity.append(idx2)
                idx2 += 1

            encoded_decode[predicted_class] = [offset[i] for i in temp_entity]

        idx += 1

    result = [" "] * 3
    for key in encoded_decode.keys():
        res = encoded_decode[key][0]
        res = text[0][res[0].item():res[1].item() + 1].strip()
        result[key - 1] = res
    return result