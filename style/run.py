import os, torch
from .model import StyleEstimator
from .lookup import Lookup

def load_style_resources(model_folder, model_name, lookup_folder):
    # load lookup
    
    lookup = Lookup(type="bpe")
    lookup.load(file_prefix=os.path.join(lookup_folder, "bpe", "tok"))

    # load model
    model = StyleEstimator.load_from_checkpoint(os.path.join(model_folder,model_name), tags_csv=os.path.join(model_folder,"meta_tags.csv"))
    model.eval()

    return model, lookup

def get_styles(sentences, model, lookup, bw_context_len=5, fw_context_len=3):
    X = []
    # sliding window
    for i in range(len(sentences)):
        if i < bw_context_len:
            X.append([''] * (bw_context_len - i) + sentences[0:fw_context_len + 1 + i])
            c_len = len(X[-1])
            if c_len < bw_context_len + 1 + fw_context_len: # maybe there are not enough sentences left to the right
                X[-1] = X[-1] + [''] * (bw_context_len + 1 + fw_context_len - c_len)

            print("A:{}".format(len(X[-1])))
        elif i >= len(sentences) - fw_context_len:
            # print("@{}/{}".format(i, len(xs)))
            remaining = len(sentences) - i - 1
            X.append(sentences[i - bw_context_len:i + remaining + 1] + [''] * (fw_context_len - remaining))
            # print(X[-1])
            # print(len(X[-1]))
            # input("?? {}/len{}".format(i, len(xs)))
            print("B:{}".format(len(X[-1])))
        else:
            X.append(sentences[i - bw_context_len:i + fw_context_len + 1])
            print("C:{}".format(len(X[-1])))

    # encode as ints with the lookup object
    X_temp = []
    for instance in X:
        ins = []
        print("    >>>>>>>>>>>>")
        for en, sentence in enumerate(instance):
            print("{}: {}".format(en,sentence))
            ins.append(lookup.encode(sentence, add_bos_eos_tokens=True))
        X_temp.append(ins)
    X = X_temp

    # convert to tensors
    # get max lengths
    max_len = 0
    for instance in X:
        max_len = max([max_len] + [len(sentence) for sentence in instance])

    x_tensor = []  # should be [batch_size, no_of_sentences, max_len], where no_of_sentence = bw + 1 + fw
    x_lengths = []  # tensor of [batch_size * no_of_sentences]
    for instance in X:
        instance_tensor = []
        for sentence in instance:
            sentence_len = len(sentence)
            x_lengths.append(sentence_len)
            padded_sentence = torch.tensor(sentence + [0] * (max_len - sentence_len), dtype=torch.long)
            instance_tensor.append(padded_sentence.unsqueeze(0))  # lista cu [1, max_len]
        instance_tensor = torch.cat(instance_tensor, dim=0)  # [no_of_sentences, max_len]
        x_tensor.append(instance_tensor.unsqueeze(0))  # list of [1, no_of_sentences, max_len]
    x_tensor = torch.cat(x_tensor, dim=0)  # [batch_size, no_of_sentences, max_len]

    # print(x_tensor.size())
    # return x_tensor, x_lengths
    return model(x_tensor, x_lengths)

if __name__ == "__main__":
    """
    model, lookup = load_style_resources("/home/echo/work/ebooks/eBookService/style/experiments/lightning_logs/version_27-04-2020--17-28-09","_ckpt_epoch_54.ckpt","lookup")

    sentences = ["This is a test", "Wow, no more tests?", "This is another test.", "I would like this to sound different, wouldn't you?","Yes I would, said nobody", "`Then what's the fuss about`, said the ginger man", "'Nothing, shut up and go back from where you came from' said a red, tired, smelly mouse."]

    style = get_styles(sentences, model, lookup)

    print(">>>")
    print(style)
    """
    pass