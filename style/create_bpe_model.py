import os, sys, json, pickle, ntpath
import numpy as np
from lookup import Lookup
import sentencepiece as spm


vocab_size = 1024

output_lookup_folder = os.path.join("lookup","bpe")

folder_path = "data/gsts"
bw_context_len=5
fw_context_len=3

# create data files
all_files = [x for x in os.listdir(folder_path) if not os.path.isdir(x)]
text_files = [x for x in all_files if x.endswith(".txt")]
npy_files = [x for x in all_files if x.endswith(".npy")]
assert len(text_files) == len(npy_files), "File error"
print("Loading {} files...".format(len(text_files)))

speakers = {}
for f in text_files:
    _, name = ntpath.split(f)
    if "_" in name:
        parts = name.split("_")
    else:
        parts = name.split("-")
    if parts[0] not in speakers:
        speakers[parts[0]] = []
    speakers[parts[0]].append(os.path.join(folder_path, f))

# create BPE model
print("Creating BPE model...")
all_lines = []
for speaker in speakers:
    for f in speakers[speaker]:
        with open(f, "r", encoding="utf8") as tf:
            all_lines.append(tf.read())
with open("temp.txt","w",encoding="utf8") as f:
    for line in all_lines:
        f.write(line)

if not os.path.exists(output_lookup_folder):
    os.makedirs(output_lookup_folder)

# TRAIN SENTENCEPIECE MODELS & CREATE LOOKUPS

spm.SentencePieceTrainer.Train('--input=temp.txt --model_prefix=' + os.path.join(output_lookup_folder, "tok")+ ' --character_coverage=1.0 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --max_sentence_length=8000 --vocab_size=' + str(
    vocab_size))
print("Done.")
lookup = Lookup(type="bpe")
lookup.save_special_tokens(file_prefix=os.path.join(output_lookup_folder, "tok"))

# check everything is ok
lookup = Lookup(type="bpe")
lookup.load(file_prefix=os.path.join(output_lookup_folder,"tok"))
text = "This is a simple test."

token_ids = lookup.encode(text)
print("Encode: {}".format(token_ids))
recreated_string = lookup.decode(token_ids)
print("Decode: [{}]".format(recreated_string))
print("Map w2i:")
tokens = lookup.tokenize(text)
for i in range(len(tokens)):
    print("\t[{}] = [{}]".format(tokens[i], lookup.convert_tokens_to_ids(tokens[i])))

print("Map i2w:")
for i in range(len(token_ids)):
    print("\t[{}] = [{}]".format(token_ids[i], lookup.convert_ids_to_tokens(token_ids[i])))

token_ids = lookup.encode(text, add_bos_eos_tokens=True)
print("Encode with bos/eos: [{}]".format(token_ids))
recreated_string = lookup.decode(token_ids)
print("Decode with bos/eos: [{}]".format(recreated_string))
recreated_string = lookup.decode(token_ids, skip_bos_eos_tokens=True)
print("Decode w/o  bos/eos: [{}]".format(recreated_string))



# create training data
print("Creating training data...")
X = []
Y = []
X_test = []
Y_test = []
for speaker in speakers:
    print("Speaker {} has {} files.".format(speaker, len(speakers[speaker])))
    xs = []
    ys = []
    xs_test = []
    ys_test = []

    train_section = int(len(speakers[speaker]) * 0.9)
    for f in speakers[speaker][:train_section]:
        with open(f, "r", encoding="utf8") as tf:
            xs.append(tf.read().strip())
        ys.append(np.load(f.replace(".txt", ".gst.npy"))[0])

    for f in speakers[speaker][train_section:]:
        with open(f, "r", encoding="utf8") as tf:
            xs_test.append(tf.read().strip())
        ys_test.append(np.load(f.replace(".txt", ".gst.npy"))[0].tolist())
    # break

    # sliding window for train
    for i in range(len(xs)):  # sliding window!
        if i < bw_context_len:
            X.append([''] * (bw_context_len - i) + xs[0:fw_context_len + 1 + i])
        elif i >= len(xs) - fw_context_len:
            # print("@{}/{}".format(i, len(xs)))
            remaining = len(xs) - i - 1
            X.append(xs[i - bw_context_len:i + remaining + 1] + [''] * (fw_context_len - remaining))
            # print(X[-1])
            # print(len(X[-1]))
            # input("?? {}/len{}".format(i, len(xs)))
        else:
            X.append(xs[i - bw_context_len:i + fw_context_len + 1])
        Y.append(ys[i])

    # now for test, same as train
    for i in range(len(xs_test)):  # sliding window!
        if i < bw_context_len:
            X_test.append([''] * (bw_context_len - i) + xs_test[0:fw_context_len + 1 + i])
        elif i >= len(xs_test) - fw_context_len:
            # print("@{}/{}".format(i, len(xs)))
            remaining = len(xs_test) - i - 1
            X_test.append(xs_test[i - bw_context_len:i + remaining + 1] + [''] * (fw_context_len - remaining))
        else:
            X_test.append(xs_test[i - bw_context_len:i + fw_context_len + 1])
        Y_test.append(ys_test[i])

X_temp = []
for instance in X:
    ins = []
    for sentence in instance:
        ins.append(lookup.encode(sentence, add_bos_eos_tokens=True))
    X_temp.append(ins)
X = X_temp
X_temp = []
for instance in X_test:
    ins = []
    for sentence in instance:
        ins.append(lookup.encode(sentence, add_bos_eos_tokens=True))
    X_temp.append(ins)
X_test = X_temp
with open("data/train.x", "w", encoding="utf8") as f:
    json.dump(X, f, indent=2)
with open("data/train.y", "wb") as f:
    pickle.dump(Y, f)
with open("data/test.x", "w", encoding="utf8") as f:
    json.dump(X_test, f, indent=2)
with open("data/test.y", "wb") as f:
    pickle.dump(Y_test, f)



