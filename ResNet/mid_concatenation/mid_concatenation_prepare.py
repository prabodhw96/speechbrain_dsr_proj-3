from speechbrain_dsr.recipes.TIMIT.timit_prepare import prepare_timit
import pandas as pd
import os

data_folder = "TIMIT"
splits = ["train", "dev", "test"]
save_folder = "TIMIT_prepared"
prepare_timit(data_folder, splits, save_folder)
train_df = pd.read_csv("TIMIT_prepared/train.csv")
train_df["wav"] = train_df["wav"].apply(lambda x: x.replace("TIMIT", "/content/TIMIT_4_channels/TIMIT_LA2"))
train_df.to_csv("TIMIT_prepared/train.csv", index=False)
dev_df = pd.read_csv("TIMIT_prepared/dev.csv")
dev_df["wav"] = dev_df["wav"].apply(lambda x: x.replace("TIMIT", "/content/TIMIT_4_channels/TIMIT_LA2"))
dev_df.to_csv("TIMIT_prepared/dev.csv", index=False)
test_df = pd.read_csv("TIMIT_prepared/test.csv")
test_df["wav"] = test_df["wav"].apply(lambda x: x.replace("TIMIT", "/content/TIMIT_4_channels/TIMIT_LA2"))
test_df.to_csv("TIMIT_prepared/test.csv", index=False)
!cp "/content/TIMIT_prepared/train.csv" -r "/content/TIMIT_4_channels/TIMIT_LA2"
!cp "/content/TIMIT_prepared/dev.csv" -r "/content/TIMIT_4_channels/TIMIT_LA2"
!cp "/content/TIMIT_prepared/test.csv" -r "/content/TIMIT_4_channels/TIMIT_LA2"
for root, file, dirs in os.walk("/content/TIMIT_4_channels/TIMIT_LA2"):
    for dir in dirs:
        filename = os.path.join(root,dir)
        if filename.endswith("_LA2.wav"):
            new_filename = filename.replace("_LA2.wav", ".wav")
            os.rename(filename, new_filename)
for root, file, dirs in os.walk("/content/TIMIT_4_channels/TIMIT_LA3"):
    for dir in dirs:
        filename = os.path.join(root,dir)
        if filename.endswith("_LA3.wav"):
            new_filename = filename.replace("_LA3.wav", ".wav")
            os.rename(filename, new_filename)
for root, file, dirs in os.walk("/content/TIMIT_4_channels/TIMIT_LA4"):
    for dir in dirs:
        filename = os.path.join(root,dir)
        if filename.endswith("_LA4.wav"):
            new_filename = filename.replace("_LA4.wav", ".wav")
            os.rename(filename, new_filename)
for root, file, dirs in os.walk("/content/TIMIT_4_channels/TIMIT_LA5"):
    for dir in dirs:
        filename = os.path.join(root,dir)
        if filename.endswith("_LA5.wav"):
            new_filename = filename.replace("_LA5.wav", ".wav")
            os.rename(filename, new_filename)