import os
import sys
import torch
import random
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
import pickle
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from utils import *
from BERT_model import codeModel, Config, load_codeBERT
from loss_func import Focal_loss, DiceLoss


class Data_processor(object):

    def __init__(self, set_type, batch_size, dir_path, keyword):
        self.set_type = set_type
        self.batch_size = batch_size
        self.dir_path = dir_path
        self.keyword = keyword
        print("Loading codeBERT model...")
        self.bert_model, self.tokenizer = load_codeBERT()
        print('codeBERT loaded')

        print('Loading %s dataset...' % (set_type + keyword))
        diff_type, msg_type, label_type = load_data(self.dir_path, set_type, keyword)

        print("Begin encoding...")
        self.encoded_diff = self.bert_encode_pair(diff_type, msg_type)
        self.encoded_msg = self.bert_encode_pair(msg_type, diff_type)
        self.labels = np.asarray(label_type)
        print("%s dataset loaded.")

        print("Making data loader...")
        self.train_diff = self.make_data(self.encoded_diff)
        self.train_msg = self.make_data(self.encoded_msg)
        self.processed_dataloader = self.make_loader()
        print("Finished making data loader.")

    def bert_encode(self, input_lst):
        # bert encoding
        encoded_input = self.tokenizer(input_lst, padding=True, truncation=True, max_length=512, return_tensors='pt')
        return encoded_input

    def bert_encode_pair(self, input_lst1, input_lst2):
        encoded_input_pair = self.tokenizer(input_lst1, input_lst2,
                                            padding=True, truncation=True,
                                            max_length=512, return_tensors='pt')
        return encoded_input_pair

    def make_data(self, encoded_data):
        input_ids, attention_masks = encoded_data['input_ids'], encoded_data['attention_mask']
        # Convert to Pytorch Data Types
        inputs = torch.tensor(input_ids)
        masks = torch.tensor(attention_masks)
        labels = torch.tensor(self.labels)
        train_data = (inputs, masks, labels)
        return train_data

    def make_loader(self):
        tensor_data = TensorDataset(self.train_diff[0], self.train_diff[1], self.train_msg[0],
                                    self.train_msg[1], self.train_diff[2])
        dataloader = DataLoader(tensor_data, batch_size=self.batch_size)
        return dataloader

    def save_loader(self):
        with open(self.dir_path + self.set_type + '_dataloader.pkl', 'wb') as hander:
            pickle.dump(self.processed_dataloader, hander)
        pass


def save_model(epoch, model, training_stats):
    base_dir = './codebert_sbp_python/epoch_' + str(epoch) + '/'
    out_dir = base_dir + 'model.ckpt'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print('Saving model to %s' % out_dir)
    torch.save(model.state_dict(), out_dir)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats.to_json(base_dir + "training_stats.json")


def train_model(dir_path, keyword):
    config = Config()
    train_dt = Data_processor('train', config.batch_size, dir_path, keyword)
    traindata_loader = train_dt.processed_dataloader
    test_dt = Data_processor('val', config.batch_size, dir_path, keyword)
    testdata_loader = test_dt.processed_dataloader
    print('Dataloader created!')

    # Move model to GPU
    model = codeModel(config)
    model = model.cuda()  # Move model to single GPU
    print('codeBERT model created!')

    # Optimizer & Learning Rate Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = config.num_epochs
    total_steps = len(traindata_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    # Set random seeds for reproducibility
    seed_val = 3407
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # Training stats list
    training_stats = []

    # Measure the total training time
    total_t0 = time.time()
    model.eval()
    loss_fn = Focal_loss(alpha=0.25, gamma=2, num_classes=2)
    print("Begin training...")
    progress_bar = tqdm(range(total_steps))
    for epoch_i in range(epochs):
        print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
        print('Training...')
        t0 = time.time()
        epoch_train_loss = 0
        model.train()

        for step, batch in enumerate(traindata_loader):
            if step % 100 == 0 and step != 0:
                elapsed = format_time(time.time() - t0)
                print(f'  Batch {step} of {len(traindata_loader)}. Elapsed: {elapsed}.')

            diff_input_ids = batch[0].to(config.device)
            diff_input_mask = batch[1].to(config.device)

            msg_input_ids = batch[2].to(config.device)
            msg_input_mask = batch[3].to(config.device)

            batch_labels = batch[4].to(config.device)

            model.zero_grad()
            diff_input = (diff_input_ids, diff_input_mask)
            msg_input = (msg_input_ids, msg_input_mask)
            batch_outputs = model(diff_input, msg_input)
            print("batch_outputs:", batch_outputs.shape)
            print("batch_labels:", batch_labels.shape)
            print("batch_labels_len:",batch_labels.numel())
            print("batch_labels_content:",batch_labels)
            loss = loss_fn(batch_outputs, batch_labels)
            epoch_train_loss += loss.item()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        avg_train_loss = epoch_train_loss / len(traindata_loader)
        training_time = format_time(time.time() - t0)
        print(f"\n====== Average training loss: {avg_train_loss:.2f}")
        print(f"====== Training epoch took: {training_time}")

        print("Running Testing....")
        t0 = time.time()
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        for batch in testdata_loader:
            with torch.no_grad():
                diff_input_ids = batch[0].to(config.device)
                diff_input_mask = batch[1].to(config.device)

                msg_input_ids = batch[2].to(config.device)
                msg_input_mask = batch[3].to(config.device)

                batch_labels = batch[4].to(config.device)
                diff_input = (diff_input_ids, diff_input_mask)
                msg_input = (msg_input_ids, msg_input_mask)

                b_outputs = model(diff_input, msg_input)
                print("b_outputs:" , b_outputs.shape)
                print("b_labels:" , batch_labels.shape)

            loss = loss_fn(b_outputs, batch_labels)
            total_eval_loss += loss.item()

            preds = torch.max(b_outputs.data, 1)[1].cpu().numpy()
            labels = batch_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(preds, labels)

        avg_val_accuracy = total_eval_accuracy / len(testdata_loader)
        print(f"  Accuracy: {avg_val_accuracy:.2f}")

        avg_val_loss = total_eval_loss / len(testdata_loader)
        test_time = format_time(time.time() - t0)

        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {test_time}")

        # Record statistics
        training_stats.append({
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': test_time
        })

        save_model(epoch_i + 1, model, training_stats)

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)} (h:mm:ss)")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # Directly use the GPU without the distributed setup
    dir_path = "./top_repos_10000/sbp_python/"
    train_model(dir_path, 'sbp')
    print(dir_path)