# encoding=utf-8
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from BERT_model import codeModel3, Config, load_codeBERT
from loss_func import Focal_loss, DiceLoss
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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
    # todo change the model dir
    base_dir = '../autodl-tmp/codebert_todo_newpython_old_maxlen512_multi/epoch_' + str(epoch) + '/'
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
    print('Dataloader maked!')
    model = codeModel3(config).to(config.device)
    print('codeBERT model created!')

    # Optimizer & Learning Rate Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the training data.
    epochs = config.num_epochs
    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(traindata_loader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    # We are ready to kick off the training
    # Set the seed value all over the place to make this reproducible.
    seed_val = 3407
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()
    model.eval()
    # todo change loss function
    print("-----Using focal loss")
    # loss_fn = F.cross_entropy
    loss_fn = Focal_loss(alpha=0.25, gamma=2, num_classes=2)
    # with_logits=True, ohem_ratio=0.01
    # loss_fn = DiceLoss(with_logits=True, ohem_ratio=0.1)
    print("Begin training...")
    progress_bar = tqdm(range(total_steps))
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        epcho_train_loss = 0
        model.train()

        for step, batch in enumerate(traindata_loader):
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(traindata_loader), elapsed))

            diff_input_ids = batch[0].to(config.device)
            diff_input_mask = batch[1].to(config.device)

            msg_input_ids = batch[2].to(config.device)
            msg_input_mask = batch[3].to(config.device)

            batch_labes = batch[4].to(config.device)

            model.zero_grad()
            diff_input = (diff_input_ids, diff_input_mask)
            msg_input = (msg_input_ids, msg_input_mask)
            batch_outputs = model(diff_input, msg_input)
            loss = loss_fn(batch_outputs, batch_labes)
            epcho_train_loss += loss.item()
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Calculate the average loss over all of the batches.
        avg_train_loss = epcho_train_loss / len(traindata_loader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("====== Average training loss: {0:.2f}".format(avg_train_loss))
        print("====== Training epcoh took: {:}".format(training_time))

        print("Running Testing....")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        for batch in testdata_loader:
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # values prior to applying an activation function like the softmax.
                diff_input_ids = batch[0].to(config.device)
                diff_input_mask = batch[1].to(config.device)

                msg_input_ids = batch[2].to(config.device)
                msg_input_mask = batch[3].to(config.device)

                batch_labes = batch[4].to(config.device)
                diff_input = (diff_input_ids, diff_input_mask)
                msg_input = (msg_input_ids, msg_input_mask)

                b_outputs = model(diff_input, msg_input)

            loss = loss_fn(b_outputs, batch_labes)
            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # move labels to CPU
            preds = torch.max(b_outputs.data, 1)[1].cpu().numpy()
            # print("preds:", type(preds), preds.shape)
            labels = batch_labes.to('cpu').numpy()
            # Calculate the accuracy for this batch of test sentences, and
            total_eval_accuracy += flat_accuracy(preds, labels)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(testdata_loader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(testdata_loader)

        # Measure how long the validation run took.
        test_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(test_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {'epoch': epoch_i + 1,
             'Training Loss': avg_train_loss,
             'Valid. Loss': avg_val_loss,
             'Valid. Accur.': avg_val_accuracy,
             'Training Time': training_time,
             'Validation Time': test_time
             })

        save_model(epoch_i + 1, model, training_stats)

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


if __name__ == '__main__':
    # todo change the dir path and the path of model
    # new dataset (filter out fixme xxx)
    JAVA_TGT_DIR = "./top_repos_10000/new_java/"
    PYTHON_TGT_DIR = "./top_repos_10000/new_python/"
    dir_path = JAVA_TGT_DIR
    train_model(dir_path, 'todo')
    print(dir_path)
