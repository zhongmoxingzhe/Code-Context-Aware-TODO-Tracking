# encoding=utf-8
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from BERT_model import TODO_Checker, Config
from codebert_train import Data_processor
from loss_func import Focal_loss, DiceLoss


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def write_file(filename, data):
    with open(filename, 'w') as f:
        for i in data:
            f.write(str(i).strip() + '\n')


def test_model(dir_path, epoch_id, keyword):
    config = Config()
    test_dt = Data_processor('test', config.batch_size, dir_path, keyword)
    testdata_loader = test_dt.processed_dataloader
    print('Dataloader maked!')
    # TODo modify the stat path
    keyword = 'todo'
    model_state_path = '../autodl-tmp/codebert_'+ keyword +'_newjava_maxlen512_4/epoch_' + str(epoch_id) + '/model.ckpt'
    print(model_state_path)
    # load model
    model = codeModel4(config).to(config.device)
    model.load_state_dict(torch.load(model_state_path))
    model.eval()
    print("model loaded")

    print("Running Testing...")
    t0 = time.time()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
    print("-----Using focal loss")
    # loss_fn = F.cross_entropy
    loss_fn = Focal_loss(alpha=0.25, gamma=2, num_classes=2)
    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    prob_result_lst = []
    all_pre_label = []
    
    total_steps = len(testdata_loader)
    progress_bar = tqdm(range(total_steps))
    # Evaluate data for one epoch
    for batch in testdata_loader:
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # values prior to applying an activation function like the softmax.
            diff_input_ids = batch['diff_ids'].to(config.device)
            diff_input_mask = batch['diff_mask'].to(config.device)
            msg_input_ids = batch['msg_ids'].to(config.device)
            msg_input_mask = batch['msg_mask'].to(config.device)
            batch_labes = batch['label'].to(config.device)
            diff_edge_index = batch['diff_edge_index']  # list of edge_index tensors
            msg_edge_index = batch['msg_edge_index']

            model.zero_grad()
            diff_input = (diff_input_ids, diff_input_mask)
            msg_input = (msg_input_ids, msg_input_mask)
            b_outputs = model(diff_input, msg_input, diff_edge_index, msg_edge_index)

        loss = loss_fn(b_outputs, batch_labes)
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # torch tensor
        lr = torch.nn.Softmax(dim=1)
        preds_prob = lr(b_outputs.data)
        # print("lr prob:", preds_prob, type(preds_prob), preds_prob.shape)
        # tensor to numpy
        preds_prob = preds_prob.cpu().detach().numpy()
        # print("lr prob:", preds_prob, type(preds_prob), preds_prob.shape)
        prob_result_lst.append(preds_prob)

        # move labels to CPU
        preds = torch.max(b_outputs.data, 1)[1].cpu().numpy()
        # print("preds:", type(preds), preds.shape)
        labels = batch_labes.to('cpu').numpy()
        all_pre_label.extend(preds.flatten())
        # Calculate the accuracy for this batch of test sentences, and
        total_eval_accuracy += flat_accuracy(preds, labels)
        progress_bar.update(1)

    #
    prob_result = np.vstack(prob_result_lst)
    print("prob_result:", type(prob_result), prob_result.shape)

    # output to csv
    output_path = model_state_path[:-18] + 'codebert_newjava_todo_prob.csv'
    np.savetxt(output_path, prob_result, delimiter=",")
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(testdata_loader)
    print("  Accuracy: {0:.3f}".format(avg_val_accuracy))

    # all_pre_labels = np.vstack(all_pre_label)
    cal_f1(all_pre_label,dir_path)
    cal_precision(all_pre_label, dir_path)
    cal_recall(all_pre_label, dir_path)
    cal_auc(prob_result, dir_path)
    cal_ce(prob_result, dir_path)
    # todo
    write_file('./output_dir/tdcheck_' + keyword +'_newjava_maxlen512_' + str(epoch_id), all_pre_label)

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(testdata_loader)

    # Measure how long the validation run took.
    test_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.3f}".format(avg_val_loss))
    print("  Validation took: {:}".format(test_time))


if __name__ == '__main__':
    # todo change the dir path and the path of model
    # new dataset (filter out fixme xxx)
    JAVA_TGT_DIR = "./top_repos_10000/new_java/"
    PYTHON_TGT_DIR = "./top_repos_10000/new_python/"
    dir_path = JAVA_TGT_DIR
    print(dir_path)
    for epoch_id in range(4,7):
        test_model(dir_path, epoch_id, 'todo')


