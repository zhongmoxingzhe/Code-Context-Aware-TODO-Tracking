# encoding=utf-8
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
#from BERT_model import TransformerModel1, Config
from BERT_model import codeModel3, Config
from codebert_train_sbp_base import Data_processor
from loss_func import Focal_loss, DiceLoss


os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def write_file(filename, data):
    with open(filename, 'w') as f:
        for i in data:
            f.write(str(i).strip() + '\n')


def test_model(dir_path, epoch_id, keyword, language):
    config = Config()
    test_dt = Data_processor('test', config.batch_size, dir_path, keyword)
    testdata_loader = test_dt.processed_dataloader
    print('Dataloader created!')

    # Adjust file paths based on the selected language
    model_state_path = f'../autodl-tmp/codebert_{keyword}_new{language}/epoch_{epoch_id}/model.ckpt'
    print(model_state_path)

    # Load model
    model =  codeModel3(config).to(config.device)
    model.load_state_dict(torch.load(model_state_path, map_location=config.device))
    model.eval()
    print("Model loaded")

    print("Running Testing...")
    t0 = time.time()
    model.eval()

    print("-----Using focal loss")
    loss_fn = Focal_loss(alpha=0.25, gamma=2, num_classes=2)

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    prob_result_lst = []
    all_pre_label = []
    
    total_steps = len(testdata_loader)
    progress_bar = tqdm(range(total_steps))

    for batch in testdata_loader:
        with torch.no_grad():
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
            for index in msg_edge_index:
                print(index.shape)
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

    prob_result = np.vstack(prob_result_lst)
    print("Probability results:", type(prob_result), prob_result.shape)

    output_path = model_state_path.replace('model.ckpt', f'{keyword}_{language}_test_prob.csv')
    np.savetxt(output_path, prob_result, delimiter=",")
    avg_val_accuracy = total_eval_accuracy / len(testdata_loader)
    print(f"  Accuracy: {avg_val_accuracy:.3f}")
    
    cal_f1(all_pre_label,dir_path,keyword)
    cal_precision(all_pre_label, dir_path, keyword)
    cal_recall(all_pre_label, dir_path, keyword)
    cal_auc(prob_result, dir_path, keyword)
    cal_ce(prob_result, dir_path, keyword)
    write_file(f'./output_dir/tdreminder_{keyword}_{language}_{epoch_id}', all_pre_label)

    avg_val_loss = total_eval_loss / len(testdata_loader)
    test_time = format_time(time.time() - t0)

    print(f"  Validation Loss: {avg_val_loss:.3f}")
    print(f"  Validation took: {test_time}")


if __name__ == '__main__':
    # Choose language: 'python' or 'java'
    language = 'java'  # Change to 'java' for Java testing

    DATA_DIRS = {
        'python': "./top_repos_10000/sbp_python/",
        'java': "./top_repos_10000/sbp_java/"
    }

    dir_path = DATA_DIRS[language]
    print(f"Selected language: {language}")
    print(f"Directory path: {dir_path}")

    for epoch_id in range(3, 4):
        test_model(dir_path, epoch_id, 'sbp', language)
