import os, shutil, argparse
import time
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import yaml
import utils.cls_tools.parameters as parameters
import torch.cuda.amp as amp

from monaural.dataset_loader import Dataset_loader

from models.resnet_conformer_audio import ResnetConformer_seddoa_nopool_2026

from lr_scheduler.tri_stage_lr_scheduler import TriStageLRScheduler

from utils.cls_tools.cls_compute_seld_results import ComputeSELDResults
# from utils.cls_tools.cls_compute_sed_results import ComputeSEDResults
from utils.write_csv import write_output_format_file
from monaural.preprocess import ResnetConformer_2026
from utils.sed_doa import process_foa_input_sed_doa_labels, process_mic_input_sed_doa_labels, SedDoaLoss, SedDoaResult_2023


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None

def main(args):
    params = parameters.get_params()
    dataset_dir = params['dataset_dir']
    dataset_combination = '{}_{}'.format(params['dataset'], 'dev')
    audio_dir = os.path.join(dataset_dir, dataset_combination)
    
    #log
    log_output_folder = os.path.dirname(args['result']['log_output_path'])
    os.makedirs(log_output_folder, exist_ok=True)
    logging.basicConfig(filename=args['result']['log_output_path'], filemode='w', level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.info(args)

    if args['model']['in_channel']==10:
        data_process_fn = process_mic_input_sed_doa_labels
    else:
        data_process_fn = process_foa_input_sed_doa_labels
    result_class = SedDoaResult_2023
    criterion = SedDoaLoss(loss_weight=[0.1,1])

    segment_len = args['data']['segment_len']
    cal_sig_len = segment_len * 2400
    # model = ResnetConformer_seddoa_nopool_2026(in_channel=args['model']['in_channel'], in_dim=args['model']['in_dim'], out_dim=args['model']['out_dim'])
    model = ResnetConformer_2026(in_channel=args['model']['in_channel'], in_dim=args['model']['in_dim'], out_dim=args['model']['out_dim'],sig_len=cal_sig_len, params=params)


    train_split = [1,2,3,5,6]
    train_dataset = Dataset_loader(
        keys_file=args['data']['keys_file'], 
        audio_dir=audio_dir,
        label_dir=args['data']['train_label_dir'],
        segment_len_sec=segment_len // 10, # config가 100프레임이면 10초
        fs=24000,
        split=train_split
        )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=args['data']['batch_size'], shuffle=True, 
        num_workers=args['train']['train_num_workers'], collate_fn=train_dataset.collater
    )

    test_split = [4]
    test_dataset = Dataset_loader(
        keys_file=args['data']['keys_file'], 
        audio_dir=audio_dir,
        label_dir=args['data']['test_label_dir'],
        segment_len_sec=segment_len // 10, # config가 100프레임이면 10초
        fs=24000,
        split=test_split
        )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=args['data']['batch_size'], shuffle=False, 
        num_workers=args['train']['test_num_workers'], collate_fn=test_dataset.collater
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else "cpu")
    #pdb.set_trace()
    model = model.to(device)
    logger.info(model)
    set_random_seed(12332)

    if args['model']['pre-train']:
        model.load_state_dict(torch.load(args['model']['pre-train_model']))
    logger.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args['train']['lr'])
    total_steps = args['train']['nb_steps']
    warmup_steps = int(total_steps*0.1)
    hold_steps = int(total_steps*0.6)
    decay_steps = int(total_steps*0.3)
    scheduler = TriStageLRScheduler(optimizer, peak_lr=args['train']['lr'], init_lr_scale=0.01, final_lr_scale=0.05, 
                                    warmup_steps=warmup_steps, hold_steps=hold_steps, decay_steps=decay_steps)

    best_seld_scr = float('inf')
    epoch_count = 0
    step_count = 0

    stop_training = False
    # scaler = torch.amp.GradScaler('cuda',enabled=(device.type == "cuda"))
    use_amp = False                                                                     #True
    while not stop_training:
        train_loss = []
        test_loss = []
        epoch_count += 1

        start_time = time.time()
        model.train()
        for data in train_dataloader:
            input = data['input'].to(device)                        # ([32, 10, 500, 64]) -> torch.Size([32, 4, 240000])
            target = data['target'].to(device)                      # ([32, 100, 52])

            #pdb.set_trace()
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda',enabled=(device.type == "cuda")):
                    output = model(input)
                    # pdb.set_trace()
                    loss = criterion(output, target)
            else:
                output = model(input)
                loss = criterion(output, target)
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # scheduler.step()

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            step_count += 1
            if step_count % args['result']['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('epoch: {}, step: {}/{}, lr:{:.6f}, train_loss:{:.4f}'.format(epoch_count, step_count, total_steps, lr, loss.item()))
            if step_count == total_steps:
                stop_training = True
                break
            
        torch.cuda.empty_cache()
        train_time = time.time() - start_time

        start_time = time.time()
        model.eval()
        test_result = result_class(segment_length=segment_len)
        for data in test_dataloader:
            input = data['input'].to(device)
            target = data['target'].to(device)
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target)
                test_loss.append(loss.item())

            test_result.add_items(data['wav_names'], output)
        output_dict = test_result.get_result()
        test_time = time.time() - start_time
        
        dcase_output_val_dir = os.path.join(args['result']['dcase_output_dir'], 'epoch{}_step{}'.format(epoch_count, step_count))
        # if os.path.exists(dcase_output_val_dir):
        #     shutil.rmtree(dcase_output_val_dir)
        os.makedirs(dcase_output_val_dir, exist_ok=True)
        for csv_name, perfile_out_dict in output_dict.items():
            output_file = os.path.join(dcase_output_val_dir, '{}.csv'.format(csv_name))
            write_output_format_file(output_file, perfile_out_dict)
        
        score_obj = ComputeSELDResults(ref_files_folder=args['data']['ref_files_dir'])
        val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_dir)
        logger.info('epoch: {}, step: {}/{}, train_time:{:.2f}, test_time:{:.2f}, average_train_loss:{:.4f}, average_test_loss:{:.4f}'.format(epoch_count, step_count, total_steps, train_time, test_time, np.mean(train_loss), np.mean(test_loss)))
        logger.info('ER/F/LE/LR/SELD: {}'.format('{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr)))

        checkpoint_output_dir = args['result']['checkpoint_output_dir']
        os.makedirs(checkpoint_output_dir, exist_ok=True)

        model_output_dir = args['result']['model_output_dir']
        os.makedirs(model_output_dir, exist_ok=True)
        
        if val_seld_scr < best_seld_scr:
            best_seld_scr = val_seld_scr
            best_model_path = os.path.join(model_output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info('Found new best model with SELD score: {:.4f}. Saved to {}'.format(best_seld_scr, best_model_path))

        model_path = os.path.join(checkpoint_output_dir, 'checkpoint_epoch{}_step{}.pth'.format(epoch_count, step_count))
        torch.save(model.state_dict(), model_path)
        logger.info('save checkpoint: {}'.format(model_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('train')
    parser.add_argument('-c', '--config_name', type=str, default='foa_dev_multi_accdoa_nopool', help='name of config')
    input_args = parser.parse_args()

    # foa_dev_seddoa_nopool
    # foa_dev_accdoa_nopool
    # foa_dev_multi_accdoa_nopool
    with open(os.path.join('config', '{}.yaml'.format(input_args.config_name)), 'r') as f:
        args = yaml.safe_load(f)
    main(args)