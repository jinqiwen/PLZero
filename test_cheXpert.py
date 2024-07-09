import os
import numpy as np
import time
import sys

from ChexnetTrainer import ChexnetTrainer
from arguments import parse_args


def main():
    args = parse_args()

    try:
        os.mkdir(args.save_dir)
    except OSError as error:
        print(error)
    args.num_classes = 12
    args.pretrained = True

    trainer = ChexnetTrainer(args)
    print('Testing the trained model')

    test_ind_auroc, F1_mean, Acc_mean, Mcc_mean, mAP = trainer.test_cheXpert_12_get_5()

    test_ind_auroc = np.array(test_ind_auroc)

    trainer.print_auroc_cheXpert_12_5_for_test(test_ind_auroc, trainer.test_dl_cp.dataset.class_ids_loaded_test, prefix='test')
    trainer.print_base_indicator(Mcc_mean, 'Mcc', prefix='test')
    trainer.print_base_indicator(F1_mean, 'F1_score', prefix='test')
    trainer.print_base_indicator(Acc_mean, 'Acc', prefix='test')
    trainer.print_base_indicator(mAP, 'mAP', prefix='test')
if __name__ == '__main__':
    main()





