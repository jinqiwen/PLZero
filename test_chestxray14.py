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

    args.num_classes = 14
    args.cuda_num = 1
    args.pretrained = True

    trainer = ChexnetTrainer(args)
    print('Testing the trained model')


    test_ind_auroc, F1_mean, Acc_mean, Mcc_mean, mAP = trainer.test()

    test_ind_auroc = np.array(test_ind_auroc)

    trainer.print_auroc_test(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids],trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    #
    trainer.print_auroc_test(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids,prefix='\ntest_unseen')

    trainer.print_base_indicator(Mcc_mean, 'Mcc', prefix='test')
    trainer.print_base_indicator(F1_mean, 'F1_score', prefix='test')
    trainer.print_base_indicator(Acc_mean, 'Acc', prefix='test')
    trainer.print_base_indicator(mAP, 'mAP',  prefix='test')
if __name__ == '__main__':
    main()





