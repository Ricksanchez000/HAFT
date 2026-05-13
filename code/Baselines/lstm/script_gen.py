spare_cuda = [0,5,6,7]
if __name__ == '__main__':
    for idnex, i in enumerate(['airfoil', 'amazon_employee',
                                                                   'ap_omentum_ovary', 'german_credit',
                                                                   'higgs', 'housing_boston', 'ionosphere',
                                                                   'lymphography', 'messidor_features', 'openml_620',
                                                                   'pima_indian', 'spam_base', 'spectf', 'svmguide3',
                                                                   'uci_credit_card', 'wine_red', 'wine_white',
                                                                   'openml_586',
                                                                   'openml_589', 'openml_607', 'openml_616',
                                                                   'openml_618',
                                                                   'openml_637']):
        print(f'~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name {i} --gpu {spare_cuda[idnex % 4]}')



"""
17:
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name airfoil --gpu 0
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name amazon_employee --gpu 5
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name ap_omentum_ovary --gpu 6
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name german_credit --gpu 7
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name messidor_features --gpu 0
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name openml_620 --gpu 5
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name uci_credit_card --gpu 6
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name wine_red --gpu 7
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name openml_589 --gpu 6
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name openml_607 --gpu 7
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name openml_616 --gpu 0
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name openml_618 --gpu 5
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name openml_637 --gpu 6

212
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name higgs --gpu 0
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name housing_boston --gpu 1
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name ionosphere --gpu 2
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name lymphography --gpu 3
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name pima_indian --gpu 6
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name spam_base --gpu 7
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name spectf --gpu 0
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name svmguide3 --gpu 5
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name wine_white --gpu 0
~/miniconda3/envs/shaow/bin/python -u ./lstm/train_controller.py --task_name openml_586 --gpu 5


17


212







"""