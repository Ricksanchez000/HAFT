running examples, RL might be sensitive to reward and feature sequence:

python new2_main_final.py --dataset=higgs --id=123 --critic_lr=0.001 --num_episodes=50 --max_steps=10 --op_coef=0.1 --feature_lr=0.0001 --n_splits=5

python new_main_final.py --dataset=amazon_employee --id=123 --critic_lr=1e-5 --num_episodes=40 --max_steps=12 --op_coef=0.2 --feature_lr=0.0001 --n_splits=5

python new2_main_final.py --dataset=pima_indian --id=123 --critic_lr=1e-5 --num_episodes=30 --max_steps=12 --ent_coef=0.1 --feature_lr=1e-5 --n_splits=5

python new2_main_final.py --dataset=spectf --id=123 --critic_lr=1e-04 --feature_lr=1e-05 --num_episodes=100 --max_steps=11 --ent_coef=0.2 --n_splits=5


python new2_main_final.py --dataset=svmguide3 --id=123 --critic_lr=1e-03 --feature_lr=1e-03 --num_episodes=100 --max_steps=15 --op_coef=0.1 --n_splits=5


python new2_main_final.py --dataset=german_credit --id=123 --critic_lr=1e-05 --feature_lr=1e-05 --num_episodes=100 --max_steps=5 --ent_coef=0.1 --n_splits=5


python new2_main_final.py --dataset=uci_credit_card --id=123 --critic_lr=1e-03 --feature_lr=1e-04 --num_episodes=30 --max_steps=15 --ent_coef=0.1 --n_splits=5

python new2_main_final.py --dataset=messidor_features --id=123 --critic_lr=1e-03 --feature_lr=1e-04 --num_episodes=50 --max_steps 25 --ent_coef=0.01 --n_splits=5


python new2_main_final.py --dataset=wine_red --id=123 --critic_lr=1e-04 --feature_lr=1e-05 --num_episodes=50 --max_steps=15 --ent_coef=0.2 --n_splits=5


python new2_main_final.py --dataset=wine_white --id=123 --critic_lr=1e-03 --feature_lr=1e-04 --num_episodes=30 --max_steps=5 --ent_coef=0.5 --n_splits=5


python new2_main_final.py --dataset=spam_base --id=123 --critic_lr=1e-05 --feature_lr=1e-05 --num_episodes=40 --max_steps=5 --ent_coef=0.5 --n_splits=5


python new2_main_final.py --dataset=ap_omentum_ovary --id=123 --critic_lr=1e-05 --feature_lr=1e-04 --num_episodes=20 --max_steps=4 --op_coef=0.3 --ent_coef=0.5 --n_splits=5 --ap_k=100 --top_k=23


python new_main_final.py --dataset=lymphography --id=123 --critic_lr=1e-03 --feature_lr=1e-04 --num_episodes=40 --max_steps=11 --ent_coef=0.01 --n_splits=2


python new2_main_final.py --dataset=ionosphere --id=123 --critic_lr=1e-05 --feature_lr=1e-05 --num_episodes=30 --max_steps=6 --op_coef=0.01 --ent_coef=0.1 --n_splits=5 --top_k=5


python new2_main_final.py --dataset=housing_boston --id=123 --critic_lr=0.001 --feature_lr=0.001 --num_episodes=100 --max_steps=5 --ent_coef=0.1 --n_splits=2

python new2_main_final.py --dataset=airfoil --id=test_001 --critic_lr=0.1 --feature_lr=0.1 --num_episodes=40 --max_steps=16 --op_coef=0.1 --ent_coef=0.1 --n_splits=5 --top_k=19


python new2_main_final.py --dataset=openml_618 --id=123 --critic_lr=1e-03 --feature_lr=1e-03 --num_episodes=50 --max_steps=2 --op_coef=0.1 --ent_coef=0.1 --n_splits=5


python new2_main_final.py --dataset=openml_589 --id=123 --critic_lr=1e-04 --feature_lr=1e-04 --num_episodes=100 --max_steps=15 --op_coef=0.1 --ent_coef=0.3 --n_splits=5

python new2_main_final.py --dataset=openml_616 --id=123 --critic_lr=1e-05 --feature_lr=1e-04 --num_episodes=9 --max_steps=5 --op_coef=0.02 --ent_coef=0.01 --n_splits=5 --top_k=20


python new2_main_final.py --dataset=openml_607 --id=123 --critic_lr=0.1 --feature_lr=0.001 --num_episodes=20 --max_steps=17 --op_coef=0.2 --ent_coef=0.01 --n_splits=5 --top_k=13 


python new2_main_final.py --dataset=openml_620 --id=123 --critic_lr=0.001 --feature_lr=0.001 --num_episodes=10 --max_steps=22 --op_coef=0.2 --ent_coef=0.1 --n_splits=5 --top_k=9

python new2_main_final.py --dataset=openml_637 --id=123 --critic_lr=1e-05 --feature_lr=1e-05 --num_episodes=20 --max_steps=19 --op_coef=0.02 --ent_coef=0.1 --n_splits=5 --top_k=19


python new2_main_final.py --dataset=openml_586 --id=123 --critic_lr=1e-05 --feature_lr=1e-05 --num_episodes=100 --max_steps=23 --op_coef=0.01 --ent_coef=0.1 --n_splits=5 --top_k=9



