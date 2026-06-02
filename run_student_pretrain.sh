python -m pip install -r requirements.txt
python data.py --config config/expr_200k_depth4.yaml
python data.py --config config/eval_10k_depth5.yaml
python pretrain.py --config config/student_pretrain.yaml

python visualize_loss.py --config config/student_pretrain.yaml
python visualize_weight.py --config config/student_pretrain.yaml