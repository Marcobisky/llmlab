python -m pip install -r requirements.txt
python data.py --config config/expr_500k_depth5.yaml
python data.py --config config/eval_10k_depth5.yaml
python grpo.py --config config/teacher_grpo.yaml

python visualize_loss.py --config config/teacher_pretrain.yaml config/teacher_grpo.yaml
python visualize_weight.py --config config/teacher_pretrain.yaml config/teacher_grpo.yaml
python visualize_loss.py --config config/teacher_grpo.yaml
python visualize_weight.py --config config/teacher_grpo.yaml