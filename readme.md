# Classifying the presence or absence of a disease using chest X-ray data


## Experiments
|Model|CV|LB| EPOCHS | Transforms |
 |---|---|---|---|---|
|efficientnet-b0|0.92|0.209433|8|rm_vt_flip|



## Default
```bash
"Train"
$python main.py --action train --seed 0 --model efficientnet-b6 --epochs 100 --batchsize 32 --savepath savemodel

"TEST"
$python main.py --action test --seed 0 --model efficientnet-b6 --epochs 100 --batchsize 64 --savepath savemodel
```
