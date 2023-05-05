# Implement-papaer-using-lightning

### What is lightning?

[Lightning Official website](https://lightning.ai/), [Lightning Story](https://lightning.ai/pages/blog/introducing-lightning-2-0/)

Lightning은 pytorch를 단순하게 설계하기 위해 만들어 졌으며, pytorch lightning과 fabric이 분리돼있다.

![Lightning history](https://camo.githubusercontent.com/edbf0564ec826ef611ec8cdeb805144c5d555d29b304de881607610c907013c7/68747470733a2f2f706c2d7075626c69632d646174612e73332e616d617a6f6e6177732e636f6d2f6173736574735f6c696768746e696e672f636f6e74696e75756d2e706e67)

Fabric은 다음과 같이 정의하고 있다.

> Lightning Fabric creates a continuum between raw PyTorch and the fully-managed PyTorch Lightning experience. It allows you to supercharge your PyTorch code through accelerators, distributed strategies, and mixed precision, while still retaining full control on your training loop.
>> Lightning Fabric는 원시 PyTorch와 완전히 관리되는 PyTorch Lightning 경험 사이에 연속성을 만들어줍니다. 이를 통해 가속기, 분산 전략 및 혼합 정밀도를 활용하여 PyTorch 코드를 최적화할 수 있으면서, 여전히 훈련 루프에 대한 완전한 제어권을 유지할 수 있습니다.

쉽게 말해, Fabric은 기존의 pytorch와 pytorch lightning 사이에서 관리하지 못했던 가속기, 분산 전략 및 혼합 정밀도등을 fabric을 통해 관리있도록 만들었다.

### Requirements

```
pip install -r requirements.txt
```

### Run

```
usage: torchrun run.py [-n] [--data_path] [--img_size]

optional arguments:
  -h, --help            show this help message and exit
  -n, --model_name      Please refer to the model list in the README.md file

  --data_path        Path of dataset
  --img_size            The image is resized to the suggested size.
```

### Model list
# Implement-papaer-using-lightning

### What is lightning?

[Lightning Official website](https://lightning.ai/), [Lightning Story](https://lightning.ai/pages/blog/introducing-lightning-2-0/)

Lightning은 pytorch를 단순하게 설계하기 위해 만들어 졌으며, pytorch lightning과 fabric이 분리돼있다.

![Lightning history](https://camo.githubusercontent.com/edbf0564ec826ef611ec8cdeb805144c5d555d29b304de881607610c907013c7/68747470733a2f2f706c2d7075626c69632d646174612e73332e616d617a6f6e6177732e636f6d2f6173736574735f6c696768746e696e672f636f6e74696e75756d2e706e67)

Fabric은 다음과 같이 정의하고 있다.

> Lightning Fabric creates a continuum between raw PyTorch and the fully-managed PyTorch Lightning experience. It allows you to supercharge your PyTorch code through accelerators, distributed strategies, and mixed precision, while still retaining full control on your training loop.

>> Lightning Fabric는 원시 PyTorch와 완전히 관리되는 PyTorch Lightning 경험 사이에 연속성을 만들어줍니다. 이를 통해 가속기, 분산 전략 및 혼합 정밀도를 활용하여 PyTorch 코드를 최적화할 수 있으면서, 여전히 훈련 루프에 대한 완전한 제어권을 유지할 수 있습니다.

쉽게 말해, Fabric은 기존의 pytorch와 pytorch lightning 사이에서 관리하지 못했던 가속기, 분산 전략 및 혼합 정밀도등을 fabric을 통해 관리있도록 만들었다.

### Requirements

```

pip install -r requirements.txt

```

### Run

```

usage: python3 run.py [--config] [--data_path] [--log_path]
                      [--ckpt_path] [--num_workers] [--batch_size] 
                      [--max_epochs]

optional arguments:

  -h, --help            show this help message and exit.
  --config,             Path of model config file.
  --data_path           Path of data.
  --log_path            Path of lightning logs.
  --ckpt_path           Path of ckpt.
  --num_workera         Number of DataLoader workers.
  --batch_size          Data batch size.
  --max_epochs          Epoch lenghts.

```
