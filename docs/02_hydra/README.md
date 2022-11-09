# Configuration 관리하기: Hydra

PyTorch 실험을 통해 좋은 모델을 찾다보면, 어떤 설정값들을 사용했는지 기록하는 것이 굉장히 중요합니다. 하지만 내가 어떤 설정값들을 변경하고 있는지, 어떤 설정값에서 최고의 성능이 나왔는지 관리하는 것이 생각보다는 어려운데, 개인적으로 경험한 문제들은 세 가지로 나눠볼 수 있을 것 같은데요.
1. 설정값을 넣어주는 방식
    이는 argparse와 같은 라이브러리를 통해 내가 사용하고자 하는 설정값을 코드에 뿌려주는 방법을 뜻합니다. 자료형을 어떤 것을 쓰는지가 항상 귀찮은 이슈인데요, 딕셔너리 형태로 주게 되면, 매번 슬라이싱으로 뽑아야 하는 귀찮음이 있고,  `.` 호출이 가능한 `dataclass`로 관리하자니 `.json`이나 다른 형태로 설정값을 저장하는 방식을 만들어줘야해서 귀찮음이 있습니다.
2. 설정값을 로컬에서 관리하는 방식
    dataclass를 사용하면 계속 스크립트 내부에 설정값을 관리해야 하거나 `.json`으로 저장을 해야 하는데 이걸 다시 직접 정형화하는 것도 귀찮습니다. 그리고 `.json`은 웹에서 많이 쓰이긴 하지만, 실제로는 가독성이 좋은 편이 아니기도하고 파일 입출력에 있어서 indent나 원소들의 순서 등 신경 써줘야할 부분들이 더 있기 때문에 용이하진 않습니다. 조금 더 편한 `.yaml` 파일도 존재하지만 이렇게 설정값들을 관리하는 것은 많이 피곤합니다.
    
3. 설정값을 원격에서 관리하는 방식
    이는 wandb와 같은 실험을 돕는 툴들로 해결이 가능합니다.

위와 같은 어려움들을 해결하기 위해 저도 많은 라이브러리를 사용해보았는데요.
+ `argparse`, `configparser`: 둘 다 python built-in library인데 argparse의 경우 config를 하나 추가할 때마다 추가해야하는 라인도 많고 가독성이 심히 불편합니다.
+ `easydict` dictionary값들을 `.` 호출이 가능하지만, 생각보다 dictionary로 다시 변환하고 변환하는 작업이 귀찮습니다.
+ `dataclass`: 직접 class를 구성해서 안에 넣고 싶은 값들을 만들어주는데, 어쨌든 python 문법으로 모든 걸 작성해야 하기 떄문에 약간의 귀찮음이 수반됩니다. 그리고 설정값들을 파일로 저장하는 것도 별도로 구현해야하거나, `argparse-dataclass`라는 [라이브러리](https://pypi.org/project/argparse-dataclass/)가 있긴하지만, 불편한 건 여전합니다. 또한 huggingface에서 지원하는 `HfArgumentParser`가 있긴한데, 자연어처리 외의 task를 할 때 굳이 huggingface를 깔아야하는 불편함이 있습니다.


## `.yaml` 사용하기
hydra는 기본적으로 `.yaml` 파일을 선택해서 사용하는데요 간단한 yaml 문법을 알아봅시다. Yaml 또한 json과 마찬가지로 key-value 형태의 값을 저장할 수 있는 파일인데요, 사용법과 가독성이 json보다 (개인적으로) 우수하다는 점이 있습니다. ([참고1](https://stackoverflow.com/questions/1726802/what-is-the-difference-between-yaml-and-json), [참고2](http://yaml.org/spec/1.2-old/spec.html#id2759572)) 문법은 항상 **key: value** 형태로 진행되며, value의 내용이 많다면 개행 후 탭으로 그 깊이를 구분합니다. 사용법은 [위키](https://en.wikipedia.org/wiki/YAML#Basic_components)에 굉장히 자세히 나와있는데 우리가 필요한 몇 가지만 살펴봅시다.

.yaml에서 이해할 수 있는 자료형 중 대표적인 것은 숫자형/boolean/리스트/문자열입니다. 이 네 가지만 잘 사용해도 문제 없이 yaml을 사용할 수 있는데요, 바로 아래와 같이요.

```yaml
learning_rate: 1e-3
num_classes: 2
data_dir: /opt/ml/
hidden_dim: [64, 128, 128, ${num_classes}] # !우리 코드에 사용되지 않습니다
# 혹은
hidden_dim:
    - 64
    - 128
    - 128
    - ${num_classes}

data:
  asset_dir: asset/
  data_dir: /opt/ml/input/data
  file_name: train_data.csv
```
+ 문자열의 경우 따옴표('', "")를 별도로 넣지 않아도 문자열로 인식합니다.
+ 숫자형의 경우 1e-3과 같은 지수형태도 인식합니다.
+ Boolean의 경우 True, true, False, false, null 등 전부 사용할 수 있습니다.
+ key-value에서 value가 또 다시 key-value를 가질 수 있습니다.
+ 다른 key-value를 참조할 수 있습니다. (${참조할변수명})

## hydra

이제 본격적으로 hydra를 사용해봅시다. hydra의 기본적인 사용법은 먼저 설정값에 활용할 `.yaml` 파일을 만들어준 후에 진행되는데요, 위와 같은 파일이 있었다고 가정하고, 우리가 실질적으로 돌려야할 함수 위에 decorator를 걸어주고, 인자에 config를 넣어주면 완료됩니다.

```python
import hydra

@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def main(config: omegaconf.DictConfig = None) -> None:
    return train(config)
```

이러면 `config` 인자에는 방금 `.yaml` 파일 내에 들어있는 자료들을 key-value 형태로 사용할 수 있게됩니다. 단순히 dict 형태로 쓸 수 있는 것 이외의 몇 가지 기능들이 있습니다.

### `hydra.utils.instantiate`

Learning rate scheduler나 loss function을 바꿔줄 때는 거의 추가적인 코드를 수반하게 됩니다. 가령 BCEwithlogitloss에서 L1Loss로 바꿔준다고 하면, arguments에서 이름만 받아주고 `get_loss`와 같은 함수로 직접 내부에서 객체화 시켜줘야하는데요, 이것을 쉽게 수행시킬 있도록 도와주는 hydra의 기능이 있습니다.

우리 베이스라인 코드에서 모델을 바꿔주는 경우를 예로 들어봅시다. 현재 모델은 code/dkt/model.py 안에 LSTM, LSTMATTN, BERT 총 3개의 모델이 있는데요, 베이스라인 코드 내부에서는 args.model 인자를 변경하여 수행한 후, `get_model` 이라는 code/dkt/trainer.py 스크립트 내의 함수에서 이를 수행해주게 됩니다.

여기서 hydra를 이용해 코드 수를 줄여봅시다. 아래 예제 코드는 전체적인 부분이 많이 생략된 의사코드입니다. 자세한 내용은 [Hydra instantiate with objects](https://hydra.cc/docs/advanced/instantiate_objects/overview/)를 참고해봅시다.
```python
# model.yaml
model:
  _target_: code.dkt.src.model.LSTM

# In python
import hydra
model = hydra.utils.instantiate(config.model)
```

### Extending Configs

실험을 관리하다보면 여러 개의 다른 모델이나 다른 손실함수를 사용해야 하는 경우가 있는데, 각 종류 별로 다르게 설정값을 넣어줘야 할 수 있습니다. 가령 learning rate scheduler의 경우 ReduceLROnPlateau와 CosineAnnealing은 서로 다른 인자를 받게 되어 있습니다. 이런 경우를 대비해 configuration yaml 파일을 여러 개 둘 수 있는데요, 기본적인 문법은 아래와 같습니다. 예제의 경우 모델을 예로 들어봅시다.

```bash
configs
⎥-- default.yaml
⎿-- model
    ⎥-- lstm.yaml
    ⎥-- lstmattn.yaml
    ⎿-- bert.yaml
```
```yaml
# default.yaml
defaults:
  - model: lstm
```

이런 식으로 세부적인 설정값들을 분리한 후, defaults라는 최상위 config.yaml에서 파일 명을 적어 이용할 수 있습니다. 처음에는 복잡해보이지만, 익숙해지면 많이 편한 구조입니다. [Documentation](https://hydra.cc/docs/patterns/extending_configs/)

### In Command Line Interface

설정값에서 제일 중요한 건, 터미널에서도 설정값을 조절할 수 있느냐인데요, 일반적인 argparse와 유사하게 사용할 수 있습니다. 계층깊이가 내려갈 때마다 `.`을 하나씩 적어서 접근할 수 있습니다. 가령 맨 위에서 나온 예시 중 `data` 내에 `data_dir`를 변경하고 싶으면, 아래처럼 사용하면 됩니다.
```bash
python.py data.data_dir="./data"
```


## 마치며

포스트를 통해 설정값을 쉽게 코드에 뿌려주고 관리할 수 있는 hydra에 대해 알아보았는데요, 이 외에도 이를 돕는 툴은 여러 가지 있습니다. 대표적으로 google에서 개발한 [`fire`](https://github.com/google/python-fire)라는 라이브러리인데요, 이 또한 CLI에서 설정값을 쉽게 뿌려줄 수 있도록 설계된 라이브러리입니다. 설정값 관리는 딥러닝 실험에서 가장 중요하지만 귀찮은 작업이기 때문에, 빨리 손에 익는 방법을 터득하는 것을 추천합니다!