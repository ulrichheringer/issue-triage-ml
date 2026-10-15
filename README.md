# issue-triage-ml

MVP clássico para sugerir possíveis issues duplicadas com TF-IDF e estimar horas de resolução com Ridge sobre TF-IDF.

## Dados

O projeto usa apenas dados sintéticos gerados em `pipeline.py` com semente fixa. Não representa um benchmark público nem dados de usuários. A divisão de avaliação é holdout aleatório de 25%.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib --input '{"issue_a":"login timeout after password reset","issue_b":"sign in request times out","issue":"query times out under load"}'
```

## Métricas obtidas

Na execução determinística atual (seed 42): duplicatas — acurácia **0,6833**, F1 **0,5366**; resolução — MAE **1,8374 h**, RMSE **2,3171 h**, R² **0,4209**. Os números são referências do gerador sintético; não indicam desempenho em issues reais.

Limitação principal: o vocabulário e a distribuição dos dados são pequenos e artificiais. Para uso real, substitua os geradores por dados históricos licenciados e avalie por período ou por projeto para reduzir vazamento entre issues relacionadas.
