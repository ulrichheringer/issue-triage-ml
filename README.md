# issue-triage-ml

MVP clássico para sugerir possíveis issues duplicadas com TF-IDF e estimar horas de resolução com Ridge sobre TF-IDF.

## Dados

O projeto usa apenas dados sintéticos gerados em `pipeline.py` com semente fixa. Não representa um benchmark público nem dados de usuários. A avaliação retém pares duplicados idênticos e templates de resolução por grupo para evitar que o mesmo exemplo-base apareça no treino e no teste.

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

Na execução determinística atual (seed 42): duplicatas — acurácia **0,6231**, F1 **0,4842**; resolução — MAE **2,9106 h**, RMSE **3,4901 h**, R² **0,0202**. O R² baixo no holdout por template mostra que este baseline não generaliza bem a descrições não vistas. Os números não indicam desempenho em issues reais.

Limitação principal: o vocabulário e a distribuição dos dados são pequenos e artificiais. Para uso real, substitua os geradores por dados históricos licenciados e avalie por período ou por projeto para reduzir vazamento entre issues relacionadas.
