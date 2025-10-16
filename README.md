📊 Análise Didática de Ações da B3 — v12

ML Ensemble + NeuralProphet + LSTM + Optuna + Indicadores Avançados + PDF

Ferramenta educacional para entender indicadores técnicos e testar previsões de ações da B3 🇧🇷.
Combina modelos clássicos (ensemble), séries neurais (NeuralProphet) e agora LSTM multivariado, com tuning via Optuna, indicadores adicionais (ADX, MACD, Bollinger) e exportação em PDF.

Aviso: Projeto didático. Não é recomendação de investimento.

✨ O que há de novo na v12

🔮 Aba “LSTM” (experimental): modelo recorrente multivariado (TensorFlow/Keras) para prever retorno futuro usando janelas temporais (lookback) e várias features (RSI, distâncias às médias, retornos). Mostra MAE/MAPE e R² de retornos + previsão out-of-sample do próximo ponto.

🧭 Aba “Optuna”: busca automática de hiperparâmetros do LSTM (lookback, units, dropout, learning rate…). Retorna melhor MAE e o conjunto ótimo.

📈 Indicadores adicionais (opcionais no gráfico): MACD, Bandas de Bollinger e ADX (+DI/−DI).

🗂️ Aba “PDF”: gera relatório com preço, Δ vs SMA20, RSI, regime (SMA200), parâmetros de ML, KPIs OOS e observações.

🧠 NeuralProphet reforçado: previsões históricas + futuras com cores diferentes, métricas R²/MAE/MAPE, e scatter (real vs previsto).

Mantém tudo da v11: Modo simples + Presets, Confiabilidade & Trades, tema escuro fixo. 

README

🧱 Estrutura do projeto
.
├── streamlit_app.py
├── b3_utils.py
├── data/
│   └── b3_tickers.csv           # lista de tickers da B3
├── .streamlit/
│   └── config.toml               # tema escuro padrão
├── requirements.txt
└── README.md


Dados: Yahoo Finance via yfinance (preços ajustados).

🚀 Como executar
# Python 3.10+ recomendado
pip install -r requirements.txt
streamlit run streamlit_app.py


Abra no navegador o endereço exibido (geralmente http://localhost:8501
).

Publicar no Streamlit Cloud

Envie o projeto para o GitHub (estrutura acima).

No Streamlit Cloud: New app → selecione repo/branch → streamlit_app.py.

Se faltar dependência, ajuste requirements.txt.

🪄 Modo simples + Presets (para começar rápido)
Preset	Objetivo	Limiar (otimização)	min_prob	Banda neutra	Tendência (Preço>SMA200)	Contrarian (RSI<30)	Dist. máx. à SMA20	Holding	Splits/Teste	Custos
Conservador	reduzir drawdown	Calmar OOS	0.62	±0.06	ON	OFF	−0.03	3d	5 / 60	8/5 bps
Balanceado	suavizar	Sharpe OOS	0.58	±0.05	ON	ON	−0.05	2d	5 / 60	6/3 bps
Agressivo	retorno bruto	Retorno OOS	0.54	±0.03	OFF	ON	−0.08	1d	4 / 40	6/3 bps

Aplique um preset e, se quiser, ajuste manualmente.

⚙️ Parâmetros (explicação simples)

Horizonte (dias): em quantos dias o modelo tenta prever se sobe (1 = reativo; 5–10 = mais suave).

Método do limiar: como escolhemos o corte de probabilidade que vira sinal:

Sharpe OOS: consistência risco/retorno,

Calmar OOS: retorno considerando menor drawdown,

Retorno OOS: retorno bruto,

Youden: foca taxa de acerto.

Confiança mínima (min_prob): probabilidade mínima para operar (↑ = menos trades, mais seletos).

Banda neutra: faixa perto de 50% onde não opera (reduz ruído).

Tendência (Preço>SMA200): só opera a favor da tendência longa.

Contrarian (RSI≤30) + Distância à SMA20: permitem repiques controlados.

Splits / Test Size: validação temporal (walk-forward). Poucos dados → splits=3–4, test_size=30–60.

Custos / Slippage (bps): fricções por operação.

Holding mínimo: dias mínimos em posição (evita entra-e-sai).

📈 Indicadores Técnicos (inclui extras)

SMA20/50/200 — médias móveis de curto/médio/longo prazo.

RSI(14) — “termômetro” de força (≥70 sobrecompra; ≤30 sobrevenda).

MACD — momentum (linha MACD, sinal e histograma).

Bandas de Bollinger — desvio-padrão em torno da média (expande/contrai).

ADX (+DI/−DI) — força da tendência e direção.

Ative/desative indicadores avançados no sidebar.

🤖 Previsão com ML Ensemble (didático)

Features: retornos, volatilidade, RSI, distâncias às SMA20/50/200…

Alvo: 1 se Close[t+h] > Close[t] (subiu em h dias), senão 0.

Modelos: HGB + XGBoost + LightGBM → média de probabilidades (ensemble).

Calibração: deixa as probabilidades “honestas”.

Walk-forward CV: treino no passado e teste no futuro (OOS).

Limiar + filtros: banda neutra, tendência, contrarian, custos, holding.

Backtest: curva da estratégia x Buy & Hold (com custos/slippage).

Métricas chave:

Acurácia / Balanced Acc. — qualidade de acerto,

ROC AUC — vantagem estatística (0,5 ~ acaso; 0,60+ começa a ser útil),

Brier — calibração (↓ melhor),

OOS — tamanho da amostra fora do treino.

🧠 NeuralProphet (histórico + futuro)

Previsões no histórico (in-sample) e projeções futuras (out-of-sample).

Cores diferentes no gráfico: histórico (pontilhado) vs futuro (contínuo).

Métricas: R² (nível e retornos), MAE e MAPE.

Scatter “real vs previsto” (nível e retornos) com linha ideal.

Defina dias à frente (1–365) e baixe CSVs.

Ótimo para tendências e sazonalidades. Combine com o ensemble (curto prazo).

🔮 LSTM Multivariado (experimental)

Usa janelas (lookback) com várias features (retornos, RSI, distâncias às médias…) para prever retorno futuro (regressão).

Treinamento com TensorFlow/Keras e exibição de MAE/MAPE e R² de retornos no teste.

Faz previsão out-of-sample do próximo ponto.

Controles: lookback, horizonte, epochs, unidades, dropout.

Captura dependências temporais; demanda mais dados e tempo que o ensemble.

Dependência: tensorflow-cpu (ou tensorflow) – opcional.

🧭 Optuna (tuning do LSTM)

Busca automática de hiperparâmetros (objetivo: MAE menor).

Exibe melhor trial e parâmetros ótimos.

Dependência: optuna – opcional.
Dica: use 10–20 trials para começar; tuning é pesado.

🗂️ PDF (relatório)

Gera PDF com: título, período, Fechamento, Δ vs SMA20, RSI, regime (SMA200), parâmetros de ML, KPIs OOS e observações.

Útil para compartilhar ou guardar histórico.

Dependência: reportlab – opcional.

📊 Confiabilidade & Trades

Curva de Confiabilidade (calibration): confere se “70%” previsto ≈ 70% observado.

Tabela de Trades (OOS): data de entrada/saída, retorno (%) e duração; download CSV.

Abaixo da diagonal ⇒ modelo otimista; acima ⇒ conservador.

🔧 Requisitos (adicione o que for usar no requirements.txt)
plotly>=5.0
streamlit>=1.36
yfinance>=0.2.40
pandas>=2.0
numpy>=1.23
scikit-learn>=1.3
xgboost>=1.7
lightgbm>=4.0
neuralprophet>=0.6.0      # NeuralProphet (opcional)
tensorflow-cpu>=2.12      # LSTM (opcional)
optuna>=3.0.0             # Tuning (opcional)
reportlab>=3.6.12         # PDF (opcional)


Em nuvem (Streamlit Cloud), evite GPU. Use tensorflow-cpu.

🧩 Glossário rápido

Candle (OHLC) • SMA (tendência) • RSI (força 0–100) • Sharpe/Calmar (retorno/risco) • Drawdown (queda máxima) • bps (0,01%) • Walk-forward (validação temporal).

🛠️ Solução de problemas

Sem dados / muitos NaNs: aumente período; reduza splits/test_size.

CV quebrando: splits=3–4 e test_size=30–60.

Lento no cloud: menos trials (Optuna), menos epochs (LSTM) e horizonte menor.

Dependência faltando: ajuste requirements.txt e redeploy.

📄 Licença e créditos

Defina uma licença (ex.: MIT).
Dados: yfinance • Clássico: scikit-learn / XGBoost / LightGBM • Séries: NeuralProphet • Deep: TensorFlow/Keras • UI: Streamlit • PDF: ReportLab.

Dica final: comece com Balanceado, veja Confiabilidade & Trades, compare estratégia x Buy & Hold, e depois explore NeuralProphet e LSTM. Boa análise! 📈
