# 📊 Análise Didática de Ações da B3 — v12

**ML Ensemble + NeuralProphet + LSTM + Optuna + Indicadores Avançados + PDF**

Ferramenta educacional para **entender** indicadores técnicos e **testar previsões** de ações da **B3** 🇧🇷.
Combina **modelos clássicos** (ensemble), **séries neurais** (NeuralProphet) e agora **LSTM multivariado**, com **tuning via Optuna**, **indicadores adicionais (ADX, MACD, Bollinger)** e **exportação em PDF**.

> **Aviso**: Projeto didático. **Não é recomendação de investimento.**

---

## ✨ O que há de novo na v12

* 🔮 **Aba “LSTM” (experimental)**: modelo recorrente multivariado (TensorFlow/Keras) para prever **retorno futuro** usando janelas temporais (lookback) e várias features (RSI, distâncias às médias, retornos). Mostra **MAE/MAPE** e **R² de retornos** + previsão out-of-sample do próximo ponto.
* 🧭 **Aba “Optuna”**: busca automática de hiperparâmetros do LSTM (lookback, units, dropout, learning rate…). Retorna **melhor MAE** e o **conjunto ótimo**.
* 📈 **Indicadores adicionais** (opcionais no gráfico): **MACD**, **Bandas de Bollinger** e **ADX (+DI/−DI)**.
* 🗂️ **Aba “PDF”**: gera **relatório** com preço, Δ vs SMA20, RSI, regime (SMA200), parâmetros de ML, KPIs OOS e observações.
* 🧠 **NeuralProphet reforçado**: previsões **históricas** + **futuras** com **cores diferentes**, métricas **R²/MAE/MAPE**, e **scatter** (real vs previsto).

Mantém tudo da v11: **Modo simples + Presets**, **Confiabilidade & Trades**, **tema escuro fixo**.
