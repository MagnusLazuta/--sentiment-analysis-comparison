"""Summary of input tokens per prediction, by model, dataset and condition (2026-11-23).

Sources: results/tokens-840068/tokens_{main,laya}.csv (src/tokens_entrada.py, job 840068, main run
20261117-181106-838446); Kev-4B from that run's predictions.csv (server usage.input_tokens of one
request answering the 3 option orders, divided by 3).
Run: uv run --with pandas python tokens_resumo.py
Output: tokens_resumo-20261123.csv
"""
import pandas as pd

R = "../results"
d = pd.concat([pd.read_csv(f"{R}/tokens-840068/tokens_main.csv"), pd.read_csv(f"{R}/tokens-840068/tokens_laya.csv")])
k = pd.read_csv(f"{R}/20261117-181106-838446/eval/predictions.csv",
                usecols=["model", "dataset", "id", "condition", "variant", "tokens"])
k = k[k.model == "kev-4b"].assign(tokens=lambda x: x.tokens / 3, passes=1 / 3, tokens_max=None)
d = pd.concat([d, k])
lab = {"twitter-roberta": "twitter-roberta", "deberta-nli": "DeBERTa-NLI", "qwen3.5-4b": "Qwen3.5-4B",
       "mistral-7b": "Mistral-7B", "kev-4b": "Kev-4B", "laya": "Laya", "nanojev": "NanoJev"}
ds = {"mustardpp": "MUStARD++", "mtcsd": "MT-CSD", "sarcasm_nlp": "Sarcasm_NLP", "filatova": "Filatova"}
cond = {"no_context": "sem contexto", "with_context": "com contexto"}


def summ(g):
    return pd.Series({"n_previsoes": len(g), "passadas_por_previsao": round(g.passes.mean(), 3),
                      "tokens_por_previsao_media": round(g.tokens.mean(), 1),
                      "tokens_por_previsao_mediana": round(g.tokens.median(), 1),
                      "tokens_por_previsao_max": round(g.tokens.max(), 1),
                      "tokens_total": round(g.tokens.sum()),
                      "nanojev_limite_superior_media": round(pd.to_numeric(g.tokens_max, errors="coerce").mean(), 1)})


rows = [d.groupby("model").apply(summ).assign(base="todas", condicao="todas"),
        d.groupby(["model", "condition"]).apply(summ).reset_index("condition").assign(base="todas"),
        d.groupby(["model", "dataset", "condition"]).apply(summ).reset_index(["dataset", "condition"])]
out = pd.concat(rows).reset_index()
out = out.rename(columns={"dataset": "base_raw", "condition": "cond_raw"})
out["base"] = out.base_raw.map(ds).fillna(out.base)
out["condicao"] = out.cond_raw.map(cond).fillna(out.condicao)
out["modelo"] = out.model.map(lab)
out["modelo"] = pd.Categorical(out.modelo, list(lab.values()))
out = out.sort_values(["modelo", "base", "condicao"])
cols = ["modelo", "base", "condicao", "n_previsoes", "passadas_por_previsao", "tokens_por_previsao_media",
        "tokens_por_previsao_mediana", "tokens_por_previsao_max", "tokens_total", "nanojev_limite_superior_media"]
out[cols].to_csv("tokens_resumo-20261123.csv", index=False)
print(out[cols].to_string(index=False))
