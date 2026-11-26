# Kev vs Qwen agreement over orders and temperature-scaled ECE (T = 2.41) on Sarcasm_NLP, old vs new.
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(readr) })
runs <- c(old = "../results/20261117-181106-838446/eval", new = "../results/20261121-220717-839516/merged/eval")
ece <- function(conf, ok, b = 15) { k <- pmin(floor(conf * b), b - 1); sum(tapply(seq_along(conf), k, function(i) length(i) * abs(mean(ok[i]) - mean(conf[i])))) / length(conf) }
for (nm in names(runs)) {
  p <- read_csv(file.path(runs[nm], "predictions.csv"), show_col_types = FALSE,
                col_select = c(model, dataset, id, sarcastic, gold, condition, variant, pred, correct, confidence, p_negative, p_neutral, p_positive)) |>
    filter(model %in% c("kev-4b", "qwen3.5-4b"), condition == "no_context", sarcastic == 0)
  cat(nm, "agreement Kev-Qwen by dataset x order:\n")
  print(p |> select(model, dataset, id, variant, pred) |> pivot_wider(names_from = model, values_from = pred) |>
    group_by(dataset, variant) |> summarise(a = round(100 * mean(`kev-4b` == `qwen3.5-4b`), 1), .groups = "drop") |>
    pivot_wider(names_from = variant, values_from = a))
  for (ds in c("sarcasm_nlp", "mustardpp")) {
    q <- p |> filter(model == "qwen3.5-4b", variant == "v1", dataset == ds)
    P <- as.matrix(q[, c("p_negative", "p_neutral", "p_positive")])^(1 / 2.41); P <- P / rowSums(P)
    pr <- c("negative", "neutral", "positive")[max.col(P)]
    k <- p |> filter(model == "kev-4b", variant == "v1", dataset == ds)
    cat(nm, ds, "Qwen ECE raw", round(ece(q$confidence, q$correct), 3), "T=2.41", round(ece(apply(P, 1, max), pr == q$gold), 3),
        "Kev", round(ece(k$confidence, k$correct), 3), "\n")
  }
}
