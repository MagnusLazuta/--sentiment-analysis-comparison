# Numbers quoted in the paper text that depend on Sarcasm_NLP: old run ("the topic") vs merged
# run (target "nuclear power"). Usage: Rscript recalc_snlp.R > recalc_snlp-20261121.txt
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(readr) })
options(width = 200, dplyr.summarise.inform = FALSE)
runs <- c(old = "../results/20261117-181106-838446/eval", new = "../results/20261121-220717-839516/merged/eval")
r1 <- function(x) round(100 * x, 1)
six <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
for (nm in names(runs)) {
  cat("\n######## ", nm, "\n")
  p <- read_csv(file.path(runs[nm], "predictions.csv"), show_col_types = FALSE,
                col_select = c(model, dataset, id, sarcastic, gold, condition, variant, pred, correct, confidence)) |>
    filter(model %in% six)
  m <- read_csv(file.path(runs[nm], "metrics.csv"), show_col_types = FALSE) |> filter(model %in% six)
  p1 <- p |> filter(variant %in% c("v1", "none"), condition == "no_context")
  s <- p1 |> filter(dataset == "sarcasm_nlp")
  cat("## SNLP acc nonir / iron, neg-rate nonir, neutral-rate nonir, prec of 'negative' nonir\n")
  print(s |> group_by(model) |> summarise(acc_nonir = r1(mean(correct[sarcastic == 0])), acc_iron = r1(mean(correct[sarcastic == 1])),
    neg_rate = r1(mean(pred[sarcastic == 0] == "negative")), neu_rate = r1(mean(pred[sarcastic == 0] == "neutral")),
    prec_neg = r1(mean(gold[sarcastic == 0 & pred == "negative"] == "negative")),
    neg_nonir = r1(mean(correct[sarcastic == 0 & gold == "negative"])), neg_iron = r1(mean(correct[sarcastic == 1 & gold == "negative"]))) |>
    mutate(loss = neg_nonir - neg_iron))
  cat("## T1 recall of negatives vs T3 accuracy, Spearman over 6 models x 3 datasets (18 pairs)\n")
  x <- p1 |> filter(dataset %in% c("mustardpp", "sarcasm_nlp", "filatova")) |> group_by(model, dataset) |>
    summarise(r1n = mean(correct[sarcastic == 0 & gold == "negative"]), a3 = mean(correct[sarcastic == 1]))
  cat("n =", nrow(x), "spearman =", round(cor(x$r1n, x$a3, method = "spearman"), 3), "\n")
  cat("## ECE / F1 / acc non-sarcastic, no context, v1/none (SNLP and MT-CSD)\n")
  print(m |> filter(variant %in% c("v1", "none"), condition == "no_context", group == "non_sarcastic",
                    dataset %in% c("sarcasm_nlp", "mtcsd")) |>
    select(dataset, model, n, accuracy, macro_f1, ece, `confident_errors_0.9`) |> arrange(dataset, model) |>
    mutate(across(c(accuracy, macro_f1, ece, `confident_errors_0.9`), ~ round(., 3))), n = Inf)
  cat("## Share of non-ironic items with confidence >= 0.9, no context, v1/none, all datasets\n")
  print(p1 |> filter(sarcastic == 0) |> group_by(model) |> summarise(n = n(), conf09 = r1(mean(confidence >= 0.9))))
  cat("## Calibration figure numbers: confident errors (weighted over the figure cells), same answer in 3 orders\n")
  cells <- tribble(~dataset, ~condition, ~group, "mustardpp", "no_context", "non_sarcastic", "mtcsd", "no_context", "non_sarcastic",
    "mustardpp", "with_context", "non_sarcastic", "mtcsd", "with_context", "non_sarcastic", "mustardpp", "no_context", "sarcastic",
    "sarcasm_nlp", "no_context", "sarcastic", "filatova", "no_context", "sarcastic")
  cal <- m |> filter(variant %in% c("v1", "none")) |> inner_join(cells, by = c("dataset", "condition", "group"))
  print(cal |> group_by(model) |> summarise(conf_err = r1(weighted.mean(`confident_errors_0.9`, n)),
    ece_T3 = round(weighted.mean(ece[group == "sarcastic"], n[group == "sarcastic"]), 3)))
  print(p |> filter(variant %in% c("v1", "v2", "v3")) |> select(model, id, condition, variant, pred) |>
    pivot_wider(names_from = variant, values_from = pred) |> group_by(model) |>
    summarise(same3 = r1(mean(v1 == v2 & v2 == v3))))
  cat("## Kev vs Qwen, non-ironic, no context, v1: neutral rate, recall neutral/positive, agreement, by dataset\n")
  kq <- p1 |> filter(model %in% c("kev-4b", "qwen3.5-4b"), sarcastic == 0)
  print(kq |> group_by(dataset, model) |> summarise(neu = r1(mean(pred == "neutral")), rec_neu = r1(mean(correct[gold == "neutral"])),
    rec_pos = r1(mean(correct[gold == "positive"])), acc = r1(mean(correct))), n = Inf)
  print(kq |> select(model, dataset, id, pred) |> pivot_wider(names_from = model, values_from = pred) |>
    group_by(dataset) |> summarise(agree = r1(mean(`kev-4b` == `qwen3.5-4b`))))
  cat("## All six wrong (majority over orders not used: v1/none, no context), % of items per dataset\n")
  print(p1 |> group_by(dataset, id) |> summarise(allwrong = all(correct == 0), k = n()) |>
    group_by(dataset) |> summarise(n = n(), pct = r1(mean(allwrong))))
}
