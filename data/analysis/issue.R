# Follow-up D (2026-11-22): Sarcasm_NLP asked about "the topic", "nuclear power", "the nuclear issue
# it discusses (...)" and the same plus "A news headline that only reports facts is neutral."
# Usage: Rscript issue.R <issue.csv> > issue-<date>.txt   (issue.csv from issue_dados.py)
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(readr) })
options(width = 220, dplyr.summarise.inform = FALSE)
d <- read_csv(commandArgs(TRUE)[1], show_col_types = FALSE) |>
  filter(model != "twitter-roberta") |> rename(theme = topic) |>
  mutate(set = ifelse(sarcastic == 1, "iron", "nonir"),
         target = factor(target, c("topic", "nuclear", "issue", "issuenote")))
r1 <- function(x) round(100 * x, 1)
mc <- function(x, y) { b <- sum(x == 1 & y == 0); c <- sum(x == 0 & y == 1); if (b + c == 0) 1 else binom.test(b, b + c)$p.value }
cat("## 0. Cells\n"); print(count(d, model, target, variant) |> pivot_wider(names_from = variant, values_from = n), n = Inf)
cat("\n## 1. Accuracy by set, v1 (DeBERTa-NLI: v1 only) and mean over orders\n")
print(d |> filter(variant == "v1") |> group_by(set, model, target) |> summarise(acc = r1(mean(correct))) |>
  pivot_wider(names_from = target, values_from = acc) |> arrange(set, model), n = Inf)
print(d |> group_by(set, model, target) |> summarise(acc = r1(mean(correct))) |>
  pivot_wider(names_from = target, values_from = acc) |> arrange(set, model), n = Inf)
cat("\n## 2. McNemar vs nuclear, v1\n")
w <- d |> filter(variant == "v1") |> select(model, id, set, target, correct) |> pivot_wider(names_from = target, values_from = correct)
print(w |> group_by(set, model) |> summarise(n = n(), d_issue = r1(mean(issue - nuclear)), p_issue = signif(mc(nuclear, issue), 2),
  d_note = r1(mean(issuenote - nuclear)), p_note = signif(mc(nuclear, issuenote), 2)), n = Inf)
cat("\n## 3. By topic x gold, v1, accuracy\n")
print(d |> filter(variant == "v1", theme %in% c("Energy", "Politics", "General")) |> group_by(set, theme, gold, model, target) |>
  summarise(n = n(), acc = r1(mean(correct))) |> pivot_wider(names_from = target, values_from = acc) |> arrange(set, theme, gold, model), n = Inf)
cat("\n## 4. Answer distribution non-ironic, v1 (gold 27.3 / 61.7 / 11.0)\n")
print(d |> filter(variant == "v1", set == "nonir") |> group_by(model, target) |>
  summarise(neg = r1(mean(pred == "negative")), neu = r1(mean(pred == "neutral")), pos = r1(mean(pred == "positive"))), n = Inf)
cat("\n## 5. Irony cost on negatives (non-ironic neg recall - ironic neg recall), v1\n")
print(d |> filter(variant == "v1") |> group_by(model, target) |>
  summarise(neg_nonir = r1(mean(correct[set == "nonir" & gold == "negative"])), neg_iron = r1(mean(correct[set == "iron" & gold == "negative"]))) |>
  mutate(cost = neg_nonir - neg_iron), n = Inf)

cat("\n## 6. All 5,929 items (ironic + non-ironic), v1 and mean over orders; macro-F1 v1\n")
f1 <- function(g, p) mean(sapply(c("negative", "neutral", "positive"), function(l) {
  tp <- sum(g == l & p == l); pr <- tp / max(sum(p == l), 1); rc <- tp / max(sum(g == l), 1); if (pr + rc == 0) 0 else 2 * pr * rc / (pr + rc) }))
print(d |> group_by(model, target) |> summarise(acc_v1 = r1(mean(correct[variant == "v1"])), acc_mean = r1(mean(correct)),
  f1_v1 = r1(f1(gold[variant == "v1"], pred[variant == "v1"]))) |> arrange(model, target), n = Inf)
