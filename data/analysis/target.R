# Follow-up C (2026-11-21): Sarcasm_NLP with the annotated target ("nuclear power")
# vs the main run ("the topic"). Usage (from experimento/analysis):
#   Rscript target.R <target.csv> <followup.csv> > target-<date>.txt
# target.csv from target_dados.py; followup.csv from followup_dados.py (for the T1 -> T3
# correlation over the other datasets). Paired comparisons: exact McNemar. N per cell printed.
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(readr) })
options(width = 220, dplyr.summarise.inform = FALSE)
a <- commandArgs(TRUE)
d <- read_csv(a[1], show_col_types = FALSE)
r1 <- function(x) round(100 * x, 1)
mc <- function(x, y) { b <- sum(x == 1 & y == 0); c <- sum(x == 0 & y == 1); if (b + c == 0) 1 else binom.test(b, b + c)$p.value }
f1 <- function(g, p) mean(sapply(c("negative", "neutral", "positive"), function(l) {
  tp <- sum(g == l & p == l); pr <- tp / max(sum(p == l), 1); rc <- tp / max(sum(g == l), 1)
  if (pr + rc == 0) 0 else 2 * pr * rc / (pr + rc) }))
d <- d |> mutate(set = ifelse(sarcastic == 1, "iron", "nonir"), main = variant %in% c("v1", "none"))

cat("## 0. Cells (rows per model x prompt x target x variant)\n")
print(count(d, model, prompt, target, variant) |> pivot_wider(names_from = variant, values_from = n), n = Inf)
cat("\n## 0b. Determinism: twitter-roberta input does not contain the target (must be identical)\n")
print(d |> filter(model == "twitter-roberta") |> select(id, prompt, target, p_negative) |>
  pivot_wider(names_from = c(prompt, target), values_from = p_negative) |>
  summarise(n = n(), max_diff = max(abs(orig_nuclear - orig_topic), na.rm = TRUE)))

cat("\n## 1. Accuracy and macro-F1, v1/none, topic -> nuclear (McNemar on accuracy)\n")
w <- d |> filter(main) |> select(model, id, set, gold, prompt, target, pred, correct) |>
  pivot_wider(names_from = target, values_from = c(pred, correct))
print(w |> group_by(prompt, set, model) |>
  summarise(n = n(), acc_topic = r1(mean(correct_topic)), acc_nuc = r1(mean(correct_nuclear)),
            d = r1(mean(correct_nuclear - correct_topic)), p = signif(mc(correct_topic, correct_nuclear), 2),
            f1_topic = r1(f1(gold, pred_topic)), f1_nuc = r1(f1(gold, pred_nuclear)),
            same_pred = r1(mean(pred_topic == pred_nuclear))) |> arrange(prompt, set, model), n = Inf)

cat("\n## 1b. Same, mean over orders v1-v3 (LLMs and System One)\n")
print(d |> filter(variant != "none") |> group_by(prompt, set, model, target) |> summarise(acc = r1(mean(correct))) |>
  pivot_wider(names_from = target, values_from = acc) |> mutate(d = nuclear - topic) |> arrange(prompt, set, model), n = Inf)

cat("\n## 2. Answer distribution (% neg / neu / pos), non-ironic, v1/none; gold row first\n")
dist <- function(x) x |> summarise(n = n(), neg = r1(mean(pred == "negative")), neu = r1(mean(pred == "neutral")),
                                   pos = r1(mean(pred == "positive")))
print(d |> filter(main, model == "twitter-roberta", prompt == "orig", target == "topic", set == "nonir") |>
  mutate(pred = gold) |> group_by(set) |> dist())
print(d |> filter(main, set == "nonir") |> group_by(prompt, model, target) |> dist() |> arrange(prompt, model, target), n = Inf)

cat("\n## 3. By dataset topic column (Energy / Politics / General), non-ironic, orig, v1/none\n")
print(d |> filter(main, set == "nonir", prompt == "orig", topic %in% c("Energy", "Politics", "General")) |>
  group_by(topic) |> summarise(n = n_distinct(id), gold_neg = r1(mean(gold == "negative")),
                               gold_neu = r1(mean(gold == "neutral")), gold_pos = r1(mean(gold == "positive"))))
print(d |> filter(main, set == "nonir", prompt == "orig", topic %in% c("Energy", "Politics", "General")) |>
  group_by(topic, model, target) |> summarise(n = n(), acc = r1(mean(correct)),
    neu_as_neg = r1(mean(pred[gold == "neutral"] == "negative")), neu_ok = r1(mean(correct[gold == "neutral"])),
    pos_ok = r1(mean(correct[gold == "positive"]))) |>
  pivot_wider(names_from = target, values_from = c(acc, neu_as_neg, neu_ok, pos_ok)) |> arrange(topic, model), n = Inf)

cat("\n## 4. Irony cost: accuracy on non-ironic negatives minus ironic (all ironic gold = ?), v1/none\n")
print(count(d |> filter(model == "laya", variant == "v1", prompt == "orig", target == "topic"), set, gold))
print(d |> filter(main) |> group_by(prompt, model, target) |>
  summarise(neg_nonir = r1(mean(correct[set == "nonir" & gold == "negative"])),
            acc_iron = r1(mean(correct[set == "iron"])), neg_iron = r1(mean(correct[set == "iron" & gold == "negative"]))) |>
  mutate(cost_neg = neg_nonir - neg_iron) |> arrange(prompt, model, target), n = Inf)

cat("\n## 5. Negative bias in T1 (% 'negative' answers on non-ironic gold neutral/positive) vs T3, v1/none\n")
print(d |> filter(main, set == "nonir", gold != "negative") |> group_by(prompt, model, target) |>
  summarise(n = n(), neg_rate = r1(mean(pred == "negative"))) |> pivot_wider(names_from = target, values_from = c(n, neg_rate)), n = Inf)

cat("\n## 6. T1 negative recall vs T3 accuracy, Spearman over model x dataset (as followup B7: v1, four LLM/System One models)\n")
fu <- read_csv(a[2], show_col_types = FALSE) |>
  filter(cond == "k0", variant == "v1", model %in% c("qwen3.5-4b", "mistral-7b", "kev-4b", "laya"),
         dataset %in% c("mustardpp", "filatova"))
oth <- fu |> group_by(model, dataset, prompt) |>
  summarise(r1n = mean(correct[sarcastic == 0 & gold == "negative"]), a3 = mean(correct[sarcastic == 1]))
sn <- d |> filter(variant == "v1", model %in% c("qwen3.5-4b", "mistral-7b", "kev-4b", "laya")) |>
  group_by(model, target, prompt) |>
  summarise(r1n = mean(correct[sarcastic == 0 & gold == "negative"]), a3 = mean(correct[sarcastic == 1])) |>
  mutate(dataset = "sarcasm_nlp")
for (pp in c("orig", "swap")) for (tg in c("topic", "nuclear")) {
  x <- bind_rows(filter(oth, prompt == pp), filter(sn, prompt == pp, target == tg))
  cat(pp, tg, "n =", nrow(x), "spearman =", round(cor(x$r1n, x$a3, method = "spearman"), 2), "\n")
}
print(sn |> mutate(across(c(r1n, a3), r1)) |> arrange(prompt, model, target), n = Inf)

cat("\n## 7. Examples: non-ironic items whose answer changes for most models (orig, v1/none)\n")
ch <- w |> filter(prompt == "orig", set == "nonir") |> group_by(id, gold) |>
  summarise(fixed = sum(correct_nuclear > correct_topic), broken = sum(correct_nuclear < correct_topic)) |> ungroup()
print(ch |> summarise(items = n(), any_fixed = sum(fixed > 0), any_broken = sum(broken > 0)))
print(ch |> arrange(desc(fixed - broken)) |> head(8))
print(ch |> arrange(desc(broken - fixed)) |> head(5))

cat("\n## 8. Gold by irony x dataset topic column (the two subsets follow different rules?)\n")
print(d |> filter(model == "twitter-roberta", prompt == "orig", target == "topic", topic %in% c("Energy", "Politics", "General")) |>
  group_by(set, topic) |> summarise(n = n(), neg = r1(mean(gold == "negative")), neu = r1(mean(gold == "neutral")),
                                    pos = r1(mean(gold == "positive"))), n = Inf)

cat("\n## 9. Energy-topic subset only (both readings coincide best): accuracy, mean over orders (v1-v3) or none\n")
print(d |> filter(topic == "Energy") |> group_by(prompt, model, target, set) |> summarise(acc = r1(mean(correct))) |>
  pivot_wider(names_from = c(set, target), values_from = acc) |>
  mutate(cost_topic = nonir_topic - iron_topic, cost_nuclear = nonir_nuclear - iron_nuclear) |> arrange(prompt, model), n = Inf)

cat("\n## 10. Ironic items answered 'neutral' with the nuclear target, by topic (orig, mean over orders)\n")
print(d |> filter(set == "iron", prompt == "orig", topic %in% c("Energy", "Politics", "General")) |>
  rename(tg = target) |> group_by(model, topic, tg) |> summarise(neu = r1(mean(pred == "neutral"))) |>
  pivot_wider(names_from = tg, values_from = neu, names_prefix = "neu_") |> arrange(model, topic), n = Inf)
cat("N ironic items: Energy 757, General 338, Politics 1460; N non-ironic Energy 1012\n")
