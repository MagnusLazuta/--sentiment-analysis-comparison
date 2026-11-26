# Follow-up experiments of 2026-11-21 (make_extra.py, make_swap.py).
# Usage (from experimento/analysis): Rscript followup.R <followup.csv> > followup-<date>.txt
# Input: table from followup_dados.py. MT-CSD differences use a paired thread-cluster
# bootstrap (13 threads, 2,000 resamples, seed 5); other datasets use exact McNemar.
# A. Context composition (MT-CSD, 785 items with K >= 2): k0, k1, kK, oldest, noparent.
# B. Option descriptions: prompt "orig" (System One with descriptions, LLMs without) vs
#    "swap" (LLMs with descriptions, System One without), every condition.
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(readr) })
options(width = 220, dplyr.summarise.inform = FALSE)
f <- read_csv(commandArgs(TRUE)[1], show_col_types = FALSE)
thr <- read_csv("mtcsd_threads-20261117-181106-838446.csv", show_col_types = FALSE) |> select(id, thread)
r1 <- function(x) round(100 * x, 1)
mc <- function(x, y) { ok <- !is.na(x) & !is.na(y); x <- x[ok]; y <- y[ok]; b <- sum(x == 1 & y == 0); c <- sum(x == 0 & y == 1); if (b + c == 0) 1 else binom.test(b, b + c)$p.value }
cb <- function(x, y, th, B = 2000) {  # paired thread-cluster bootstrap CI of mean(y - x), in points
  ok <- !is.na(x) & !is.na(y); x <- x[ok]; y <- y[ok]; th <- th[ok]
  set.seed(5); g <- split(seq_along(x), th)
  bs <- replicate(B, { i <- unlist(g[sample(names(g), replace = TRUE)], use.names = FALSE); mean(y[i] - x[i]) })
  q <- quantile(bs, c(.025, .975)); sprintf("%+.1f [%.1f; %.1f]%s", 100 * mean(y - x), 100 * q[1], 100 * q[2],
                                             ifelse(q[1] > 0 | q[2] < 0, " *", ""))
}
main <- c("v1", "none")
fam <- c("twitter-roberta" = "BERT", "deberta-nli" = "BERT", "qwen3.5-4b" = "LLM", "mistral-7b" = "LLM",
         "kev-4b" = "System One", "laya" = "System One")

cat("## 0. Determinism check: classifier / NLI swap copies vs main run (must be identical)\n")
print(f |> filter(model %in% c("twitter-roberta", "deberta-nli")) |>
  select(model, id, variant, prompt, cond, p_negative) |>
  pivot_wider(names_from = prompt, values_from = p_negative) |> filter(!is.na(swap), !is.na(orig)) |>
  group_by(model) |> summarise(n = n(), max_abs_diff = max(abs(swap - orig))))

# ---------------------------------------------------------------- A
cat("\n## A1. MT-CSD context composition, items with K >= 2 (v1/none, prompt orig)\n")
a <- f |> filter(dataset == "mtcsd", prompt == "orig", variant %in% main, K >= 2) |>
  mutate(cond = ifelse(cond == paste0("k", K), "kK", cond)) |>
  filter(cond %in% c("k0", "k1", "kK", "oldest", "noparent")) |>
  select(model, id, gold, cond, correct, pred) |> inner_join(thr, by = "id")
aw <- a |> select(-pred) |> pivot_wider(names_from = cond, values_from = correct)
print(aw |> group_by(model) |> summarise(n = n(), across(c(k0, k1, kK, oldest, noparent), ~ r1(mean(.x)))), n = Inf)
cat("\nPaired differences, thread-cluster bootstrap (* = CI excludes 0):\n")
print(aw |> group_by(model) |> summarise(
  `k0->oldest` = cb(k0, oldest, thread), `k0->noparent` = cb(k0, noparent, thread),
  `noparent->kK (add parent)` = cb(noparent, kK, thread), `k1->kK` = cb(k1, kK, thread)), n = Inf)
cat("\nBy gold class (k0 / k1 / kK / oldest / noparent):\n")
print(aw |> group_by(model, gold) |> summarise(n = n(), across(c(k0, k1, kK, oldest, noparent), ~ r1(mean(.x)))) |>
  arrange(gold, model), n = Inf)
cat("\nNeutral items: add parent (noparent -> kK) and oldest only, thread bootstrap:\n")
print(aw |> filter(gold == "neutral") |> group_by(model) |>
  summarise(n = n(), `k0->oldest` = cb(k0, oldest, thread), `noparent->kK` = cb(noparent, kK, thread)), n = Inf)

cat("\n## A2. Parent contagion: agreement with the model's own no-context answer to the parent comment\n")
par <- read_csv("mtcsd_parent-20261117-181106-838446.csv", show_col_types = FALSE)  # id, parent_id, parent_gold
pa <- f |> filter(dataset == "mtcsd", prompt == "orig", variant %in% main, cond == "k0") |>
  select(model, parent_id = id, parent_pred = pred)
ag <- a |> inner_join(par, by = "id") |> inner_join(pa, by = c("model", "parent_id")) |>
  mutate(agree = as.integer(pred == parent_pred), same_gold = gold == parent_gold)
print(ag |> group_by(model, cond) |> summarise(n = n(), agree = r1(mean(agree))) |>
  pivot_wider(names_from = cond, values_from = agree), n = Inf)
cat("\nAccuracy when the item's gold differs from the parent's gold (k0 / k1 / kK / oldest / noparent):\n")
print(ag |> select(model, id, thread, same_gold, cond, correct) |> pivot_wider(names_from = cond, values_from = correct) |>
  group_by(model, same_gold) |> summarise(n = n(), `k0->k1` = cb(k0, k1, thread), `noparent->kK` = cb(noparent, kK, thread),
                                          across(c(k0, k1, kK, oldest, noparent), ~ r1(mean(.x)))), n = Inf)

# ---------------------------------------------------------------- B
cat("\n## B1. No context: accuracy and neutral share, orig vs swap (v1)\n")
b0 <- f |> filter(cond == "k0", variant == "v1", model %in% c("qwen3.5-4b", "mistral-7b", "kev-4b", "laya")) |>
  mutate(grp = ifelse(sarcastic == 1, "irônico", "não irônico"))
bw <- b0 |> select(model, dataset, grp, id, prompt, correct, pred) |>
  pivot_wider(names_from = prompt, values_from = c(correct, pred))
print(bw |> group_by(dataset, grp, model) |>
  summarise(n = n(), acc_orig = r1(mean(correct_orig)), acc_swap = r1(mean(correct_swap)),
            p = signif(mc(correct_orig, correct_swap), 2),
            neu_orig = r1(mean(pred_orig == "neutral")), neu_swap = r1(mean(pred_swap == "neutral")),
            neg_orig = r1(mean(pred_orig == "negative")), neg_swap = r1(mean(pred_swap == "negative")),
            agree = r1(mean(pred_orig == pred_swap))) |>
  arrange(dataset, grp, model), n = Inf)
cat("\nMT-CSD rows above use McNemar; thread bootstrap for MT-CSD orig -> swap:\n")
print(bw |> filter(dataset == "mtcsd") |> inner_join(thr, by = "id") |> group_by(model) |>
  summarise(`orig->swap` = cb(correct_orig, correct_swap, thread)))

cat("\n## B2. Same information for every family: accuracy (v1/none, no context, non-ironic and ironic)\n")
cat("with_desc = LLM swap + System One orig; no_desc = LLM orig + System One swap; BERTs unchanged\n")
both <- f |> filter(cond == "k0", variant %in% main) |>
  mutate(setting = case_when(fam[model] == "LLM" & prompt == "swap" ~ "with_desc",
                             fam[model] == "LLM" & prompt == "orig" ~ "no_desc",
                             fam[model] == "System One" & prompt == "orig" ~ "with_desc",
                             fam[model] == "System One" & prompt == "swap" ~ "no_desc",
                             prompt == "orig" ~ "bert")) |> filter(!is.na(setting)) |>
  mutate(grp = ifelse(sarcastic == 1, "irôn", "não irôn"))
print(both |> group_by(dataset, grp, setting, model) |> summarise(acc = r1(mean(correct))) |>
  pivot_wider(names_from = model, values_from = acc) |> arrange(dataset, grp, setting), n = Inf)

cat("\n## B3. Position bias with and without descriptions: neg / neu / pos share by option order (no context, non-ironic)\n")
print(f |> filter(cond == "k0", sarcastic == 0, model %in% c("qwen3.5-4b", "mistral-7b", "kev-4b", "laya")) |>
  group_by(model, prompt, variant) |>
  summarise(n = n(), acc = r1(mean(correct)), neg = r1(mean(pred == "negative")), neu = r1(mean(pred == "neutral")),
            pos = r1(mean(pred == "positive"))), n = Inf)
cat("\nSame answer in all three orders (no context, all items):\n")
print(f |> filter(cond == "k0", model %in% c("qwen3.5-4b", "mistral-7b", "kev-4b", "laya")) |>
  select(model, prompt, id, variant, pred) |> pivot_wider(names_from = variant, values_from = pred) |>
  group_by(model, prompt) |> summarise(all3 = r1(mean(v1 == v2 & v2 == v3))))

cat("\n## B4. Context effect with swapped descriptions (v1): k0 -> k1 and k0 -> kK\n")
cx <- f |> filter(variant == "v1", dataset %in% c("mustardpp", "mtcsd"), model %in% c("qwen3.5-4b", "mistral-7b", "kev-4b", "laya")) |>
  mutate(cond = ifelse(cond == paste0("k", K), "kK", cond)) |> filter(cond %in% c("k0", "k1", "kK")) |>
  mutate(set = ifelse(sarcastic == 1, "MUStARD++ irôn", dataset)) |>
  select(model, set, id, prompt, cond, correct) |> pivot_wider(names_from = cond, values_from = correct) |>
  mutate(k1 = coalesce(k1, kK)) |>  # K = 1: the previous turn is the whole context
  left_join(thr, by = "id")
print(cx |> group_by(set, model, prompt) |>
  summarise(n = n(),
            d1 = if (set[1] == "mtcsd") cb(k0, k1, thread) else sprintf("%+.1f p=%.2g", 100 * mean(k1 - k0), mc(k0, k1)),
            dK = if (set[1] == "mtcsd") cb(k0, kK, thread) else sprintf("%+.1f p=%.2g", 100 * mean(kK - k0), mc(k0, kK)),
            k0 = r1(mean(k0)), k1 = r1(mean(k1, na.rm = TRUE)), kK = r1(mean(kK))) |>
  arrange(set, model, prompt), n = Inf)

cat("\n## B5. Per-class context effect with swapped descriptions, MT-CSD (v1, k0 -> kK)\n")
print(f |> filter(variant == "v1", dataset == "mtcsd", model %in% c("qwen3.5-4b", "mistral-7b", "kev-4b", "laya")) |>
  mutate(cond = ifelse(cond == paste0("k", K), "kK", cond)) |> filter(cond %in% c("k0", "kK")) |>
  select(model, gold, id, prompt, cond, correct) |> pivot_wider(names_from = cond, values_from = correct) |>
  inner_join(thr, by = "id") |> group_by(model, prompt, gold) |>
  summarise(n = n(), d = cb(k0, kK, thread), k0 = r1(mean(k0)), kK = r1(mean(kK))) |> arrange(gold, model, prompt), n = Inf)

cat("\n## B6. Ablation cohort K >= 4 with swapped descriptions (v1): accuracy by k\n")
print(f |> filter(variant == "v1", K >= 4, model %in% c("qwen3.5-4b", "mistral-7b", "kev-4b", "laya"),
                  cond %in% paste0("k", 0:4)) |>
  mutate(set = ifelse(sarcastic == 1, "MUStARD++ irôn", dataset)) |>
  group_by(set, model, prompt, cond) |> summarise(acc = r1(mean(correct))) |>
  pivot_wider(names_from = cond, values_from = acc) |> arrange(set, model, prompt), n = Inf)

cat("\n## B7. T1 negative recall vs T3 accuracy under swap (Spearman over model x dataset)\n")
nb <- b0 |> filter(dataset != "mtcsd") |> group_by(model, dataset, prompt) |>
  summarise(r1n = mean(correct[sarcastic == 0 & gold == "negative"]), a3 = mean(correct[sarcastic == 1]))
print(nb |> mutate(across(c(r1n, a3), r1)) |> arrange(prompt, dataset, model), n = Inf)
for (pp in c("orig", "swap")) { x <- filter(nb, prompt == pp); cat(pp, "spearman =", round(cor(x$r1n, x$a3, method = "spearman"), 2), "\n") }
