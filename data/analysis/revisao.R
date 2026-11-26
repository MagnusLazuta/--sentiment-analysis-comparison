# Corrected analyses after the 2026-11-21 review (no new inference needed).
# Usage (from experimento/analysis): Rscript revisao.R > revisao-20261121.txt
#  1. MT-CSD: cluster bootstrap by thread (13 threads), since items of a thread share context.
#  2. Filatova: accuracy on the reviews no model truncated (over_budget) vs all.
#  3. Subgroups controlled for the gold class mix (per-class accuracy and class-standardised accuracy).
#  4. Confidence as an error filter (AUROC), complementing confident-error rates.
# Inputs: main run 20261117-181106-838446, ablation run 20261118-094702-838561,
# mtcsd_threads-20261117-181106-838446.csv (threads.py). Seed 5; 2,000 resamples.
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(readr) })
options(width = 220, dplyr.summarise.inform = FALSE, pillar.sigfig = 4)
set.seed(5)
B <- 2000
R <- "../results/20261117-181106-838446/eval"
thr <- read_csv("mtcsd_threads-20261117-181106-838446.csv", show_col_types = FALSE) |> select(id, thread)
abl <- read_csv("../results/20261118-094702-838561/eval/ablation.csv", show_col_types = FALSE) |>
  filter(model != "nanojev")
pr <- read_csv(file.path(R, "predictions.csv"), show_col_types = FALSE,
  col_select = c(model, dataset, id, subgroup, sarcastic, gold, condition, variant, pred, correct, confidence, over_budget)) |>
  filter(model != "nanojev")
r1 <- function(x) round(100 * x, 1)

# paired cluster bootstrap of mean(y) - mean(x); d has columns thread, x, y
cboot <- function(d) {
  g <- split(seq_len(nrow(d)), d$thread)
  est <- mean(d$y) - mean(d$x)
  bs <- replicate(B, { i <- unlist(g[sample(names(g), replace = TRUE)], use.names = FALSE); mean(d$y[i]) - mean(d$x[i]) })
  q <- quantile(bs, c(.025, .975))
  tibble(n = nrow(d), diff = r1(est), lo = r1(q[1]), hi = r1(q[2]), sig = q[1] > 0 | q[2] < 0,
         threads_up = sum(tapply(d$y - d$x, d$thread, mean) > 0), threads_down = sum(tapply(d$y - d$x, d$thread, mean) < 0))
}
cboot_acc <- function(d) {  # cluster bootstrap CI of an accuracy; d has thread, y
  g <- split(seq_len(nrow(d)), d$thread)
  bs <- replicate(B, { i <- unlist(g[sample(names(g), replace = TRUE)], use.names = FALSE); mean(d$y[i]) })
  q <- quantile(bs, c(.025, .975)); tibble(n = nrow(d), acc = r1(mean(d$y)), lo = r1(q[1]), hi = r1(q[2]))
}

cat("## 1a. MT-CSD accuracy with thread-cluster CI (v1/none)\n")
m <- abl |> filter(dataset == "mtcsd", variant %in% c("v1", "none")) |> inner_join(thr, by = "id")
w <- m |> group_by(model, variant, id, thread, gold) |>
  summarise(k0 = correct[k == 0], k1 = correct[k == 1], kK = correct[k == K], K = K[1], .groups = "drop")
print(bind_rows(
  w |> group_by(model) |> group_modify(~ cboot_acc(transmute(.x, thread, y = k0))) |> mutate(cond = "k0"),
  w |> group_by(model) |> group_modify(~ cboot_acc(transmute(.x, thread, y = k1))) |> mutate(cond = "k1"),
  w |> group_by(model) |> group_modify(~ cboot_acc(transmute(.x, thread, y = kK))) |> mutate(cond = "kK")) |>
  arrange(model, cond), n = Inf)

cat("\n## 1b. MT-CSD paired differences, thread-cluster bootstrap, all option orders\n")
wa <- abl |> filter(dataset == "mtcsd") |> inner_join(thr, by = "id") |>
  group_by(model, variant, id, thread, gold) |>
  summarise(k0 = correct[k == 0], k1 = correct[k == 1], kK = correct[k == K], .groups = "drop")
print(bind_rows(
  wa |> group_by(model, variant) |> group_modify(~ cboot(transmute(.x, thread, x = k0, y = k1))) |> mutate(cmp = "k0->k1"),
  wa |> group_by(model, variant) |> group_modify(~ cboot(transmute(.x, thread, x = k0, y = kK))) |> mutate(cmp = "k0->kK"),
  wa |> group_by(model, variant) |> group_modify(~ cboot(transmute(.x, thread, x = k1, y = kK))) |> mutate(cmp = "k1->kK")) |>
  arrange(cmp, model, variant), n = Inf)

cat("\n## 1c. MT-CSD per gold class k0 -> kK, thread-cluster bootstrap (v1/none)\n")
print(wa |> filter(variant %in% c("v1", "none")) |> group_by(model, gold) |>
  group_modify(~ cboot(transmute(.x, thread, x = k0, y = kK))) |> arrange(gold, model), n = Inf)

cat("\n## 1d. MT-CSD cohort K>=4 k0 -> k4 and k0 -> k1, thread-cluster bootstrap (v1/none)\n")
co <- abl |> filter(dataset == "mtcsd", variant %in% c("v1", "none"), K >= 4, k %in% c(0, 1, 4)) |>
  inner_join(thr, by = "id") |> select(model, id, thread, k, correct) |>
  pivot_wider(names_from = k, values_from = correct, names_prefix = "k")
print(bind_rows(co |> group_by(model) |> group_modify(~ cboot(transmute(.x, thread, x = k0, y = k1))) |> mutate(cmp = "k0->k1"),
                co |> group_by(model) |> group_modify(~ cboot(transmute(.x, thread, x = k0, y = k4))) |> mutate(cmp = "k0->k4")) |>
  arrange(cmp, model), n = Inf)

cat("\n## 1e. MT-CSD T1 accuracy minus majority (58.6), thread-cluster CI (v1/none)\n")
print(pr |> filter(dataset == "mtcsd", condition == "no_context", variant %in% c("v1", "none")) |>
  inner_join(thr, by = "id") |> group_by(model) |> group_modify(~ cboot_acc(transmute(.x, thread, y = correct))), n = Inf)

cat("\n## 2. Filatova: reviews truncated by any model vs not (no_context)\n")
fil <- pr |> filter(dataset == "filatova", condition == "no_context")
trunc_ids <- fil |> filter(over_budget %in% c(TRUE, "True")) |> distinct(id) |> pull(id)
cat("reviews truncated by at least one model:", length(trunc_ids), "of", n_distinct(fil$id), "\n")
print(fil |> filter(over_budget %in% c(TRUE, "True"), variant %in% c("v1", "none")) |> count(model, sarcastic))
print(fil |> filter(variant %in% c("v1", "none")) |> mutate(sub = ifelse(id %in% trunc_ids, "truncated_any", "fits_all")) |>
  group_by(model, sarcastic, sub) |> summarise(n = n(), acc = r1(mean(correct))) |>
  pivot_wider(names_from = sub, values_from = c(n, acc)), n = Inf)
print(fil |> filter(variant %in% c("v1", "none"), !id %in% trunc_ids) |> count(sarcastic, gold))

cat("\n## 3. Subgroups: per-class accuracy and class-standardised accuracy (no_context, non-ironic, v1/none)\n")
sg <- pr |> filter(condition == "no_context", variant %in% c("v1", "none"), sarcastic == 0,
                   dataset %in% c("mustardpp", "sarcasm_nlp", "mtcsd")) |>
  filter(!subgroup %in% c("GOLDENGIRLS", "Non-Sarcastic", "unknown"))
mix <- sg |> distinct(dataset, id, gold) |> count(dataset, gold) |> group_by(dataset) |> mutate(w = n / sum(n)) |> select(-n)
pc <- sg |> group_by(dataset, subgroup, model, gold) |> summarise(n = n(), acc = mean(correct), .groups = "drop")
print(pc |> left_join(mix, by = c("dataset", "gold")) |> group_by(dataset, subgroup, model) |>
  summarise(raw_n = sum(n), std_acc = r1(sum(acc * w)), .groups = "drop") |>
  pivot_wider(names_from = model, values_from = std_acc), n = Inf)
print(sg |> distinct(dataset, subgroup, id, gold) |> count(dataset, subgroup, gold) |> group_by(dataset, subgroup) |>
  mutate(p = r1(n / sum(n))) |> select(-n) |> pivot_wider(names_from = gold, values_from = p), n = Inf)

cat("\n## 4. AUROC of confidence for correct vs wrong (no_context, v1/none)\n")
auroc <- function(conf, ok) { r <- rank(conf); n1 <- sum(ok); n0 <- sum(!ok)
  if (n1 == 0 || n0 == 0) return(NA); (sum(r[ok]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
print(pr |> filter(condition == "no_context", variant %in% c("v1", "none")) |>
  mutate(grp = ifelse(sarcastic == 1, "irônico", "não irônico")) |>
  group_by(dataset, grp, model) |> summarise(n = n(), auroc = round(auroc(confidence, correct == 1), 3),
                                             conf_ge_0.9 = r1(mean(confidence >= 0.9)), .groups = "drop") |>
  pivot_wider(names_from = model, values_from = c(auroc)), n = Inf)
