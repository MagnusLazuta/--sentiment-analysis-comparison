# 95% CIs for the fixed MT-CSD cohort used in the paper (2026-11-26).
#
# Source: context ablation run 20261118-094702-838561 (eval/ablation.csv).
# Cohort: MT-CSD items with >= 4 context turns (n = 456, 13 threads), k = 0..4, same items at
# every k; all option orders pooled (as in turnos_mtcsd_slides.R); NanoJev excluded.
# Paired difference in accuracy to k = 0 (same model, order and item), cluster bootstrap by
# thread (same method as turnos_todos.R): 2,000 resamples, seed 5.
# Output (this folder): coorte_ic-20261126.txt
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr)})
set.seed(5)
B <- 2000
ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
thr <- read_csv("mtcsd_threads-20261117-181106-838446.csv", show_col_types = FALSE) |> select(id, thread)
d <- read_csv("../results/20261118-094702-838561/eval/ablation.csv", show_col_types = FALSE) |>
  filter(dataset == "mtcsd", model %in% ord) |>
  group_by(model, variant, id) |> filter(max(k) >= 4) |> ungroup() |> filter(k <= 4) |>
  left_join(thr, by = "id")
stopifnot(!any(is.na(d$thread)))
base <- d |> filter(k == 0) |> select(model, variant, id, y0 = correct)
dd <- d |> inner_join(base, by = c("model", "variant", "id"))

boot_diff <- function(x) {  # cluster bootstrap of mean(y - y0) by thread
  g <- split(x$y - x$y0, x$thread)
  m <- replicate(B, { s <- sample(length(g), replace = TRUE); mean(unlist(g[s])) })
  tibble(dif = 100 * mean(x$y - x$y0), lo = 100 * quantile(m, 0.025), hi = 100 * quantile(m, 0.975))
}
res <- dd |> filter(k %in% c(1, 4)) |> group_by(model, k) |>
  group_modify(~ boot_diff(transmute(.x, thread, y = correct, y0))) |> ungroup() |>
  mutate(model = factor(model, ord)) |> arrange(model, k)

sink("coorte_ic-20261126.txt")
cat(sprintf("MT-CSD fixed cohort: %d items, %d threads; orders pooled.\n",
            n_distinct(d$id), n_distinct(d$thread)))
cat("Paired accuracy difference to k = 0 (pp) [95% CI, cluster bootstrap by thread, B = 2000, seed 5].\n\n")
res |> mutate(cell = sprintf("%+.1f [%.1f; %.1f]", dif, lo, hi)) |> select(model, k, cell) |>
  pivot_wider(names_from = k, values_from = cell, names_prefix = "k") |>
  as.data.frame() |> print(row.names = FALSE)
sink()
