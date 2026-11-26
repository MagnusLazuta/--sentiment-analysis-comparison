# Context-length ablation analyses. Every comparison uses the same items at every k.
# Usage (from experimento/analysis): Rscript ablacao.R <ablation RUN_ID>
# Input: results/<RUN_ID>/eval/ablation.csv (ablacao_dados.py). k = number of last context
# turns shown (0 = no context, K = whole context). Main prompt order (v1 / none) unless noted.
# Output: figures/ablacao-<name>-<RUN_ID>.pdf and tables on stdout.
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(readr); library(ggplot2) })
run_id <- commandArgs(TRUE)[1]
set.seed(5)
lab <- c("twitter-roberta" = "twitter-roberta", "deberta-nli" = "DeBERTa-NLI", "qwen3.5-4b" = "Qwen3.5-4B",
         "mistral-7b" = "Mistral-7B", "kev-4b" = "Kev-4B", "laya" = "Laya")
fam <- c("twitter-roberta" = "BERT", "deberta-nli" = "BERT", "qwen3.5-4b" = "LLM", "mistral-7b" = "LLM",
         "kev-4b" = "System One", "laya" = "System One")
cols <- c("twitter-roberta" = "#6baed6", "DeBERTa-NLI" = "#08519c", "Qwen3.5-4B" = "#9e9ac8",
          "Mistral-7B" = "#54278f", "Kev-4B" = "#fd8d3c", "Laya" = "#a63603")
ds_lab <- c(mustardpp = "MUStARD++", mtcsd = "MT-CSD")
gold_lab <- c(negative = "Negativo / contra", neutral = "Neutro / nenhuma", positive = "Positivo / a favor")
C <- 4  # cohort: items with at least C context turns, shown with k = 0..C

all <- read_csv(file.path("../results", run_id, "eval", "ablation.csv"), show_col_types = FALSE) |>
  filter(model %in% names(lab)) |>
  mutate(m = factor(lab[model], lab), f = factor(fam[model], c("BERT", "LLM", "System One")),
         ds = factor(ds_lab[dataset], ds_lab))
a <- filter(all, variant %in% c("v1", "none"))
dir.create("figures", showWarnings = FALSE)
save <- function(p, name, w, h) {
  out <- file.path("figures", paste0("ablacao-", name, "-", run_id, ".pdf")); ggsave(out, p, width = w, height = h)
  cat("wrote", out, "\n")
}
boot_ci <- function(x, B = 1000) {
  q <- quantile(replicate(B, mean(sample(x, replace = TRUE))), c(.025, .975)); 100 * q
}
mcnemar_p <- function(x, y) {  # exact, paired 0/1 vectors
  b <- sum(x == 1 & y == 0); c <- sum(x == 0 & y == 1)
  if (b + c == 0) 1 else binom.test(b, b + c)$p.value
}
bars <- function(d, xvar, facets, ylab) {
  ggplot(d, aes(x = {{ xvar }}, y = acc, fill = m)) +
    geom_col(position = position_dodge(0.85), width = 0.8) +
    facets + scale_fill_manual(values = cols, name = NULL) +
    scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.02))) +
    labs(x = "Turnos de contexto mostrados (k)", y = ylab) +
    guides(fill = guide_legend(nrow = 1)) + theme_bw(base_size = 22) +
    theme(legend.position = "top", panel.grid.minor = element_blank())
}

# ---- 0. sample sizes -------------------------------------------------------------
cat("Items (non-ironic / ironic) by dataset and K:\n")
print(a |> distinct(ds, sarcastic, id, K) |> mutate(K = pmin(K, 7)) |> count(ds, sarcastic, K) |>
        pivot_wider(names_from = K, values_from = n, names_prefix = "K="), n = Inf)

# ---- A. fixed cohort: K >= C, k = 0..C ---------------------------------------------
coh <- a |> filter(sarcastic == 0, K >= C, k <= C)
cat("\nA. Cohort K >=", C, "(non-ironic), N items:\n")
print(coh |> distinct(ds, id) |> count(ds))
A <- coh |> group_by(ds, f, m, k) |> summarise(acc = 100 * mean(correct), n = n(), .groups = "drop")
print(A |> select(-n) |> pivot_wider(names_from = k, values_from = acc, names_prefix = "k=") |>
        mutate(across(starts_with("k="), \(x) round(x, 1))), n = Inf)
save(bars(A, factor(k), facet_grid(ds ~ f), "Acurácia (%)"), "coorte", 16, 10)

# ---- B. all items: k = 0, 1, K -------------------------------------------------------
B <- a |> filter(sarcastic == 0) |> group_by(model, id) |> filter(k %in% c(0, 1, max(k))) |>
  mutate(cond = ifelse(k == 0, "0", ifelse(k == 1 & K > 1, "1", ifelse(k == K, "K", NA)))) |>
  ungroup() |> filter(!is.na(cond))
# items with K = 1 have k = 1 = K; count them in both "1" and "K"
B <- bind_rows(B, a |> filter(sarcastic == 0, K == 1, k == 1) |> mutate(cond = "1"))
cat("\nB. All non-ironic items with context: accuracy at k = 0, 1, K (95% bootstrap CI), McNemar p\n")
Bt <- B |> group_by(ds, f, m, cond) |>
  summarise(acc = 100 * mean(correct), lo = boot_ci(correct)[1], hi = boot_ci(correct)[2], n = n(), .groups = "drop")
Bw <- B |> select(ds, m, id, cond, correct) |> pivot_wider(names_from = cond, values_from = correct) |>
  group_by(ds, m) |> summarise(p_0vs1 = mcnemar_p(`0`, `1`), p_1vsK = mcnemar_p(`1`, K), p_0vsK = mcnemar_p(`0`, K),
                               .groups = "drop")
print(Bt |> mutate(across(c(acc, lo, hi), \(x) round(x, 1))) |>
        transmute(ds, m, cond, n, acc = paste0(acc, " [", lo, ", ", hi, "]")) |>
        pivot_wider(names_from = cond, values_from = acc) |> left_join(Bw, by = c("ds", "m")) |>
        mutate(across(starts_with("p_"), \(x) signif(x, 2))), n = Inf, width = Inf)
pB <- ggplot(Bt, aes(x = factor(cond, c("0", "1", "K"), c("0", "1", "Todos")), y = acc, fill = m)) +
  geom_col(position = position_dodge(0.85), width = 0.8) +
  geom_errorbar(aes(ymin = lo, ymax = hi), position = position_dodge(0.85), width = 0.3) +
  facet_grid(ds ~ f) + scale_fill_manual(values = cols, name = NULL) +
  scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.02))) +
  labs(x = "Turnos de contexto mostrados", y = "Acurácia (%)") + guides(fill = guide_legend(nrow = 1)) +
  theme_bw(base_size = 22) + theme(legend.position = "top", panel.grid.minor = element_blank())
save(pB, "0-1-todos", 16, 10)

# ---- C. per gold class, cohort --------------------------------------------------------
Cc <- coh |> mutate(g = factor(gold_lab[gold], gold_lab)) |> group_by(ds, g, f, m, k) |>
  summarise(acc = 100 * mean(correct), n = n(), .groups = "drop")
cat("\nC. Cohort, accuracy per gold class (N items per class):\n")
print(coh |> distinct(ds, gold, id) |> count(ds, gold))
save(bars(Cc, factor(k), facet_grid(ds + g ~ f), "Acurácia na classe (%)"), "classe", 18, 24)

# ---- D. what the models answer: share of each predicted label, cohort ------------------
D <- coh |> count(ds, f, m, k, pred) |> group_by(ds, m, k) |> mutate(share = 100 * n / sum(n)) |> ungroup() |>
  mutate(pred = factor(gold_lab[pred], gold_lab))
gold_share <- coh |> distinct(ds, id, gold) |> count(ds, gold) |> group_by(ds) |>
  mutate(share = round(100 * n / sum(n), 1))
cat("\nD. Gold label shares in the cohort:\n"); print(gold_share)
pD <- ggplot(D, aes(x = factor(k), y = share, fill = pred)) + geom_col(width = 0.8) +
  facet_grid(ds ~ m) +
  scale_fill_manual(values = c("#d6604d", "#bababa", "#4393c3"), name = "Resposta") +
  labs(x = "Turnos de contexto mostrados (k)", y = "Respostas (%)") + theme_bw(base_size = 20) +
  theme(legend.position = "top", panel.grid.minor = element_blank())
save(pD, "respostas", 22, 9)

# ---- E. flips per added turn: k-1 -> k, cohort ------------------------------------------
E <- coh |> arrange(model, id, k) |> group_by(model, id) |>
  mutate(prev = lag(correct)) |> ungroup() |> filter(k > 0) |>
  group_by(ds, f, m, k) |>
  summarise(Ajudou = 100 * mean(prev == 0 & correct == 1), Atrapalhou = -100 * mean(prev == 1 & correct == 0),
            .groups = "drop") |>
  pivot_longer(c(Ajudou, Atrapalhou), names_to = "efeito", values_to = "pct")
pE <- ggplot(E, aes(x = factor(k), y = pct, fill = m, alpha = efeito)) +
  geom_col(position = position_dodge(0.85), width = 0.8) + geom_hline(yintercept = 0) +
  facet_grid(ds ~ f) + scale_fill_manual(values = cols, name = NULL) +
  scale_alpha_manual(values = c(Ajudou = 1, Atrapalhou = 0.45), name = NULL) +
  labs(x = "Turno acrescentado (k - 1 para k)", y = "Itens que mudaram (%)") +
  guides(fill = guide_legend(nrow = 1, order = 1), alpha = guide_legend(nrow = 1, order = 2)) +
  theme_bw(base_size = 22) + theme(legend.position = "top", legend.box = "vertical", panel.grid.minor = element_blank())
save(pE, "viradas", 16, 10)
cat("\nE. Flips per added turn (% of cohort items; + helped, - hurt):\n")
print(E |> mutate(pct = round(pct, 1)) |> pivot_wider(names_from = c(efeito, k), values_from = pct) |> select(-f),
      n = Inf, width = Inf)

# ---- F. ironic MUStARD++ items, cohort ------------------------------------------------------
Fd <- a |> filter(sarcastic == 1, K >= C, k <= C)
cat("\nF. Ironic MUStARD++ cohort, N =", n_distinct(Fd$id), "\n")
Ft <- Fd |> group_by(ds, f, m, k) |> summarise(acc = 100 * mean(correct), .groups = "drop")
print(Ft |> mutate(acc = round(acc, 1)) |> pivot_wider(names_from = k, values_from = acc, names_prefix = "k=") |>
        select(-ds), n = Inf)
save(bars(Ft, factor(k), facet_grid(. ~ f), "Acurácia nos irônicos (%)"), "ironicos", 16, 6)

# ---- G. robustness: same cohort with every option order --------------------------------------
G <- all |> filter(sarcastic == 0, K >= C, k %in% c(0, 1, C)) |> group_by(ds, m, variant, k) |>
  summarise(acc = 100 * mean(correct), .groups = "drop") |>
  pivot_wider(names_from = k, values_from = acc, names_prefix = "k=") |>
  mutate(`d1` = `k=1` - `k=0`, dC = .data[[paste0("k=", C)]] - `k=0`)
cat("\nG. Cohort effect by option order (points vs k = 0):\n")
print(G |> mutate(across(where(is.numeric), \(x) round(x, 1))), n = Inf)
