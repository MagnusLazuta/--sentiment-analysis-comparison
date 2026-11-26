# Turn-by-turn views of MT-CSD for the slides (2026-11-23).
# Source: ablation run 20261118-094702-838561 (eval/ablation.csv). Fixed cohort: MT-CSD items with
# >= 4 context turns (n = 456), k = 0..4, same items at every k. All option orders pooled
# (twitter-roberta and DeBERTa-NLI have one). NanoJev excluded.
# A. accuracy by k, one line per model;  B. Qwen3.5-4B confusion matrix by k (row %), red outline
#    when the row % changed >= 10 points vs the previous k.
# Outputs: turnos_mtcsd_linhas-20261123.{pdf,png,txt}, turnos_mtcsd_confusao_qwen-20261123.{pdf,png}
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
lab <- c("twitter-roberta", "DeBERTa-NLI", "Qwen3.5-4B", "Mistral-7B", "Kev-4B", "Laya")
cols <- c("#7F8C8D", "#C0392B", "#2B4C7E", "#B35806", "#1B7837", "#7D3C98")
d <- read_csv("../results/20261118-094702-838561/eval/ablation.csv", show_col_types = FALSE) |>
  filter(dataset == "mtcsd", model %in% ord) |>
  group_by(model, variant, id) |> filter(max(k) >= 4) |> ungroup() |> filter(k <= 4) |>
  mutate(model = factor(model, ord, lab))
n <- n_distinct(d$id)

a <- d |> group_by(model, k) |> summarise(acc = 100 * mean(correct), .groups = "drop")
sink("turnos_mtcsd_linhas-20261123.txt")
cat(sprintf("MT-CSD cohort >= 4 turns, n = %d items; accuracy (%%) by k, orders pooled.\n", n))
print(a |> mutate(acc = round(acc, 1)) |> pivot_wider(names_from = k, values_from = acc, names_prefix = "k") |>
        mutate(k1_k0 = k1 - k0, k4_k0 = k4 - k0) |> as.data.frame(), row.names = FALSE)
sink()
la <- a |> filter(k == 4) |> arrange(acc) |> mutate(y = acc)
for (i in seq_len(nrow(la))[-1]) la$y[i] <- max(la$y[i], la$y[i - 1] + 2.6)  # keep end labels apart
pA <- ggplot(a, aes(k, acc, colour = model)) +
  geom_line(linewidth = 1.4) + geom_point(size = 3.2) +
  geom_text(data = la, aes(y = y, label = model), hjust = -0.12, size = 5.6, show.legend = FALSE) +
  scale_colour_manual(values = cols, guide = "none") +
  scale_x_continuous(breaks = 0:4, expand = expansion(mult = c(0.03, 0.32))) +
  scale_y_continuous(limits = c(35, 80)) +
  labs(x = "Turnos de contexto (k)", y = "Acurácia (%)",
       caption = sprintf("MT-CSD, mesmos %d itens em todo k (pelo menos 4 turnos de contexto); três ordens de opções juntas.", n)) +
  theme_bw(base_size = 18) + theme(panel.grid.minor = element_blank(), plot.caption = element_text(size = 13))
ggsave("turnos_mtcsd_linhas-20261123.pdf", pA, width = 13, height = 6)
ggsave("turnos_mtcsd_linhas-20261123.png", pA, width = 13, height = 6, dpi = 100)

lv <- c("negative", "neutral", "positive"); pt <- c("negativo", "neutro", "positivo")
q <- d |> filter(model == "Qwen3.5-4B") |>
  mutate(gold = factor(pt[match(gold, lv)], rev(pt)), pred = factor(pt[match(pred, lv)], pt)) |>
  count(k, gold, pred, .drop = FALSE) |> group_by(k, gold) |> mutate(pct = 100 * n / sum(n)) |> ungroup() |>
  group_by(gold, pred) |> arrange(k, .by_group = TRUE) |> mutate(dlt = pct - lag(pct)) |> ungroup()
acc <- d |> filter(model == "Qwen3.5-4B") |> group_by(k) |> summarise(acc = 100 * mean(correct))
q <- q |> left_join(acc, by = "k") |> mutate(kl = factor(sprintf("k = %d  (acc. %s%%)", k, sub(".", ",", sprintf("%.1f", acc), fixed = TRUE))))
pB <- ggplot(q, aes(pred, gold, fill = pct)) +
  geom_tile(colour = "white") +
  geom_tile(data = q |> filter(abs(dlt) >= 10), colour = "#D0021B", fill = NA, linewidth = 1.4) +
  geom_text(aes(label = sprintf("%.0f%%", pct), colour = pct > 55), size = 5.2) +
  facet_wrap(~ kl, nrow = 1) +
  scale_fill_gradient(low = "#F4F6F9", high = "#2B4C7E", limits = c(0, 100), guide = "none") +
  scale_colour_manual(values = c(`FALSE` = "grey15", `TRUE` = "white"), guide = "none") +
  labs(x = "Resposta do Qwen3.5-4B", y = "Ground-truth",
       caption = "% de cada linha (ground-truth); três ordens juntas. Contorno vermelho: mudou 10 pontos ou mais em relação ao turno anterior.") +
  theme_minimal(base_size = 17) +
  theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 30, hjust = 1),
        strip.text = element_text(face = "bold"), plot.caption = element_text(size = 12))
ggsave("turnos_mtcsd_confusao_qwen-20261123.pdf", pB, width = 17, height = 5)
ggsave("turnos_mtcsd_confusao_qwen-20261123.png", pB, width = 17, height = 5, dpi = 100)

# C (2026-11-23): accuracy and macro-F1 by k, one panel per model. Macro-F1 over the three classes,
# computed per option order and then averaged (as in the overview); accuracy as above.
f1m <- function(g, p) mean(sapply(c("negative", "neutral", "positive"), function(c) {
  tp <- sum(g == c & p == c); fp <- sum(g != c & p == c); fn <- sum(g == c & p != c)
  if (tp == 0) 0 else 2 * tp / (2 * tp + fp + fn)
}))
af <- d |> group_by(model, variant, k) |>
  summarise(acc = 100 * mean(correct), f1 = 100 * f1m(gold, pred), .groups = "drop") |>
  group_by(model, k) |> summarise(across(c(acc, f1), mean), .groups = "drop")
sink("turnos_mtcsd_linhas-20261123.txt", append = TRUE)
cat("\nAccuracy and macro-F1 (%) by k (macro-F1 per order, then averaged):\n")
print(af |> mutate(across(c(acc, f1), ~ round(.x, 1))) |> pivot_wider(names_from = k, values_from = c(acc, f1)) |>
        as.data.frame(), row.names = FALSE)
sink()
afl <- af |> pivot_longer(c(acc, f1), names_to = "m", values_to = "v") |>
  mutate(m = factor(m, c("acc", "f1"), c("Acurácia", "F1 macro")))
pC <- ggplot(afl, aes(k, v, colour = m)) +
  geom_line(linewidth = 1.3) + geom_point(size = 2.8) +
  facet_wrap(~ model, nrow = 2) +
  scale_colour_manual(values = c("#2B4C7E", "#B35806"), name = NULL) +
  scale_x_continuous(breaks = 0:4) + scale_y_continuous(limits = c(30, 80)) +
  labs(x = "Turnos de contexto (k)", y = "%",
       caption = sprintf("MT-CSD, mesmos %d itens em todo k (pelo menos 4 turnos de contexto); média das três ordens de opções.", n)) +
  theme_bw(base_size = 17) + theme(legend.position = "top", panel.grid.minor = element_blank(),
                                   strip.text = element_text(face = "bold"), plot.caption = element_text(size = 12))
ggsave("turnos_mtcsd_acc_f1-20261123.pdf", pC, width = 15, height = 6.6)
ggsave("turnos_mtcsd_acc_f1-20261123.png", pC, width = 15, height = 6.6, dpi = 100)
