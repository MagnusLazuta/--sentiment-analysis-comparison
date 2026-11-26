# Class imbalance: ground-truth shares, majority-class baseline, and balanced accuracy (2026-11-22).
# Source: merged run 20261121-220717-839516/merged/eval/predictions.csv. T1 = non-ironic, no context;
# T2 = non-ironic, whole context (MUStARD++, MT-CSD); T3 = ironic, no context.
# Balanced accuracy = mean recall over the classes present in the ground truth (chance with one class
# always = 1 / number of classes present). Mean over the three option orders (LLMs, System One).
# Majority baseline = accuracy of always answering the most frequent ground-truth class.
# Outputs: desbalanceamento-20261122.txt, desbal_gold-20261122.{pdf,png}, desbal_heat-20261122.{pdf,png}
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
lab <- c("twitter-roberta", "DeBERTa-NLI", "Qwen3.5-4B", "Mistral-7B", "Kev-4B", "Laya")
ds <- c(mustardpp = "MUStARD++", mtcsd = "MT-CSD", sarcasm_nlp = "Sarcasm_NLP", filatova = "Filatova")
lv <- c("negative", "neutral", "positive"); pt <- c("negativo", "neutro", "positivo")
p <- read_csv("../results/20261121-220717-839516/merged/eval/predictions.csv", show_col_types = FALSE,
              col_select = c(model, dataset, id, sarcastic, gold, condition, variant, pred, correct)) |>
  filter(model %in% ord) |>
  mutate(tarefa = case_when(sarcastic == 0 & condition == "no_context" ~ "T1: sem contexto",
                            sarcastic == 0 & condition == "with_context" ~ "T2: com contexto",
                            sarcastic == 1 & condition == "no_context" ~ "T3: ironia")) |>
  filter(!is.na(tarefa), dataset %in% names(ds)) |> mutate(dataset = factor(ds[dataset], ds))

g <- p |> distinct(tarefa, dataset, id, gold) |> count(tarefa, dataset, gold) |>
  group_by(tarefa, dataset) |> mutate(pct = 100 * n / sum(n), N = sum(n)) |> ungroup()
maj <- g |> group_by(tarefa, dataset) |> summarise(maj = max(pct), N = first(N), k = n(), .groups = "drop")
ba <- p |> group_by(tarefa, dataset, model, variant, gold) |> summarise(rec = mean(correct), .groups = "drop") |>
  group_by(tarefa, dataset, model, variant) |> summarise(bacc = 100 * mean(rec), .groups = "drop") |>
  group_by(tarefa, dataset, model) |> summarise(bacc = mean(bacc), .groups = "drop")
acc <- p |> group_by(tarefa, dataset, model, variant) |> summarise(acc = 100 * mean(correct), .groups = "drop") |>
  group_by(tarefa, dataset, model) |> summarise(acc = mean(acc), .groups = "drop")
r <- ba |> left_join(acc, by = c("tarefa", "dataset", "model")) |> left_join(maj, by = c("tarefa", "dataset")) |>
  mutate(model = factor(model, rev(ord), rev(lab)))

sink("desbalanceamento-20261122.txt")
cat("Ground-truth shares (%):\n")
print(g |> mutate(pct = round(pct, 1)) |> select(-n) |> pivot_wider(names_from = gold, values_from = pct) |> as.data.frame(), row.names = FALSE)
cat("\nAccuracy / balanced accuracy (%), mean over orders; majority baseline and chance (1/k):\n")
print(r |> mutate(cell = sprintf("%.1f / %.1f", acc, bacc)) |> select(tarefa, dataset, maj, k, model, cell) |>
        mutate(maj = round(maj, 1)) |> pivot_wider(names_from = model, values_from = cell) |>
        arrange(tarefa, dataset) |> as.data.frame(), row.names = FALSE)
cat("\nRanks by accuracy vs by balanced accuracy (1 = best):\n")
print(r |> group_by(tarefa, dataset) |> mutate(r_acc = rank(-acc), r_bacc = rank(-bacc)) |> ungroup() |>
        filter(r_acc != r_bacc) |> select(tarefa, dataset, model, r_acc, r_bacc) |> as.data.frame(), row.names = FALSE)
sink()

# Figure 1: ground-truth shares with the majority baseline
gg <- g |> mutate(gold = factor(pt[match(gold, lv)], pt), col = sprintf("%s · %s", sub(":.*", "", tarefa), dataset)) |>
  filter(tarefa != "T2: com contexto") |>
  mutate(col = factor(col, rev(c(paste("T1", ds, sep = " · "), paste("T3", ds[c(1, 3, 4)], sep = " · ")))))
ml <- gg |> group_by(col) |> summarise(maj = max(pct), N = first(N))
f1 <- ggplot(gg, aes(pct, col, fill = gold)) + geom_col(width = 0.7) +
  geom_text(data = ml, aes(x = 103, y = col, label = sprintf("maioria %.0f%%  (n = %d)", maj, N)),
            inherit.aes = FALSE, hjust = 0, size = 5) +
  scale_fill_manual(values = c(negativo = "#C0392B", neutro = "#BDC3C7", positivo = "#2471A3"), name = NULL) +
  scale_x_continuous(breaks = c(0, 50, 100), limits = c(0, 150)) +
  labs(x = "% dos itens no ground-truth", y = NULL) +
  theme_minimal(base_size = 17) + theme(legend.position = "top", panel.grid.minor = element_blank(),
                                        panel.grid.major.y = element_blank())
ggsave("desbal_gold-20261122.pdf", f1, width = 14, height = 6)
ggsave("desbal_gold-20261122.png", f1, width = 14, height = 6, dpi = 100)

# Figure 2: balanced accuracy heatmap with a majority-baseline row
nl <- maj |> arrange(dataset) |> mutate(col = factor(sprintf("%s\n(n = %d)", dataset, N), unique(sprintf("%s\n(n = %d)", dataset, N))))
h <- r |> select(tarefa, dataset, model, v = bacc, acc) |>
  bind_rows(maj |> transmute(tarefa, dataset, model = "classe majoritária", v = 100 / k, acc = maj)) |>
  left_join(nl |> select(tarefa, dataset, col), by = c("tarefa", "dataset")) |>
  mutate(model = factor(model, c("classe majoritária", rev(lab))))
f2 <- ggplot(h, aes(col, model, fill = ifelse(model == "classe majoritária", NA, v))) +
  geom_tile(colour = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%.0f", v)), size = 6.5, fontface = "bold", nudge_y = 0.12) +
  geom_text(aes(label = sprintf("acc %.0f", acc)), size = 4.3, colour = "grey25", nudge_y = -0.22) +
  facet_grid(. ~ tarefa, scales = "free_x", space = "free_x") +
  scale_fill_gradient(low = "#F4F6F9", high = "#7F9CC4", limits = c(0, 100), na.value = "#FFFFFF") +
  labs(x = NULL, y = NULL,
       caption = "Número grande: acurácia balanceada (média da revocação por classe); abaixo, a acurácia. Linha de baixo: chutar sempre a classe majoritária.") +
  theme_minimal(base_size = 16) +
  theme(panel.grid = element_blank(), legend.position = "none", strip.text = element_text(face = "bold", size = 16),
        plot.caption = element_text(size = 12),
        axis.text.y = element_text(face = ifelse(levels(h$model) == "classe majoritária", "italic", "plain")))
ggsave("desbal_heat-20261122.pdf", f2, width = 17, height = 6.6)
ggsave("desbal_heat-20261122.png", f2, width = 17, height = 6.6, dpi = 100)

# Versions without MT-CSD for the irony part of the slides (2026-11-23)
g0 <- g; maj0 <- maj; r0 <- r; ds0 <- ds
g <- g0 |> filter(dataset != "MT-CSD"); maj <- maj0 |> filter(dataset != "MT-CSD"); r <- r0 |> filter(dataset != "MT-CSD")
ds <- ds0[ds0 != "MT-CSD"]
# Figure 1: ground-truth shares with the majority baseline
gg <- g |> mutate(gold = factor(pt[match(gold, lv)], pt), col = sprintf("%s · %s", sub(":.*", "", tarefa), dataset)) |>
  filter(tarefa != "T2: com contexto") |>
  mutate(col = factor(col, rev(c(paste("T1", ds, sep = " · "), paste("T3", ds, sep = " · ")))))
ml <- gg |> group_by(col) |> summarise(maj = max(pct), N = first(N))
f1 <- ggplot(gg, aes(pct, col, fill = gold)) + geom_col(width = 0.7) +
  geom_text(data = ml, aes(x = 103, y = col, label = sprintf("maioria %.0f%%  (n = %d)", maj, N)),
            inherit.aes = FALSE, hjust = 0, size = 5) +
  scale_fill_manual(values = c(negativo = "#C0392B", neutro = "#BDC3C7", positivo = "#2471A3"), name = NULL) +
  scale_x_continuous(breaks = c(0, 50, 100), limits = c(0, 150)) +
  labs(x = "% dos itens no ground-truth", y = NULL) +
  theme_minimal(base_size = 17) + theme(legend.position = "top", panel.grid.minor = element_blank(),
                                        panel.grid.major.y = element_blank())
ggsave("desbal_gold_semmtcsd-20261123.pdf", f1, width = 14, height = 6)
ggsave("desbal_gold_semmtcsd-20261123.png", f1, width = 14, height = 6, dpi = 100)

# Figure 2: balanced accuracy heatmap with a majority-baseline row
nl <- maj |> arrange(dataset) |> mutate(col = factor(sprintf("%s\n(n = %d)", dataset, N), unique(sprintf("%s\n(n = %d)", dataset, N))))
h <- r |> select(tarefa, dataset, model, v = bacc, acc) |>
  bind_rows(maj |> transmute(tarefa, dataset, model = "classe majoritária", v = 100 / k, acc = maj)) |>
  left_join(nl |> select(tarefa, dataset, col), by = c("tarefa", "dataset")) |>
  mutate(model = factor(model, c("classe majoritária", rev(lab))))
f2 <- ggplot(h, aes(col, model, fill = ifelse(model == "classe majoritária", NA, v))) +
  geom_tile(colour = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%.0f", v)), size = 6.5, fontface = "bold", nudge_y = 0.12) +
  geom_text(aes(label = sprintf("acc %.0f", acc)), size = 4.3, colour = "grey25", nudge_y = -0.22) +
  facet_grid(. ~ tarefa, scales = "free_x", space = "free_x") +
  scale_fill_gradient(low = "#F4F6F9", high = "#7F9CC4", limits = c(0, 100), na.value = "#FFFFFF") +
  labs(x = NULL, y = NULL,
       caption = "Número grande: acurácia balanceada (média da revocação por classe); abaixo, a acurácia. Linha de baixo: chutar sempre a classe majoritária.") +
  theme_minimal(base_size = 16) +
  theme(panel.grid = element_blank(), legend.position = "none", strip.text = element_text(face = "bold", size = 16),
        plot.caption = element_text(size = 12),
        axis.text.y = element_text(face = ifelse(levels(h$model) == "classe majoritária", "italic", "plain")))
ggsave("desbal_heat_semmtcsd-20261123.pdf", f2, width = 17, height = 6.6)
ggsave("desbal_heat_semmtcsd-20261123.png", f2, width = 17, height = 6.6, dpi = 100)

# Heatmap with macro-F1 next to accuracy (2026-11-23), without MT-CSD. Macro-F1 over the three labels,
# per option order, then averaged (as in metrics.csv); majority row: F1 of always answering the majority class.
f1m <- function(g, p) mean(sapply(lv, function(c) {
  tp <- sum(g == c & p == c); fp <- sum(g != c & p == c); fn <- sum(g == c & p != c)
  if (tp == 0) 0 else 2 * tp / (2 * tp + fp + fn)
}))
f1 <- p |> group_by(tarefa, dataset, model, variant) |> summarise(f1 = 100 * f1m(gold, pred), .groups = "drop") |>
  group_by(tarefa, dataset, model) |> summarise(f1 = mean(f1), .groups = "drop") |>
  mutate(model = factor(model, rev(ord), rev(lab))) |> filter(dataset != "MT-CSD")
majf1 <- maj |> mutate(f1 = 100 * (2 * (maj / 100) / (1 + maj / 100)) / 3)
h2 <- r |> select(tarefa, dataset, model, v = bacc, acc) |> left_join(f1, by = c("tarefa", "dataset", "model")) |>
  bind_rows(majf1 |> transmute(tarefa, dataset, model = "classe majoritária", v = 100 / k, acc = maj, f1)) |>
  left_join(nl |> select(tarefa, dataset, col), by = c("tarefa", "dataset")) |>
  mutate(model = factor(model, c("classe majoritária", rev(lab))))
sink("desbalanceamento-20261123.txt")
cat("Balanced accuracy / accuracy / macro-F1 (%), mean over orders; without MT-CSD.\n")
print(h2 |> mutate(cell = sprintf("%.1f / %.1f / %.1f", v, acc, f1)) |> select(tarefa, dataset, model, cell) |>
        pivot_wider(names_from = model, values_from = cell) |> as.data.frame(), row.names = FALSE)
sink()
f3 <- ggplot(h2, aes(col, model, fill = ifelse(model == "classe majoritária", NA, v))) +
  geom_tile(colour = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%.0f", v)), size = 6.5, fontface = "bold", nudge_y = 0.14) +
  geom_text(aes(label = sprintf("acc %.0f · F1 %.0f", acc, f1)), size = 4.1, colour = "grey25", nudge_y = -0.22) +
  facet_grid(. ~ tarefa, scales = "free_x", space = "free_x") +
  scale_fill_gradient(low = "#F4F6F9", high = "#7F9CC4", limits = c(0, 100), na.value = "#FFFFFF") +
  labs(x = NULL, y = NULL,
       caption = "Número grande: acurácia balanceada; abaixo, acurácia e F1 macro. Linha de baixo: chutar sempre a classe majoritária.") +
  theme_minimal(base_size = 16) +
  theme(panel.grid = element_blank(), legend.position = "none", strip.text = element_text(face = "bold", size = 16),
        plot.caption = element_text(size = 12),
        axis.text.y = element_text(face = ifelse(levels(h2$model) == "classe majoritária", "italic", "plain")))
ggsave("desbal_heat_f1_semmtcsd-20261123.pdf", f3, width = 17, height = 6.6)
ggsave("desbal_heat_f1_semmtcsd-20261123.png", f3, width = 17, height = 6.6, dpi = 100)
