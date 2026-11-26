# Overview: accuracy and macro-F1 per model x task x dataset (2026-11-22).
# Source: merged run 20261121-220717-839516/merged/eval/metrics.csv. Mean over the three option
# orders (v1-v3) for LLMs and System One; twitter-roberta and DeNLI have one ("none").
# T1 = non-ironic, no context; T2 = non-ironic, whole context; T3 = ironic, no context.
# NanoJev (negative control) left out. Outputs: visao_geral-20261122.{txt,pdf,png}
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
lab <- c("twitter-roberta", "DeBERTa-NLI", "Qwen3.5-4B", "Mistral-7B", "Kev-4B", "Laya")
ds <- c(mustardpp = "MUStARD++", mtcsd = "MT-CSD", sarcasm_nlp = "Sarcasm_NLP", filatova = "Filatova")
m <- read_csv("../results/20261121-220717-839516/merged/eval/metrics.csv", show_col_types = FALSE) |>
  filter(model %in% ord) |>
  mutate(tarefa = case_when(group == "non_sarcastic" & condition == "no_context" ~ "T1: sem contexto",
                            group == "non_sarcastic" & condition == "with_context" ~ "T2: com contexto",
                            group == "sarcastic" & condition == "no_context" ~ "T3: ironia")) |>
  filter(!is.na(tarefa)) |>
  group_by(model, tarefa, dataset) |>
  summarise(acc = 100 * mean(accuracy), f1 = 100 * mean(macro_f1), n = first(n), .groups = "drop") |>
  mutate(model = factor(model, rev(ord), rev(lab)), dataset = factor(ds[dataset], ds))

sink("visao_geral-20261122.txt")
cat("Accuracy / macro-F1 (%), mean over option orders.\n")
print(m |> mutate(cell = sprintf("%.1f / %.1f", acc, f1)) |> select(tarefa, dataset, n, model, cell) |>
        pivot_wider(names_from = model, values_from = cell) |> arrange(tarefa, dataset) |> as.data.frame(), row.names = FALSE)
sink()

nl <- m |> distinct(tarefa, dataset, n) |> mutate(col = sprintf("%s\n(n = %d)", dataset, n))
nl <- nl |> arrange(dataset) |> mutate(col = factor(col, unique(col)))
m <- m |> left_join(nl, by = c("tarefa", "dataset", "n"))
p <- ggplot(m, aes(col, model, fill = acc)) +
  geom_tile(colour = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%.0f", acc)), size = 6.5, fontface = "bold", nudge_y = 0.12) +
  geom_text(aes(label = sprintf("F1 %.0f", f1)), size = 4.3, colour = "grey25", nudge_y = -0.22) +
  facet_grid(. ~ tarefa, scales = "free_x", space = "free_x") +
  scale_fill_gradient(low = "#F4F6F9", high = "#7F9CC4", limits = c(0, 100), name = "Acurácia (%)") +
  labs(x = NULL, y = NULL, caption = "Acurácia (número grande) e F1 macro (abaixo), em %; média das três ordens de opções.") +
  theme_minimal(base_size = 16) +
  theme(panel.grid = element_blank(), legend.position = "none", strip.text = element_text(face = "bold", size = 16),
        plot.caption = element_text(size = 12))
ggsave("visao_geral-20261122.pdf", p, width = 17, height = 6)
ggsave("visao_geral-20261122.png", p, width = 17, height = 6, dpi = 100)

# T1 only (for the T1 slide), same data and style
p1 <- ggplot(m |> filter(tarefa == "T1: sem contexto"), aes(col, model, fill = acc)) +
  geom_tile(colour = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%.0f", acc)), size = 7.5, fontface = "bold", nudge_y = 0.12) +
  geom_text(aes(label = sprintf("F1 %.0f", f1)), size = 5, colour = "grey25", nudge_y = -0.22) +
  scale_fill_gradient(low = "#F4F6F9", high = "#7F9CC4", limits = c(0, 100)) +
  labs(x = NULL, y = NULL, caption = "T1: itens não irônicos, sem contexto. Acurácia (número grande) e F1 macro, em %; média das três ordens de opções.") +
  theme_minimal(base_size = 18) +
  theme(panel.grid = element_blank(), legend.position = "none", plot.caption = element_text(size = 13))
ggsave("visao_geral_t1-20261122.pdf", p1, width = 12, height = 6)
ggsave("visao_geral_t1-20261122.png", p1, width = 12, height = 6, dpi = 100)

# 2026-11-23: split for the slides. T1 slide = only the bases used later with context (MUStARD++, MT-CSD);
# irony part = T1 of Sarcasm_NLP and Filatova next to T3 of the three irony bases.
tile <- function(d, cap, w, h, f) {
  p <- ggplot(d, aes(col, model, fill = acc)) +
    geom_tile(colour = "white", linewidth = 1) +
    geom_text(aes(label = sprintf("%.0f", acc)), size = 7.5, fontface = "bold", nudge_y = 0.12) +
    geom_text(aes(label = sprintf("F1 %.0f", f1)), size = 5, colour = "grey25", nudge_y = -0.22) +
    facet_grid(. ~ tarefa, scales = "free_x", space = "free_x") +
    scale_fill_gradient(low = "#F4F6F9", high = "#7F9CC4", limits = c(0, 100)) +
    labs(x = NULL, y = NULL, caption = cap) +
    theme_minimal(base_size = 18) +
    theme(panel.grid = element_blank(), legend.position = "none", strip.text = element_text(face = "bold", size = 18),
          plot.caption = element_text(size = 13))
  ggsave(paste0(f, ".pdf"), p, width = w, height = h)
  ggsave(paste0(f, ".png"), p, width = w, height = h, dpi = 100)
}
tile(m |> filter(tarefa == "T1: sem contexto", dataset %in% c("MUStARD++", "MT-CSD")),
     "Itens não irônicos, sem contexto. Acurácia (número grande) e F1 macro, em %; média das três ordens.",
     9, 6, "visao_geral_t1-20261123")
tile(m |> filter((tarefa == "T1: sem contexto" & dataset %in% c("Sarcasm_NLP", "Filatova")) | tarefa == "T3: ironia") |>
       mutate(tarefa = recode(tarefa, "T1: sem contexto" = "T1: não irônicos", "T3: ironia" = "T3: irônicos")),
     "Sem contexto. Acurácia (número grande) e F1 macro, em %; média das três ordens.",
     15, 6, "visao_geral_ironia-20261123")
# irony part, v2 (2026-11-23): T1 of the three irony bases (MUStARD++ added) next to T3
tile(m |> filter((tarefa == "T1: sem contexto" & dataset %in% c("MUStARD++", "Sarcasm_NLP", "Filatova")) | tarefa == "T3: ironia") |>
       mutate(tarefa = recode(tarefa, "T1: sem contexto" = "T1: não irônicos", "T3: ironia" = "T3: irônicos")),
     "Sem contexto. Acurácia (número grande) e F1 macro, em %; média das três ordens.",
     17, 6, "visao_geral_ironia2-20261123")
