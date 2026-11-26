# Does context polarize the answers? Share of "neutral" answers by number of context turns (2026-11-22).
# Source: ablation run 20261118-094702-838561 (eval/ablation.csv). Fixed cohort: items with >= 4
# context turns, k = 0..4 (same items at every k). MT-CSD and MUStARD++ non-ironic. All option
# orders pooled. NanoJev excluded. Dotted line: share of neutral in the ground truth of the same items.
# Outputs: polarizacao-20261122.{txt,pdf,png}
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
lab <- c("twitter-roberta", "DeBERTa-NLI", "Qwen3.5-4B", "Mistral-7B", "Kev-4B", "Laya")
dl <- c(mtcsd = "MT-CSD", mustardpp = "MUStARD++ não irônico")
abl <- read_csv("../results/20261118-094702-838561/eval/ablation.csv", show_col_types = FALSE) |>
  filter(model %in% ord, dataset == "mtcsd" | sarcastic == 0) |>
  group_by(model, variant, id) |> filter(max(k) >= 4) |> ungroup() |> filter(k <= 4) |>
  mutate(model = factor(model, ord, lab), ds = factor(dl[dataset], dl))

s <- abl |> group_by(ds, model, k) |> summarise(neu = 100 * mean(pred == "neutral"), n_items = n_distinct(id), .groups = "drop")
g <- abl |> distinct(ds, id, gold) |> group_by(ds) |> summarise(neu = 100 * mean(gold == "neutral"), n = n())

sink("polarizacao-20261122.txt")
cat("% of answers = neutral by k; cohort >= 4 context turns; all orders pooled.\n")
print(g |> as.data.frame(), row.names = FALSE)
print(s |> mutate(neu = round(neu, 1)) |> select(-n_items) |>
        pivot_wider(names_from = k, values_from = neu, names_prefix = "k") |>
        mutate(delta_k4_k0 = k4 - k0) |> as.data.frame(), row.names = FALSE)
sink()

gl <- g |> mutate(lbl = sprintf("%s (n = %d)", ds, n))
s <- s |> left_join(gl |> select(ds, lbl), by = "ds")
p <- ggplot(s, aes(k, neu, colour = lbl)) +
  geom_hline(data = gl, aes(yintercept = neu, colour = lbl, linetype = "neutro no ground-truth"), linewidth = 0.8) +
  geom_line(aes(linetype = "respostas do modelo"), linewidth = 1.1) + geom_point(size = 2.5) +
  facet_wrap(~ model, nrow = 1) +
  scale_colour_manual(values = c("#2B4C7E", "#B35806"), name = NULL) +
  scale_linetype_manual(values = c("respostas do modelo" = 1, "neutro no ground-truth" = 3), name = NULL,
                        breaks = c("respostas do modelo", "neutro no ground-truth")) +
  guides(linetype = guide_legend(override.aes = list(colour = "grey30", shape = NA))) +
  scale_y_continuous(limits = c(0, 100)) + scale_x_continuous(breaks = 0:4) +
  labs(x = "Turnos de contexto (k)", y = "Respostas \"neutro\" (%)",
       title = "Com mais contexto, o modelo responde menos neutro?",
       caption = "Mesmos itens em todo k (pelo menos 4 turnos); três ordens de opções juntas.") +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", panel.grid.minor = element_blank(), plot.caption = element_text(size = 11))
ggsave("polarizacao-20261122.pdf", p, width = 17, height = 5.5)
ggsave("polarizacao-20261122.png", p, width = 17, height = 5.5, dpi = 100)
