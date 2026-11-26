# Answer distribution by number of context turns, split by the majority sentiment of the
# conversation (2026-11-22). Does more context pull the answer towards the conversation?
#
# Sources: ablation run 20261118-094702-838561 (eval/ablation.csv); conversation majority
# from conversa_tom.py (conversa_tom-20261122.csv: MT-CSD = majority of annotated stances
# of the context turns; MUStARD++ = VADER polarity of the whole context text, "tom_total").
# Fixed cohort: items with >= 4 context turns (same items at every k = 0..4). All option
# orders pooled (twitter-roberta and DeBERTa-NLI have one). NanoJev excluded. "misto" and
# "sem rótulo" groups are dropped from the figure (kept in the txt).
# Dotted line: share of that class in the gold of the same items (what the comment is).
# Outputs: turnos_tom_{mtcsd,mustard}-20261122.{pdf,png}, turnos_tom-20261122.txt
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
abl <- read_csv("../results/20261118-094702-838561/eval/ablation.csv", show_col_types = FALSE)
tom <- read_csv("conversa_tom-20261122.csv", show_col_types = FALSE) |>
  mutate(maioria = ifelse(dataset == "mustardpp", tom_total, maioria)) |> select(id, maioria)
ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
lab <- c("twitter-roberta", "DeBERTa-NLI", "Qwen3.5-4B", "Mistral-7B", "Kev-4B", "Laya")
lv <- c("negative", "neutral", "positive"); pt <- c("negativo", "neutro", "positivo")
cc <- c(negativo = "#C0392B", neutro = "#7F8C8D", positivo = "#2471A3")

d <- abl |> filter(model %in% ord, dataset == "mtcsd" | sarcastic == 0) |>
  group_by(model, variant, id) |> filter(max(k) >= 4) |> ungroup() |> filter(k <= 4) |>
  left_join(tom, by = "id") |>
  mutate(pred = factor(pred, lv, pt), gold = factor(gold, lv, pt),
         model = factor(model, ord, lab))

share <- d |> count(dataset, maioria, model, k, pred, .drop = FALSE) |>
  group_by(dataset, maioria, model, k) |> mutate(p = 100 * n / sum(n)) |> ungroup() |>
  filter(!is.na(maioria))
gold <- d |> distinct(dataset, maioria, id, gold) |> count(dataset, maioria, gold) |>
  group_by(dataset, maioria) |> mutate(p = 100 * n / sum(n), N = sum(n)) |> ungroup()

sink("turnos_tom-20261122.txt")
cat("Share of answers (%) by k and majority sentiment of the conversation; cohort >= 4 turns;",
    "all orders pooled. Gold share of the same items at the end.\n")
for (ds in c("mtcsd", "mustardpp")) {
  cat("\n=====", ds, "\n")
  print(gold |> filter(dataset == ds) |> select(maioria, gold, n, p, N) |> as.data.frame(), row.names = FALSE)
  print(share |> filter(dataset == ds) |> mutate(p = round(p, 1)) |>
          select(maioria, model, pred, k, p) |>
          pivot_wider(names_from = k, values_from = p, names_prefix = "k") |> as.data.frame(), row.names = FALSE)
}
sink()

for (ds in c("mtcsd", "mustardpp")) {
  keep <- c("negativo", "neutro", "positivo")
  g <- gold |> filter(dataset == ds, maioria %in% keep) |> rename(pred = gold)
  nl <- g |> distinct(maioria, N) |> mutate(lbl = sprintf("conversa %s\n(n = %d)", maioria, N))
  x <- share |> filter(dataset == ds, maioria %in% keep) |> left_join(nl, by = "maioria")
  g <- g |> left_join(nl |> select(maioria, lbl), by = "maioria")
  tit <- if (ds == "mtcsd") "MT-CSD: maioria da postura anotada nos turnos anteriores" else
    "MUStARD++ não irônico: tom (VADER) do texto dos turnos anteriores"
  p <- ggplot(x, aes(k, p, colour = pred)) +
    geom_hline(data = g, aes(yintercept = p, colour = pred), linetype = 3, linewidth = 0.8) +
    geom_line(linewidth = 1.1) + geom_point(size = 2) +
    facet_grid(lbl ~ model) +
    scale_colour_manual(values = cc, name = "Resposta") +
    scale_y_continuous(limits = c(0, 100)) + scale_x_continuous(breaks = 0:4) +
    labs(title = tit, x = "Turnos de contexto (k)", y = "% das respostas",
         caption = paste("Mesmos itens em todo k (pelo menos 4 turnos); três ordens de opções juntas.",
                         "Linha pontilhada: % de cada classe no ground-truth dos mesmos itens.")) +
    theme_bw(base_size = 15) +
    theme(legend.position = "top", panel.grid.minor = element_blank(),
          strip.text.y = element_text(size = 11), plot.caption = element_text(size = 11))
  f <- ifelse(ds == "mtcsd", "mtcsd", "mustard")
  ggsave(sprintf("turnos_tom_%s-20261122.pdf", f), p, width = 17, height = 10)
  ggsave(sprintf("turnos_tom_%s-20261122.png", f), p, width = 17, height = 10, dpi = 100)
}
