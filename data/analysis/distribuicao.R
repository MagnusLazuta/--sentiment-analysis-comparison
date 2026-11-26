# Answer distribution vs ground-truth, and accuracy vs share of "negative" answers (2026-11-22).
# Source: merged run 20261121-220717-839516/merged/eval/predictions.csv. T1 = non-ironic, no context;
# T3 = ironic, no context. Answer shares pooled over the three option orders (LLMs, System One).
# NanoJev left out. Also: accuracy per option order (v1, v2, v3) for every cell of the overview.
# Outputs: distribuicao-20261122.{txt,pdf,png}, negativo_acuracia-20261122.{pdf,png}
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
lab <- c("twitter-roberta", "DeBERTa-NLI", "Qwen3.5-4B", "Mistral-7B", "Kev-4B", "Laya")
ds <- c(mustardpp = "MUStARD++", mtcsd = "MT-CSD", sarcasm_nlp = "Sarcasm_NLP", filatova = "Filatova")
lv <- c("negative", "neutral", "positive"); pt <- c("negativo", "neutro", "positivo")
cc <- c(negativo = "#C0392B", neutro = "#BDC3C7", positivo = "#2471A3")
p <- read_csv("../results/20261121-220717-839516/merged/eval/predictions.csv", show_col_types = FALSE,
              col_select = c(model, dataset, id, sarcastic, gold, condition, variant, pred, correct)) |>
  filter(model %in% ord, condition == "no_context") |>
  mutate(tarefa = ifelse(sarcastic == 1, "T3", "T1"), dataset = factor(ds[dataset], ds))

gold <- p |> distinct(tarefa, dataset, id, gold) |> count(tarefa, dataset, gold) |>
  group_by(tarefa, dataset) |> mutate(pct = 100 * n / sum(n), N = sum(n)) |> ungroup() |>
  transmute(tarefa, dataset, quem = "ground-truth", classe = gold, pct, N)
pred <- p |> count(tarefa, dataset, model, pred) |> group_by(tarefa, dataset, model) |>
  mutate(pct = 100 * n / sum(n)) |> ungroup() |>
  transmute(tarefa, dataset, quem = lab[match(model, ord)], classe = pred, pct)
acc <- p |> group_by(tarefa, dataset, model) |> summarise(acc = 100 * mean(correct), .groups = "drop") |>
  mutate(quem = lab[match(model, ord)])
byv <- p |> group_by(tarefa, dataset, model, variant) |> summarise(acc = 100 * mean(correct), .groups = "drop") |>
  pivot_wider(names_from = variant, values_from = acc) |> mutate(spread = pmax(v1, v2, v3) - pmin(v1, v2, v3))

sink("distribuicao-20261122.txt")
cat("Ground-truth shares (%), no context:\n")
print(gold |> mutate(pct = round(pct, 1)) |> pivot_wider(names_from = classe, values_from = pct) |> as.data.frame(), row.names = FALSE)
cat("\nAnswer shares (%) and accuracy, pooled over orders:\n")
print(pred |> mutate(pct = round(pct, 1)) |> pivot_wider(names_from = classe, values_from = pct, values_fill = 0) |>
        left_join(acc |> select(tarefa, dataset, quem, acc), by = c("tarefa", "dataset", "quem")) |>
        mutate(acc = round(acc, 1)) |> arrange(tarefa, dataset, quem) |> as.data.frame(), row.names = FALSE)
cat("\nAccuracy (%) per option order (LLMs and System One):\n")
print(byv |> filter(!is.na(v1)) |> mutate(across(c(v1, v2, v3, spread), ~ round(.x, 1))) |> select(-none) |>
        arrange(desc(spread)) |> as.data.frame(), row.names = FALSE)
cat("\nDeBERTa-NLI precision of 'negative' (%):\n")
print(p |> filter(model == "deberta-nli", pred == "negative") |> group_by(tarefa, dataset) |>
        summarise(n_neg = n(), precision = round(100 * mean(gold == "negative"), 1), .groups = "drop") |> as.data.frame(), row.names = FALSE)
sink()

# Figure 1: answer distribution vs ground-truth
d <- bind_rows(gold |> select(-N), pred) |>
  mutate(quem = factor(quem, rev(c("ground-truth", lab))), classe = factor(pt[match(classe, lv)], pt),
         painel = factor(paste(tarefa, dataset, sep = " · "),
                         c(paste("T1", ds, sep = " · "), paste("T3", ds[c(1, 3, 4)], sep = " · "))))
f1 <- ggplot(d, aes(pct, quem, fill = classe)) + geom_col(width = 0.75) +
  facet_wrap(~ painel, nrow = 2) +
  scale_fill_manual(values = cc, name = NULL) + scale_x_continuous(breaks = c(0, 50, 100)) +
  labs(x = "% das respostas", y = NULL,
       caption = "Sem contexto; três ordens de opções juntas. T1: itens não irônicos; T3: irônicos.") +
  theme_minimal(base_size = 15) +
  theme(legend.position = "top", panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        axis.text.y = element_text(face = ifelse(rev(c("ground-truth", lab)) == "ground-truth", "bold", "plain")),
        strip.text = element_text(face = "bold"), plot.caption = element_text(size = 11))
ggsave("distribuicao-20261122.pdf", f1, width = 17, height = 7.5)
ggsave("distribuicao-20261122.png", f1, width = 17, height = 7.5, dpi = 100)

# Figure 2: accuracy vs share of negative answers
s <- pred |> filter(classe == "negative") |> select(tarefa, dataset, quem, neg = pct) |>
  right_join(acc |> select(tarefa, dataset, quem, acc), by = c("tarefa", "dataset", "quem")) |>
  mutate(neg = coalesce(neg, 0), deb = quem == "DeBERTa-NLI",
         tarefa = recode(tarefa, T1 = "T1: não irônicos", T3 = "T3: irônicos"))
gneg <- gold |> filter(classe == "negative") |>
  mutate(tarefa = recode(tarefa, T1 = "T1: não irônicos", T3 = "T3: irônicos"))
f2 <- ggplot(s, aes(neg, acc, colour = dataset)) +
  geom_point(aes(shape = deb), size = 4.5, stroke = 1.3) +
  geom_text(data = s |> filter(deb), aes(label = "DeBERTa-NLI"), nudge_y = 4, size = 4.2, show.legend = FALSE) +
  facet_wrap(~ tarefa) +
  scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 1), guide = "none") +
  scale_colour_manual(values = c("#2B4C7E", "#1B7837", "#B35806", "#7D3C98"), name = NULL) +
  scale_x_continuous(limits = c(0, 100)) + scale_y_continuous(limits = c(0, 100)) +
  labs(x = "% de respostas \"negativo\"", y = "Acurácia (%)",
       caption = "Cada ponto: um modelo em uma base (sem contexto, três ordens juntas). Círculo vazio: DeBERTa-NLI.") +
  theme_bw(base_size = 15) + theme(legend.position = "top", panel.grid.minor = element_blank(),
                                   plot.caption = element_text(size = 11))
ggsave("negativo_acuracia-20261122.pdf", f2, width = 14, height = 6.5)
ggsave("negativo_acuracia-20261122.png", f2, width = 14, height = 6.5, dpi = 100)

# Version without MT-CSD for the irony part of the slides (2026-11-23)
gold <- gold |> filter(dataset != "MT-CSD"); pred <- pred |> filter(dataset != "MT-CSD")
ds <- ds[ds != "MT-CSD"]
# Figure 1: answer distribution vs ground-truth
d <- bind_rows(gold |> select(-N), pred) |>
  mutate(quem = factor(quem, rev(c("ground-truth", lab))), classe = factor(pt[match(classe, lv)], pt),
         painel = factor(paste(tarefa, dataset, sep = " · "),
                         c(paste("T1", ds, sep = " · "), paste("T3", ds, sep = " · "))))
f1 <- ggplot(d, aes(pct, quem, fill = classe)) + geom_col(width = 0.75) +
  facet_wrap(~ painel, nrow = 2) +
  scale_fill_manual(values = cc, name = NULL) + scale_x_continuous(breaks = c(0, 50, 100)) +
  labs(x = "% das respostas", y = NULL,
       caption = "Sem contexto; três ordens de opções juntas. T1: itens não irônicos; T3: irônicos.") +
  theme_minimal(base_size = 15) +
  theme(legend.position = "top", panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
        axis.text.y = element_text(face = ifelse(rev(c("ground-truth", lab)) == "ground-truth", "bold", "plain")),
        strip.text = element_text(face = "bold"), plot.caption = element_text(size = 11))
ggsave("distribuicao_semmtcsd-20261123.pdf", f1, width = 17, height = 7.5)
ggsave("distribuicao_semmtcsd-20261123.png", f1, width = 17, height = 7.5, dpi = 100)

