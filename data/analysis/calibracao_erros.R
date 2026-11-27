# Calibration figure for the paper, with confident errors as a share of errors (2026-11-26).
#
# Panels ECE T1/T2/T3 and option-order stability: taken as-is from calibracao_slide-20261123.csv
# (slide figure of 2026-11-23). Confident errors: recomputed from the merged main run
# 20261121-220717-839516 (eval/predictions.csv), order v1 (twitter-roberta and DeBERTa-NLI: none),
# NanoJev excluded, same item set as the slide panel: T1 = no context, non-ironic MUStARD++ and
# MT-CSD; T2 = with context, same items; T3 = no context, ironic items of MUStARD++, Sarcasm_NLP
# and Filatova. Check: the old metric (confident errors / all answers) reproduces the slide CSV.
# New metric: errors with confidence >= 0.9 / all errors (%).
# Outputs (this folder): calibracao-20261126.{csv,pdf,png}
suppressPackageStartupMessages({library(dplyr); library(readr); library(ggplot2)})
ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
lab <- c("twitter-roberta", "DeBERTa-NLI", "Qwen3.5-4B", "Mistral-7B", "Kev-4B", "Laya")
cols <- setNames(c("#6BAED6", "#08519C", "#9E9AC8", "#54278F", "#FD8D3C", "#A63603"), lab)

p <- read_csv("../results/20261121-220717-839516/merged/eval/predictions.csv", show_col_types = FALSE,
              col_select = c(model, dataset, sarcastic, condition, variant, correct, confidence)) |>
  filter(model %in% ord, variant %in% c("v1", "none")) |>
  filter((dataset %in% c("mustardpp", "mtcsd") & sarcastic != 1) |
         (condition == "no_context" & dataset %in% c("mustardpp", "sarcasm_nlp", "filatova") & sarcastic == 1)) |>
  mutate(ce = confidence >= 0.9 & correct == 0)
ce <- p |> group_by(model) |>
  summarise(old = 100 * mean(ce), new = 100 * sum(ce) / sum(correct == 0), n_err = sum(correct == 0), .groups = "drop") |>
  mutate(m = factor(model, ord, lab))

slide <- read_csv("calibracao_slide-20261123.csv", show_col_types = FALSE)
chk <- slide |> filter(metric == "Erros confiantes (%)") |> inner_join(ce, by = c("m" = "m"))
stopifnot(nrow(chk) == 6, all(abs(chk$v - chk$old) < 0.01))

met <- c("ECE T1" = "ECE T1", "ECE T2" = "ECE T2", "ECE T3 irôn." = "ECE T3 ironic",
         "Mesma resposta nas 3 ordens (%)" = "Same answer in\nthe 3 orders (%)")
d <- bind_rows(
  slide |> filter(metric %in% names(met)) |> mutate(metric = met[metric]),
  ce |> transmute(metric = "Confident errors\n(% of errors)", m = as.character(m), v = new)) |>
  mutate(metric = factor(metric, c("ECE T1", "ECE T2", "ECE T3 ironic", "Confident errors\n(% of errors)",
                                   "Same answer in\nthe 3 orders (%)")),
         m = factor(m, lab),
         txt = ifelse(grepl("ECE", metric), sprintf("%.2f", v), sprintf("%.0f", v)))
write_csv(d |> select(metric, m, v), "calibracao-20261126.csv")

ref <- tibble(metric = factor(c("ECE T1", "ECE T2", "ECE T3 ironic"), levels(d$metric)), y = 0.05)
top <- tibble(metric = factor(c("ECE T1", "ECE T2", "ECE T3 ironic"), levels(d$metric)), y = 0.45)  # same scale on ECE panels
g <- ggplot(d, aes(m, v, fill = m)) +
  geom_col(width = 0.8) +
  geom_text(aes(label = txt), vjust = -0.3, size = 3.4) +
  geom_blank(data = top, aes(x = 1, y = y), inherit.aes = FALSE) +
  geom_hline(data = ref, aes(yintercept = y), linetype = "dashed", colour = "grey25") +
  facet_wrap(~ metric, nrow = 1, scales = "free_y") +
  scale_fill_manual(values = cols, name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(x = NULL, y = NULL) +
  theme_bw(base_size = 14) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), legend.position = "top",
        panel.grid.minor = element_blank())
ggsave("calibracao-20261126.pdf", g, width = 17.6, height = 6.05)
ggsave("calibracao-20261126.png", g, width = 17.6, height = 6.05, dpi = 100)
print(ce)
