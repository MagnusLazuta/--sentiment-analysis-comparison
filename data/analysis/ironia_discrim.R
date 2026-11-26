# Does a model say "negative" because the text is ironic, or because it says "negative" a lot? (2026-11-22)
# Source: merged run 20261121-220717-839516/merged/eval/predictions.csv, no context, three option orders pooled.
# Per model x dataset (MUStARD++, Sarcasm_NLP, Filatova):
#   acerto  = % "negative" among ironic items whose gold is negative (hit);
#   alarme  = % "negative" among non-ironic items whose gold is NOT negative (false alarm);
#   literal = % "negative" among non-ironic items whose gold is negative (literal negatives).
# A model that only has a negative bias has acerto ~ alarme (on the diagonal).
# Discrimination = acerto - alarme. 95% CI: item bootstrap, 2,000 resamples, seed 5.
# Outputs: ironia_discrim-20261122.{txt,pdf,png}
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
set.seed(5)
ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
lab <- c("twitter-roberta", "DeBERTa-NLI", "Qwen3.5-4B", "Mistral-7B", "Kev-4B", "Laya")
ds <- c(mustardpp = "MUStARD++", sarcasm_nlp = "Sarcasm_NLP", filatova = "Filatova")
p <- read_csv("../results/20261121-220717-839516/merged/eval/predictions.csv", show_col_types = FALSE,
              col_select = c(model, dataset, id, sarcastic, gold, condition, variant, pred)) |>
  filter(model %in% ord, condition == "no_context", dataset %in% names(ds)) |>
  mutate(grp = case_when(sarcastic == 1 & gold == "negative" ~ "acerto",
                         sarcastic == 0 & gold != "negative" ~ "alarme",
                         sarcastic == 0 & gold == "negative" ~ "literal"),
         neg = pred == "negative") |> filter(!is.na(grp))
it <- p |> group_by(model, dataset, grp, id) |> summarise(neg = mean(neg), .groups = "drop")
bt <- function(a, b, B = 2000) quantile(replicate(B, mean(sample(a, replace = TRUE)) - mean(sample(b, replace = TRUE))), c(.025, .975))
r <- it |> group_by(model, dataset) |> group_modify(~ {
  a <- .x$neg[.x$grp == "acerto"]; b <- .x$neg[.x$grp == "alarme"]; l <- .x$neg[.x$grp == "literal"]
  ci <- 100 * bt(a, b)
  tibble(acerto = 100 * mean(a), alarme = 100 * mean(b), literal = 100 * mean(l),
         n_ir = length(a), n_al = length(b), disc = 100 * (mean(a) - mean(b)), lo = ci[1], hi = ci[2])
}) |> ungroup() |> mutate(model = factor(model, ord, lab), dataset = factor(ds[dataset], ds))

sink("ironia_discrim-20261122.txt")
cat("% 'negative': ironic negatives (acerto), non-ironic non-negatives (alarme), non-ironic negatives (literal).\n")
cat("disc = acerto - alarme [95% CI item bootstrap]. No context, orders pooled.\n")
print(r |> mutate(across(c(acerto, alarme, literal, disc, lo, hi), ~ round(.x, 1))) |> arrange(dataset, desc(disc)) |>
        as.data.frame(), row.names = FALSE)
sink()

f <- ggplot(r, aes(alarme, acerto)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
  annotate("text", x = 55, y = 22, label = "na diagonal:\nsó viés", colour = "grey40", size = 4.5, hjust = 0) +
  geom_segment(aes(xend = alarme, yend = alarme), colour = "grey70", linewidth = 0.6) +
  geom_point(aes(colour = model, size = model == "DeBERTa-NLI")) +
  facet_wrap(~ dataset) +
  scale_size_manual(values = c(`FALSE` = 3.5, `TRUE` = 6), guide = "none") +
  scale_colour_manual(values = c("#7F8C8D", "#C0392B", "#2B4C7E", "#B35806", "#1B7837", "#7D3C98"), name = NULL) +
  coord_equal(xlim = c(0, 100), ylim = c(0, 100)) +
  labs(x = "Diz \"negativo\" quando NÃO é negativo (% dos não irônicos)",
       y = "Diz \"negativo\" na ironia (%)",
       caption = "Sem contexto; três ordens juntas. Linha cinza: distância até a diagonal = quanto o modelo separa a ironia do viés.") +
  theme_bw(base_size = 15) + theme(legend.position = "top", panel.grid.minor = element_blank(), plot.caption = element_text(size = 11))
ggsave("ironia_discrim-20261122.pdf", f, width = 16, height = 6.3)
ggsave("ironia_discrim-20261122.png", f, width = 16, height = 6.3, dpi = 100)
