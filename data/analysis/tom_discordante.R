# Comments whose stance differs from the tone of the conversation: are they pulled to the
# conversation's tone when the model sees more context? (2026-11-22)
#
# MT-CSD only (the context turns have human stance labels). Tone of the conversation =
# majority of the annotated stances of the context turns (conversa_tom.py; ties and
# threads without annotated turns are left out). An item "concorda" when its own gold equals
# that tone, "discorda" otherwise.
# By turn k = 0..4 on a fixed cohort (items with >= 4 context turns), from the ablation run
# 20261118-094702-838561. All option orders pooled.
# Measures per model x condition x group: % of answers equal to the conversation tone, and
# accuracy. 95% CI: bootstrap by thread (13 threads), 2,000 resamples, seed 5.
# Outputs: tom_discordante-20261122.{txt,pdf,png}
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
set.seed(5)
B <- 2000
ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
lab <- c("twitter-roberta", "DeBERTa-NLI", "Qwen3.5-4B", "Mistral-7B", "Kev-4B", "Laya")
en <- c(negativo = "negative", neutro = "neutral", positivo = "positive")

tom <- read_csv("conversa_tom-20261122.csv", show_col_types = FALSE) |>
  filter(dataset == "mtcsd", maioria %in% names(en)) |> transmute(id, tom = maioria)
thr <- read_csv("mtcsd_threads-20261117-181106-838446.csv", show_col_types = FALSE) |> select(id, thread)
abl <- read_csv("../results/20261118-094702-838561/eval/ablation.csv", show_col_types = FALSE) |>
  filter(dataset == "mtcsd", model %in% ord)

# by turn: fixed cohort of items with >= 4 context turns, k = 0..4 (same items at every k)
d <- abl |> group_by(model, variant, id) |> filter(max(k) >= 4) |> ungroup() |> filter(k <= 4) |>
  mutate(cond = k) |>
  inner_join(tom, by = "id") |> left_join(thr, by = "id") |>
  mutate(tom_en = en[tom],
         grupo = ifelse(gold == tom_en, "comentário concorda com a conversa", "comentário discorda da conversa"),
         segue = pred == tom_en,
         model = factor(model, ord, lab))

cboot <- function(y, cl) {
  g <- split(y, cl)
  m <- replicate(B, mean(unlist(g[sample(length(g), replace = TRUE)])))
  c(100 * mean(y), 100 * quantile(m, c(0.025, 0.975)))
}
res <- d |> group_by(model, cond, grupo) |>
  group_modify(~ {
    a <- cboot(.x$segue, .x$thread); b <- cboot(.x$correct, .x$thread)
    tibble(n = n_distinct(.x$id),
           medida = c("Resposta = tom da conversa (%)", "Acurácia (%)"),
           v = c(a[1], b[1]), lo = c(a[2], b[2]), hi = c(a[3], b[3]))
  }) |> ungroup()

sink("tom_discordante-20261122.txt")
cat("MT-CSD. Items with a clear conversation tone (majority of annotated context stances).\n")
print(d |> distinct(id, tom, grupo) |> count(tom, grupo) |> as.data.frame(), row.names = FALSE)
cat("\nAll option orders pooled; [95% CI] by thread bootstrap.\n")
print(res |> mutate(cell = sprintf("%.1f [%.1f; %.1f]", v, lo, hi)) |>
        select(medida, grupo, model, cond, cell) |>
        pivot_wider(names_from = cond, values_from = cell) |> arrange(medida, grupo, model) |>
        as.data.frame(), row.names = FALSE)
sink()

nn <- d |> distinct(id, grupo) |> count(grupo)
res <- res |> left_join(nn |> rename(N = n), by = "grupo") |>
  mutate(grupo_l = sprintf("%s (n = %d)", grupo, N),
         medida = factor(medida, c("Resposta = tom da conversa (%)", "Acurácia (%)")))
pd <- position_dodge(width = 0.2)
p <- ggplot(res, aes(cond, v, colour = grupo_l, group = grupo_l)) +
  geom_line(linewidth = 1.1, position = pd) +
  geom_point(size = 2.5, position = pd) +  # CIs kept in the txt, not drawn (2026-11-22)
  facet_grid(medida ~ model, switch = "y") +
  scale_colour_manual(values = c("#2B4C7E", "#C0392B"), name = NULL) +
  scale_y_continuous(limits = c(0, 100)) +
  scale_x_continuous(breaks = 0:4) +
  labs(x = "Turnos de contexto (k)", y = NULL,
       title = "MT-CSD: o comentário é puxado para o tom da conversa?",
       caption = paste("Mesmos itens em todo k (pelo menos 4 turnos). Tom da conversa: maioria das posturas anotadas nos turnos anteriores (empates fora).",
                       "Três ordens de opções juntas.")) +
  theme_bw(base_size = 15) +
  theme(legend.position = "top", strip.placement = "outside", panel.grid.minor = element_blank(),
        plot.caption = element_text(size = 11))
ggsave("tom_discordante-20261122.pdf", p, width = 17, height = 9)
ggsave("tom_discordante-20261122.png", p, width = 17, height = 9, dpi = 100)
