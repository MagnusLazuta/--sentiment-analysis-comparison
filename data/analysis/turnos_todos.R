# Accuracy as context grows turn by turn, on ALL items (2026-11-22).
#
# Source: context ablation run 20261118-094702-838561 (eval/ablation.csv): every MT-CSD and
# MUStARD++ item evaluated with the last k context turns, k = 0 (no context) .. K (whole
# conversation). Order v1 (twitter-roberta: none); NanoJev excluded (negative control).
#
# Two views, per dataset group (MT-CSD, MUStARD++ non-ironic, MUStARD++ ironic):
#  A. accuracy at each k over every item that has >= k turns (n changes with k);
#  B. paired difference to k = 0 on those same items (removes the change of item mix),
#     95% CI by bootstrap: by item (MUStARD++) or by thread (MT-CSD, 13 threads).
#     2,000 resamples, seed 5.
# k is shown while n >= 30 items.
# Outputs (this folder): turnos_todos-20261122.txt / .pdf / .png
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
set.seed(5)
B <- 2000
abl <- read_csv("../results/20261118-094702-838561/eval/ablation.csv", show_col_types = FALSE)
thr <- read_csv("mtcsd_threads-20261117-181106-838446.csv", show_col_types = FALSE) |> select(id, thread)

d <- abl |>
  filter(model != "nanojev", variant %in% c("v1", "none")) |>
  mutate(grp = case_when(dataset == "mtcsd" ~ "MT-CSD",
                         sarcastic == 1 ~ "MUStARD++ irônico",
                         TRUE ~ "MUStARD++ não irônico")) |>
  left_join(thr, by = "id") |>
  mutate(cl = ifelse(dataset == "mtcsd", thread, id))

base <- d |> filter(k == 0) |> select(model, grp, id, y0 = correct)
dd <- d |> inner_join(base, by = c("model", "grp", "id"))

boot_diff <- function(x) {  # x: cl, y, y0 ; cluster bootstrap of mean(y - y0)
  g <- split(x$y - x$y0, x$cl)
  m <- replicate(B, { s <- sample(length(g), replace = TRUE); mean(unlist(g[s])) })
  tibble(dif = 100 * mean(x$y - x$y0), lo = 100 * quantile(m, 0.025), hi = 100 * quantile(m, 0.975))
}

res <- dd |> group_by(grp, model, k) |>
  filter(n() >= 30) |>
  group_modify(~ bind_cols(tibble(n = nrow(.x), acc = 100 * mean(.x$correct)),
                           if (.y$k == 0) tibble(dif = 0, lo = 0, hi = 0)
                           else boot_diff(transmute(.x, cl, y = correct, y0)))) |>
  ungroup()

sink("turnos_todos-20261122.txt")
cat("Accuracy (%) by number of context turns k, all items with >= k turns; dif = paired",
    "difference to k = 0 on the same items [95% CI]. v1/none, no NanoJev.\n\n")
for (g in unique(res$grp)) {
  cat("==", g, "\n")
  res |> filter(grp == g) |>
    mutate(cell = sprintf("%.1f (%+.1f [%.1f; %.1f])", acc, dif, lo, hi)) |>
    select(model, k, n, cell) |> pivot_wider(names_from = model, values_from = cell) |>
    as.data.frame() |> print(row.names = FALSE)
  cat("\n")
}
sink()

ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
lab <- c("twitter-roberta", "DeBERTa-NLI", "Qwen3.5-4B", "Mistral-7B", "Kev-4B", "Laya")
cols <- c("#6BAED6", "#08519C", "#9E9AC8", "#54278F", "#FD8D3C", "#A63603")
pd <- res |> mutate(model = factor(model, ord, lab),
                    grp = factor(grp, c("MT-CSD", "MUStARD++ não irônico", "MUStARD++ irônico")))
long <- bind_rows(
  pd |> transmute(grp, model, k, panel = "Acurácia (%)", v = acc, lo = NA, hi = NA),
  pd |> transmute(grp, model, k, panel = "Diferença para k = 0 (pontos)", v = dif, lo, hi))
nlab <- pd |> distinct(grp, k, n) |> group_by(grp, k) |> summarise(n = first(n), .groups = "drop")
p <- ggplot(long, aes(k, v, colour = model)) +
  geom_hline(data = tibble(panel = "Diferença para k = 0 (pontos)", y = 0),
             aes(yintercept = y), linetype = 2, colour = "grey50") +
  geom_ribbon(aes(ymin = lo, ymax = hi, fill = model), alpha = 0.10, colour = NA, show.legend = FALSE) +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  facet_grid(panel ~ grp, scales = "free", switch = "y") +
  scale_colour_manual(values = cols, name = NULL) + scale_fill_manual(values = cols) +
  scale_x_continuous(breaks = 0:12) +
  labs(x = "Turnos de contexto (k)", y = NULL,
       caption = paste("n por k:", paste(sapply(levels(pd$grp), function(g)
         paste0(g, " ", paste(nlab$n[nlab$grp == g], collapse = "/"))), collapse = "; "))) +
  theme_bw(base_size = 18) +
  theme(legend.position = "top", strip.placement = "outside", panel.grid.minor = element_blank(),
        plot.caption = element_text(size = 12))
ggsave("turnos_todos-20261122.pdf", p, width = 16, height = 9)
ggsave("turnos_todos-20261122.png", p, width = 16, height = 9, dpi = 110)

# Accuracy only, one facet per model, one line per dataset group (2026-11-22).
gcols <- c("MT-CSD" = "#2B4C7E", "MUStARD++ não irônico" = "#E08214", "MUStARD++ irônico" = "#7A7A7A")
p2 <- ggplot(pd, aes(k, acc, colour = grp)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2.2) +
  facet_wrap(~model, nrow = 2) +
  scale_colour_manual(values = gcols, name = NULL) +
  scale_x_continuous(breaks = 0:9) + scale_y_continuous(limits = c(0, 80)) +
  labs(x = "Turnos de contexto (k)", y = "Acurácia (%)",
       caption = paste("Todos os itens com pelo menos k turnos; ordem v1. n por k:",
                       paste(sapply(levels(pd$grp), function(g)
                         paste0(g, " ", paste(nlab$n[nlab$grp == g], collapse = "/"))), collapse = "; "))) +
  theme_bw(base_size = 18) +
  theme(legend.position = "top", panel.grid.minor = element_blank(), plot.caption = element_text(size = 11))
ggsave("turnos_modelos-20261122.pdf", p2, width = 16, height = 9)
ggsave("turnos_modelos-20261122.png", p2, width = 16, height = 9, dpi = 110)

# Fixed cohort: only items with at least 4 context turns, k = 0..4, so every point has the
# same items (2026-11-22). n: MT-CSD 456, MUStARD++ non-ironic 293, ironic 292.
coh <- d |> group_by(model, grp, id) |> filter(max(k) >= 4) |> ungroup() |> filter(k <= 4)
ch <- coh |> group_by(grp, model, k) |> summarise(n = n(), acc = 100 * mean(correct), .groups = "drop") |>
  mutate(model = factor(model, ord, lab),
         grp = factor(grp, c("MT-CSD", "MUStARD++ não irônico", "MUStARD++ irônico")))
stopifnot(ch |> group_by(grp, model) |> summarise(u = n_distinct(n), .groups = "drop") |> pull(u) |> max() == 1)
cn <- ch |> distinct(grp, n)
sink("turnos_coorte-20261122.txt")
cat("Accuracy (%) by k, fixed cohort of items with >= 4 context turns (same items at every k). v1/none.\n")
print(as.data.frame(cn), row.names = FALSE)
ch |> select(grp, model, k, acc) |> mutate(acc = round(acc, 1)) |>
  pivot_wider(names_from = k, values_from = acc, names_prefix = "k") |> as.data.frame() |> print(row.names = FALSE)
sink()
p3 <- ggplot(ch, aes(k, acc, colour = grp)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2.4) +
  facet_wrap(~model, nrow = 2) +
  scale_colour_manual(values = gcols, name = NULL,
                      labels = setNames(paste0(cn$grp, " (n = ", cn$n, ")"), cn$grp)) +
  scale_x_continuous(breaks = 0:4) + scale_y_continuous(limits = c(0, 80)) +
  labs(x = "Turnos de contexto (k)", y = "Acurácia (%)",
       caption = "Mesmos itens em todos os pontos: só os que têm pelo menos 4 turnos de contexto; ordem v1.") +
  theme_bw(base_size = 18) +
  theme(legend.position = "top", panel.grid.minor = element_blank(), plot.caption = element_text(size = 12))
ggsave("turnos_coorte-20261122.pdf", p3, width = 16, height = 9)
ggsave("turnos_coorte-20261122.png", p3, width = 16, height = 9, dpi = 110)
