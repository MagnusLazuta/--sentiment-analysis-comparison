# Kev-4B confusion matrices by number of context turns k = 0..4 (2026-11-22).
# Fixed cohort: items with >= 4 context turns (same items at every k), order v1.
# Source: ablation run 20261118-094702-838561 (eval/ablation.csv).
# Outputs: turnos_confusao_kev-20261122.{txt,pdf,png}
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
abl <- read_csv("../results/20261118-094702-838561/eval/ablation.csv", show_col_types = FALSE)
lv <- c("negative", "neutral", "positive"); pt <- c("negativo", "neutro", "positivo")
grps <- c("MT-CSD", "MUStARD++ não irônico", "MUStARD++ irônico")
q <- abl |> filter(model == "kev-4b", variant == "v1") |>
  mutate(grp = case_when(dataset == "mtcsd" ~ grps[1], sarcastic == 1 ~ grps[3], TRUE ~ grps[2])) |>
  group_by(grp, id) |> filter(max(k) >= 4) |> ungroup() |> filter(k <= 4) |>
  mutate(gold = factor(gold, lv, pt), pred = factor(pred, lv, pt), grp = factor(grp, grps))
cm <- q |> count(grp, k, gold, pred, .drop = FALSE) |>
  group_by(grp, k, gold) |> mutate(row = sum(n), pct = ifelse(row > 0, 100 * n / row, NA)) |> ungroup() |>
  filter(row > 0)

# changes between consecutive k (same item)
w <- q |> select(grp, id, gold, k, pred) |> pivot_wider(names_from = k, values_from = pred, names_prefix = "k")
chg <- bind_rows(lapply(1:4, function(j) {
  a <- w[[paste0("k", j - 1)]]; b <- w[[paste0("k", j)]]
  tibble(grp = w$grp, gold = w$gold, de = a, para = b, passo = paste0(j - 1, "->", j))
})) |> filter(de != para) |> count(grp, passo, de, para)
tot <- bind_rows(lapply(1:4, function(j) {
  tibble(grp = w$grp, passo = paste0(j - 1, "->", j),
         mudou = w[[paste0("k", j - 1)]] != w[[paste0("k", j)]],
         antes = w[[paste0("k", j - 1)]] == w$gold, depois = w[[paste0("k", j)]] == w$gold)
})) |> group_by(grp, passo) |>
  summarise(n = n(), mudou = sum(mudou), errado_para_certo = sum(!antes & depois),
            certo_para_errado = sum(antes & !depois), .groups = "drop")

sink("turnos_confusao_kev-20261122.txt")
cat("Kev-4B, v1, fixed cohort (>= 4 context turns). Rows = gold, cols = prediction, counts.\n")
for (g in grps) for (kk in 0:4) {
  cat("\n==", g, " k =", kk, " acc =",
      sprintf("%.1f%%", 100 * mean(q$gold[q$grp == g & q$k == kk] == q$pred[q$grp == g & q$k == kk])), "\n")
  print(cm |> filter(grp == g, k == kk) |> select(gold, pred, n) |>
          pivot_wider(names_from = pred, values_from = n) |> as.data.frame(), row.names = FALSE)
}
cat("\n\nChanges between consecutive k (same items):\n")
print(as.data.frame(tot), row.names = FALSE)
cat("\nDirection of changes (de -> para):\n")
print(as.data.frame(chg |> arrange(grp, passo, desc(n))), row.names = FALSE)
sink()

# figure without the ironic group (only negative gold; kept in the txt) (2026-11-22)
cmf <- cm |> filter(grp != grps[3]) |> mutate(grp = droplevels(grp))
p <- ggplot(cmf, aes(pred, gold, fill = pct)) +
  geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%d\n%.0f%%", n, pct), colour = pct > 55), size = 3.6, lineheight = 0.9) +
  facet_grid(grp ~ paste("k =", k)) +
  scale_fill_gradient(low = "#F4F6F9", high = "#2B4C7E", limits = c(0, 100), name = "% da linha") +
  scale_colour_manual(values = c(`FALSE` = "#23262B", `TRUE` = "white"), guide = "none") +
  scale_y_discrete(limits = rev(pt)) +
  labs(x = "Resposta do modelo", y = "Ground-truth",
       caption = "Kev-4B, ordem v1. Mesmos itens em todo k (pelo menos 4 turnos): MT-CSD 456, MUStARD++ não irônico 293. Células: contagem e % da linha.") +
  theme_bw(base_size = 15) +
  theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 30, hjust = 1),
        plot.caption = element_text(size = 10))
ggsave("turnos_confusao_kev-20261122.pdf", p, width = 16, height = 7)
ggsave("turnos_confusao_kev-20261122.png", p, width = 16, height = 7, dpi = 110)

# Same matrices for the three option orders, stacked with patchwork, one colour per order
# (2026-11-22). v1 = negative, neutral, positive; v2 = positive, neutral, negative;
# v3 = neutral, positive, negative. Output: turnos_confusao_kev_ordens-20261122.{pdf,png}
suppressPackageStartupMessages(library(patchwork))
qa <- abl |> filter(model == "kev-4b", dataset == "mtcsd" | sarcastic == 0) |>
  mutate(grp = ifelse(dataset == "mtcsd", grps[1], grps[2])) |>
  group_by(grp, id) |> filter(max(k) >= 4) |> ungroup() |> filter(k <= 4) |>
  mutate(gold = factor(gold, lv, pt), pred = factor(pred, lv, pt), grp = factor(grp, grps[1:2]))
cma <- qa |> count(variant, grp, k, gold, pred, .drop = FALSE) |>
  group_by(variant, grp, k, gold) |> mutate(row = sum(n), pct = ifelse(row > 0, 100 * n / row, NA)) |>
  ungroup() |> filter(row > 0)
acc <- qa |> group_by(variant, grp, k) |> summarise(a = 100 * mean(gold == pred), .groups = "drop")
ordem <- c(v1 = "v1: negativo, neutro, positivo", v2 = "v2: positivo, neutro, negativo",
           v3 = "v3: neutro, positivo, negativo")
hi <- c(v1 = "#2B4C7E", v2 = "#B35806", v3 = "#1B7837")
# cells whose row % changed by >= LIM points from the previous k (red outline)
LIM <- 10
cma <- cma |> arrange(variant, grp, gold, pred, k) |> group_by(variant, grp, gold, pred) |>
  mutate(d = pct - lag(pct), big = !is.na(d) & abs(d) >= LIM) |> ungroup()
one <- function(v) {
  x <- cma |> filter(variant == v) |>
    left_join(acc |> filter(variant == v), by = c("variant", "grp", "k")) |>
    mutate(col = sprintf("k = %d (acc. %.1f%%)", k, a))
  ggplot(x, aes(pred, gold, fill = pct)) +
    geom_tile(colour = "white") +
    geom_tile(data = function(z) filter(z, big), fill = NA, colour = "#D7191C", linewidth = 1.3,
              width = 0.94, height = 0.94) +
    geom_text(aes(label = sprintf("%d\n%.0f%%", n, pct), colour = pct > 55), size = 3.2, lineheight = 0.9) +
    facet_grid(grp ~ k, labeller = labeller(k = function(z) paste("k =", z))) +
    scale_fill_gradient(low = "#F6F7F8", high = hi[[v]], limits = c(0, 100), name = "% da linha") +
    scale_colour_manual(values = c(`FALSE` = "#23262B", `TRUE` = "white"), guide = "none") +
    scale_y_discrete(limits = rev(pt)) +
    labs(title = paste("Ordem", ordem[[v]]), x = NULL, y = "Ground-truth",
         subtitle = paste(acc |> filter(variant == v) |> group_by(grp) |>
                            summarise(s = paste0(grp[1], ": ", paste(sprintf("%.1f", a), collapse = " / "))) |>
                            pull(s), collapse = "   |   ")) +
    theme_bw(base_size = 13) +
    theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title = element_text(colour = hi[[v]], face = "bold"),
          plot.subtitle = element_text(size = 11, colour = "grey30"))
}
pp <- (one("v1") / one("v2") / one("v3")) +
  plot_annotation(caption = paste("Kev-4B. Mesmos itens em todo k (pelo menos 4 turnos): MT-CSD 456,",
                                  "MUStARD++ não irônico 293. Subtítulo: acurácia de k = 0 a 4. Eixo x: resposta do modelo.",
                                  "\nContorno vermelho: célula cujo % da linha mudou 10 pontos ou mais em relação ao turno anterior."))
ggsave("turnos_confusao_kev_ordens-20261122.pdf", pp, width = 16, height = 20)
ggsave("turnos_confusao_kev_ordens-20261122.png", pp, width = 16, height = 20, dpi = 100)

# One image per dataset: the three option orders stacked (2026-11-22).
# Outputs: turnos_confusao_kev_ordens_{mtcsd,mustard}-20261122.{pdf,png}
one_ds <- function(v, g) {
  x <- cma |> filter(variant == v, grp == g) |>
    left_join(acc |> filter(variant == v, grp == g), by = c("variant", "grp", "k"))
  ggplot(x, aes(pred, gold, fill = pct)) +
    geom_tile(colour = "white") +
    geom_tile(data = function(z) filter(z, big), fill = NA, colour = "#D7191C", linewidth = 1.4,
              width = 0.94, height = 0.94) +
    geom_text(aes(label = sprintf("%d\n%.0f%%", n, pct), colour = pct > 55), size = 4, lineheight = 0.9) +
    facet_grid(. ~ k, labeller = labeller(k = function(z) paste("k =", z))) +
    scale_fill_gradient(low = "#F6F7F8", high = hi[[v]], limits = c(0, 100), name = "% da linha") +
    scale_colour_manual(values = c(`FALSE` = "#23262B", `TRUE` = "white"), guide = "none") +
    scale_y_discrete(limits = rev(pt)) +
    labs(title = paste("Ordem", ordem[[v]]), x = NULL, y = "Ground-truth",
         subtitle = paste("Acurácia de k = 0 a 4:",
                          paste(sprintf("%.1f", acc$a[acc$variant == v & acc$grp == g]), collapse = " / "))) +
    theme_bw(base_size = 15) +
    theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title = element_text(colour = hi[[v]], face = "bold"),
          plot.subtitle = element_text(size = 13, colour = "grey30"))
}
for (g in grps[1:2]) {
  nn <- n_distinct(qa$id[qa$grp == g])
  pg <- (one_ds("v1", g) / one_ds("v2", g) / one_ds("v3", g)) +
    plot_annotation(title = paste0(g, ", Kev-4B (n = ", nn, ", mesmos itens em todo k)"),
                    caption = paste("Eixo x: resposta do modelo. Células: contagem e % da linha.",
                                    "\nContorno vermelho: % da linha mudou 10 pontos ou mais em relação ao turno anterior."),
                    theme = theme(plot.title = element_text(size = 20, face = "bold")))
  f <- ifelse(g == grps[1], "mtcsd", "mustard")
  ggsave(sprintf("turnos_confusao_kev_ordens_%s-20261122.pdf", f), pg, width = 16, height = 13)
  ggsave(sprintf("turnos_confusao_kev_ordens_%s-20261122.png", f), pg, width = 16, height = 13, dpi = 100)
}

# Single ggplot per dataset: rows = option order (colour), columns = k (2026-11-22).
# Outputs: turnos_confusao_kev_facet_{mtcsd,mustard}-20261122.{pdf,png}
for (g in grps[1:2]) {
  x <- cma |> filter(grp == g) |>
    left_join(acc |> filter(grp == g), by = c("variant", "grp", "k")) |>
    mutate(ord = factor(ordem[variant], ordem),
           kk = factor(paste("k =", k), paste("k =", 0:4)))
  accl <- acc |> filter(grp == g) |> mutate(ord = factor(ordem[variant], ordem),
                                            kk = factor(paste("k =", k), paste("k =", 0:4)))
  nn <- n_distinct(qa$id[qa$grp == g])
  pf <- ggplot(x, aes(pred, gold)) +
    geom_tile(aes(fill = variant, alpha = pct), colour = "white") +
    geom_tile(data = function(z) filter(z, big), fill = NA, colour = "#D7191C", linewidth = 1.4,
              width = 0.94, height = 0.94) +
    geom_text(aes(label = sprintf("%d\n%.0f%%", n, pct), colour = pct > 55), size = 3.8, lineheight = 0.9) +
    geom_text(data = accl, aes(x = 2, y = 3.62, label = sprintf("acc. %.1f%%", a)),
              inherit.aes = FALSE, size = 3.8, colour = "grey25", vjust = 0) +
    facet_grid(ord ~ kk) +
    scale_fill_manual(values = hi, guide = "none") +
    scale_alpha_continuous(range = c(0.05, 1), limits = c(0, 100), name = "% da linha") +
    scale_colour_manual(values = c(`FALSE` = "#23262B", `TRUE` = "white"), guide = "none") +
    scale_y_discrete(limits = rev(pt), expand = expansion(add = c(0.5, 0.75))) +
    labs(title = paste0(g, ", Kev-4B (n = ", nn, ", mesmos itens em todo k)"),
         x = "Resposta do modelo", y = "Ground-truth",
         caption = paste("Linhas: ordem das opções; colunas: turnos de contexto. Células: contagem e % da linha.",
                         "\nContorno vermelho: % da linha mudou 10 pontos ou mais em relação ao turno anterior.")) +
    theme_bw(base_size = 15) +
    theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 30, hjust = 1),
          strip.text.y = element_text(size = 11), legend.position = "right",
          plot.title = element_text(face = "bold"))
  f <- ifelse(g == grps[1], "mtcsd", "mustard")
  ggsave(sprintf("turnos_confusao_kev_facet_%s-20261122.pdf", f), pf, width = 17, height = 11)
  ggsave(sprintf("turnos_confusao_kev_facet_%s-20261122.png", f), pf, width = 17, height = 11, dpi = 100)
}

# MT-CSD split by target (Tesla, SpaceX), same layout as the facet plot (2026-11-22).
# Outputs: turnos_confusao_kev_facet_mtcsd_{tesla,spacex}-20261122.{pdf,png}
qt <- qa |> filter(grp == grps[1]) |> mutate(alvo = ifelse(grepl(":Tesla:", id), "Tesla", "SpaceX"))
cmt <- qt |> count(alvo, variant, k, gold, pred, .drop = FALSE) |>
  group_by(alvo, variant, k, gold) |> mutate(row = sum(n), pct = ifelse(row > 0, 100 * n / row, NA)) |>
  ungroup() |> filter(row > 0) |>
  arrange(alvo, variant, gold, pred, k) |> group_by(alvo, variant, gold, pred) |>
  mutate(d = pct - lag(pct), big = !is.na(d) & abs(d) >= LIM) |> ungroup()
acct <- qt |> group_by(alvo, variant, k) |> summarise(a = 100 * mean(gold == pred), .groups = "drop")
sink("turnos_confusao_kev_alvo-20261122.txt")
cat("Kev-4B, MT-CSD by target, fixed cohort (>= 4 turns). Accuracy by order and k.\n")
print(qt |> distinct(alvo, id, gold) |> count(alvo, gold) |> as.data.frame(), row.names = FALSE)
print(acct |> mutate(a = round(a, 1)) |> pivot_wider(names_from = k, values_from = a, names_prefix = "k") |>
        as.data.frame(), row.names = FALSE)
sink()
for (t in c("Tesla", "SpaceX")) {
  x <- cmt |> filter(alvo == t) |>
    mutate(ord = factor(ordem[variant], ordem), kk = factor(paste("k =", k), paste("k =", 0:4)))
  accl <- acct |> filter(alvo == t) |>
    mutate(ord = factor(ordem[variant], ordem), kk = factor(paste("k =", k), paste("k =", 0:4)))
  nn <- n_distinct(qt$id[qt$alvo == t])
  pf <- ggplot(x, aes(pred, gold)) +
    geom_tile(aes(fill = variant, alpha = pct), colour = "white") +
    geom_tile(data = function(z) filter(z, big), fill = NA, colour = "#D7191C", linewidth = 1.4,
              width = 0.94, height = 0.94) +
    geom_text(aes(label = sprintf("%d\n%.0f%%", n, pct), colour = pct > 55), size = 3.8, lineheight = 0.9) +
    geom_text(data = accl, aes(x = 2, y = 3.62, label = sprintf("acc. %.1f%%", a)),
              inherit.aes = FALSE, size = 3.8, colour = "grey25", vjust = 0) +
    facet_grid(ord ~ kk) +
    scale_fill_manual(values = hi, guide = "none") +
    scale_alpha_continuous(range = c(0.05, 1), limits = c(0, 100), name = "% da linha") +
    scale_colour_manual(values = c(`FALSE` = "#23262B", `TRUE` = "white"), guide = "none") +
    scale_y_discrete(limits = rev(pt), expand = expansion(add = c(0.5, 0.75))) +
    labs(title = paste0("MT-CSD, alvo ", t, ", Kev-4B (n = ", nn, ", mesmos itens em todo k)"),
         x = "Resposta do modelo", y = "Ground-truth",
         caption = paste("Linhas: ordem das opções; colunas: turnos de contexto. Células: contagem e % da linha.",
                         "\nContorno vermelho: % da linha mudou 10 pontos ou mais em relação ao turno anterior.")) +
    theme_bw(base_size = 15) +
    theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 30, hjust = 1),
          strip.text.y = element_text(size = 11), plot.title = element_text(face = "bold"))
  ggsave(sprintf("turnos_confusao_kev_facet_mtcsd_%s-20261122.pdf", tolower(t)), pf, width = 17, height = 11)
  ggsave(sprintf("turnos_confusao_kev_facet_mtcsd_%s-20261122.png", tolower(t)), pf, width = 17, height = 11, dpi = 100)
}
