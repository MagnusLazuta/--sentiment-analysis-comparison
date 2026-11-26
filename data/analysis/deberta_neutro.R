# DeBERTa-NLI on neutral MT-CSD comments: where do its answers go as context grows? (2026-11-23)
# Source: ablation run 20261118-094702-838561 (eval/ablation.csv); fixed cohort of MT-CSD items with
# >= 4 context turns (k = 0..4, same items at every k). Only comments whose ground truth is neutral,
# split by the tone of the conversation (conversa_tom-20261122.csv: majority of annotated stances of
# the context turns; ties left out). DeBERTa-NLI has a single "variant" (none).
# Outputs: deberta_neutro-20261123.{txt,pdf,png}
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
tom <- read_csv("conversa_tom-20261122.csv", show_col_types = FALSE) |>
  filter(dataset == "mtcsd", maioria %in% c("negativo", "neutro", "positivo")) |> select(id, tom = maioria)
lv <- c("negative", "neutral", "positive"); pt <- c("negativo", "neutro", "positivo")
d <- read_csv("../results/20261118-094702-838561/eval/ablation.csv", show_col_types = FALSE) |>
  filter(dataset == "mtcsd", model == "deberta-nli") |>
  group_by(id) |> filter(max(k) >= 4) |> ungroup() |> filter(k <= 4, gold == "neutral") |>
  inner_join(tom, by = "id")
nn <- d |> distinct(id, tom) |> count(tom)
s <- d |> mutate(pred = factor(pt[match(pred, lv)], pt)) |> count(tom, k, pred, .drop = FALSE) |>
  group_by(tom, k) |> mutate(pct = 100 * n / sum(n)) |> ungroup() |> left_join(nn |> rename(N = n), by = "tom") |>
  mutate(fem = c(neutro = "neutra", negativo = "negativa", positivo = "positiva")[tom],
         painel = sprintf("conversa %s (n = %d)", fem, N),
         painel = factor(painel, unique(painel[order(match(tom, c("neutro", "negativo", "positivo")))])))
sink("deberta_neutro-20261123.txt")
cat("DeBERTa-NLI, MT-CSD neutral comments, cohort >= 4 turns: % of answers by k and conversation tone.\n")
print(s |> mutate(pct = round(pct, 1)) |> select(tom, N, k, pred, pct) |>
        pivot_wider(names_from = pred, values_from = pct) |> as.data.frame(), row.names = FALSE)
sink()
p <- ggplot(s, aes(factor(k), pct, fill = pred)) + geom_col(width = 0.75) +
  geom_text(data = s |> filter(pct >= 8), aes(label = sprintf("%.0f", pct)),
            position = position_stack(vjust = 0.5), colour = "white", size = 5) +
  facet_wrap(~ painel) +
  scale_fill_manual(values = c(negativo = "#C0392B", neutro = "#95A5A6", positivo = "#2471A3"), name = "Resposta") +
  labs(x = "Turnos de contexto (k)", y = "% das respostas",
       caption = "DeBERTa-NLI, comentários neutros do MT-CSD, mesmos itens em todo k (pelo menos 4 turnos). Cinza = acerto.") +
  theme_minimal(base_size = 18) +
  theme(legend.position = "top", panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
        strip.text = element_text(face = "bold"), plot.caption = element_text(size = 12))
ggsave("deberta_neutro-20261123.pdf", p, width = 15, height = 6)
ggsave("deberta_neutro-20261123.png", p, width = 15, height = 6, dpi = 100)

# Simpler view for the slide (2026-11-23): one panel, % "neutral" answers by k per conversation tone,
# with an end label saying where the answers went at k = 4.
l <- s |> filter(pred == "neutro")
e <- s |> filter(k == 4) |> select(tom, pred, pct) |> pivot_wider(names_from = pred, values_from = pct) |>
  mutate(txt = case_when(tom == "negativo" ~ sprintf("%.0f%% viram \"contra\"", negativo),
                         tom == "positivo" ~ sprintf("%.0f%% viram \"a favor\"", positivo),
                         TRUE ~ sprintf("%.0f%% \"contra\" + %.0f%% \"a favor\"", negativo, positivo)))
l <- l |> left_join(e |> select(tom, txt), by = "tom") |>
  mutate(lbl = sprintf("conversa %s (n = %d)", c(neutro = "neutra", negativo = "negativa", positivo = "positiva")[tom], N))
cc <- c("#7F8C8D", "#C0392B", "#2471A3"); names(cc) <- unique(l$lbl[order(match(l$tom, c("neutro", "negativo", "positivo")))])
l <- l |> mutate(lbl = factor(lbl, names(cc)))
le <- l |> filter(k == 4) |> mutate(y = pct + c(neutro = 0, negativo = 4, positivo = -4)[tom])
p2 <- ggplot(l, aes(k, pct, colour = lbl)) +
  geom_line(linewidth = 1.8) + geom_point(size = 4) +
  geom_text(data = le, aes(y = y, label = txt), hjust = -0.08, size = 6, show.legend = FALSE) +
  scale_colour_manual(values = cc, name = NULL) +
  scale_x_continuous(breaks = 0:4, expand = expansion(mult = c(0.03, 0.55))) +
  scale_y_continuous(limits = c(0, 100)) +
  labs(x = "Turnos de contexto (k)", y = "Responde \"neutro\" (%)",
       caption = "DeBERTa-NLI, comentários neutros do MT-CSD, mesmos itens em todo k (pelo menos 4 turnos). \"Neutro\" é o acerto.") +
  theme_bw(base_size = 20) +
  theme(legend.position = "top", panel.grid.minor = element_blank(), plot.caption = element_text(size = 13))
ggsave("deberta_neutro_linhas-20261123.pdf", p2, width = 14, height = 6)
ggsave("deberta_neutro_linhas-20261123.png", p2, width = 14, height = 6, dpi = 100)

# Third view for the slide (2026-11-23): one panel per conversation tone, only k = 0 vs k = 4,
# 100% bars of the answers (a single measure), panel titles state the takeaway.
tt <- c(neutro = "Conversa neutra: polariza", negativo = "Conversa negativa: vira \"contra\"",
        positivo = "Conversa positiva: vira \"a favor\"")
b <- s |> filter(k %in% c(0, 4)) |>
  mutate(kk = factor(ifelse(k == 0, "sem contexto", "4 turnos"), c("sem contexto", "4 turnos")),
         painel = factor(sprintf("%s\n(n = %d)", tt[tom], N),
                         sprintf("%s\n(n = %d)", tt[c("neutro", "negativo", "positivo")],
                                 nn$n[match(c("neutro", "negativo", "positivo"), nn$tom)])))
p3 <- ggplot(b, aes(kk, pct, fill = pred)) + geom_col(width = 0.62) +
  geom_text(data = b |> filter(pct >= 9), aes(label = sprintf("%.0f%%", pct)),
            position = position_stack(vjust = 0.5), colour = "white", size = 6.5, fontface = "bold") +
  facet_wrap(~ painel) +
  scale_fill_manual(values = c(negativo = "#C0392B", neutro = "#95A5A6", positivo = "#2471A3"), name = "Resposta", labels = c(negativo = "contra", neutro = "neutro", positivo = "a favor")) +
  scale_y_continuous(breaks = NULL) +
  labs(x = NULL, y = NULL,
       caption = "DeBERTa-NLI, comentários neutros do MT-CSD (o certo é \"neutro\"), mesmos itens nas duas barras.") +
  theme_minimal(base_size = 20) +
  theme(legend.position = "top", panel.grid = element_blank(), strip.text = element_text(face = "bold", size = 18),
        plot.caption = element_text(size = 13))
ggsave("deberta_neutro_barras-20261123.pdf", p3, width = 14, height = 6.2)
ggsave("deberta_neutro_barras-20261123.png", p3, width = 14, height = 6.2, dpi = 100)
