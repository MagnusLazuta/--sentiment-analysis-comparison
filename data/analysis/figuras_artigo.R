# Paper figures in English, sized for the SBC text width (2026-11-26).
#
# Rebuilds the ten result figures of the paper with English labels and larger text. Data logic is
# copied from the scripts that made the slide versions (Portuguese), so values are unchanged:
#   fig_contexto      no script before; acc. no context / previous turn / whole conversation, order v1
#   fig_turnos        turnos_mtcsd_slides.R (C)  accuracy and macro-F1 by k, fixed MT-CSD cohort
#   fig_qwen_confusao turnos_mtcsd_slides.R (B)  Qwen3.5-4B confusion matrices by k
#   fig_tom           tom_discordante.R          following the conversation tone, by k
#   fig_deberta       deberta_neutro.R (3rd)     DeBERTa-NLI answers on neutral MT-CSD comments
#   fig_gold          desbalanceamento.R         ground-truth shares, without MT-CSD
#   fig_distribuicao  distribuicao.R             answer shares vs ground truth, without MT-CSD
#   fig_ironia        ironia_discrim.R           "negative" on ironic vs non-ironic non-negatives
#   fig_calibracao    calibracao_erros.R         (reads calibracao-20261126.csv)
#   fig_qwenkev       no script before; Qwen3.5-4B vs Kev-4B accuracy and neutral share, order v1
# Sources: merged main run 20261121-220717-839516 (eval/predictions.csv), ablation run
# 20261118-094702-838561 (eval/ablation.csv), conversa_tom-20261122.csv, calibracao-20261126.csv.
# Outputs (this folder): fig_<name>-20261126.{pdf,png} and figuras_artigo-20261126.txt (check values).
suppressPackageStartupMessages({library(dplyr); library(readr); library(tidyr); library(ggplot2)})
W <- 7; BS <- 11; DATE <- "20261126"
ord <- c("twitter-roberta", "deberta-nli", "qwen3.5-4b", "mistral-7b", "kev-4b", "laya")
lab <- c("twitter-roberta", "DeBERTa-NLI", "Qwen3.5-4B", "Mistral-7B", "Kev-4B", "Laya")
mcol <- setNames(c("#6BAED6", "#08519C", "#9E9AC8", "#54278F", "#FD8D3C", "#A63603"), lab)
lv <- c("negative", "neutral", "positive")
ccol <- c(negative = "#C0392B", neutral = "#BDC3C7", positive = "#2471A3")
ds <- c(mustardpp = "MUStARD++", mtcsd = "MT-CSD", sarcasm_nlp = "Sarcasm_NLP", filatova = "Filatova")
th <- theme_bw(base_size = BS) +
  theme(legend.position = "top", panel.grid.minor = element_blank(), strip.background = element_rect(fill = "grey92"),
        legend.margin = margin(0, 0, 0, 0), legend.box.spacing = unit(2, "pt"))
save <- function(p, name, h, w = W) {
  f <- sprintf("fig_%s-%s.pdf", name, DATE)
  ggsave(f, p, width = w, height = h)
  embedFonts(f, outfile = f, options = "-dPDFSETTINGS=/prepress -dEmbedAllFonts=true")  # Ghostscript; fonts embedded for submission
  ggsave(sprintf("fig_%s-%s.png", name, DATE), p, width = w, height = h, dpi = 200)
}
chk <- file(sprintf("figuras_artigo-%s.txt", DATE), "w")
note <- function(title, x) { writeLines(paste("==", title), chk); capture.output(print(as.data.frame(x), row.names = FALSE)) |> writeLines(chk); writeLines("", chk) }

pr <- read_csv("../results/20261121-220717-839516/merged/eval/predictions.csv", show_col_types = FALSE,
               col_select = c(model, dataset, id, sarcastic, gold, condition, variant, pred, correct)) |>
  filter(model %in% ord)
abl <- read_csv("../results/20261118-094702-838561/eval/ablation.csv", show_col_types = FALSE) |>
  filter(model %in% ord)
tom <- read_csv("conversa_tom-20261122.csv", show_col_types = FALSE)

# 1. Context: no context, previous turn (ablation k = 1), whole conversation; order v1 ------------------
grp_of <- function(dataset, sarcastic) case_when(dataset == "mtcsd" ~ "MT-CSD",
                                                 sarcastic == 1 ~ "MUStARD++ ironic", TRUE ~ "MUStARD++")
v1 <- c("v1", "none")
c0 <- pr |> filter(variant %in% v1, dataset %in% c("mustardpp", "mtcsd")) |>
  mutate(grp = grp_of(dataset, sarcastic), cond = ifelse(condition == "no_context", "No context", "Whole conversation"))
c1 <- abl |> filter(variant %in% v1, dataset %in% c("mustardpp", "mtcsd"), k == 1) |>
  mutate(grp = grp_of(dataset, sarcastic), cond = "Previous turn")
cx <- bind_rows(c0 |> select(model, grp, cond, correct), c1 |> select(model, grp, cond, correct)) |>
  group_by(model, grp, cond) |> summarise(acc = 100 * mean(correct), n = n(), .groups = "drop") |>
  mutate(model = factor(model, ord, lab), grp = factor(grp, c("MUStARD++", "MT-CSD", "MUStARD++ ironic")),
         cond = factor(cond, c("No context", "Previous turn", "Whole conversation")))
note("context accuracy (%), order v1", cx |> mutate(acc = round(acc, 1)) |> select(-n) |> pivot_wider(names_from = cond, values_from = acc))
p <- ggplot(cx, aes(model, acc, fill = model, alpha = cond)) +
  geom_col(position = position_dodge(width = 0.85), width = 0.8) +
  facet_wrap(~ grp) +
  scale_fill_manual(values = mcol, guide = "none") +
  scale_alpha_manual(values = c(0.35, 0.65, 1), name = NULL) +
  scale_y_continuous(limits = c(0, 80), expand = expansion(mult = c(0, 0.02))) +
  labs(x = NULL, y = "Accuracy (%)") + th +
  theme(axis.text.x = element_text(angle = 40, hjust = 1), panel.grid.major.x = element_blank()) +
  guides(alpha = guide_legend(override.aes = list(fill = "grey20")))
save(p, "contexto", 2.4)

# 2-3. Fixed MT-CSD cohort (>= 4 turns), k = 0..4, orders pooled -----------------------------------------
co <- abl |> filter(dataset == "mtcsd") |> group_by(model, variant, id) |> filter(max(k) >= 4) |> ungroup() |>
  filter(k <= 4) |> mutate(model = factor(model, ord, lab))
f1m <- function(g, p) mean(sapply(lv, function(c) {
  tp <- sum(g == c & p == c); fp <- sum(g != c & p == c); fn <- sum(g == c & p != c)
  if (tp == 0) 0 else 2 * tp / (2 * tp + fp + fn)
}))
af <- co |> group_by(model, variant, k) |> summarise(acc = 100 * mean(correct), f1 = 100 * f1m(gold, pred), .groups = "drop") |>
  group_by(model, k) |> summarise(across(c(acc, f1), mean), .groups = "drop")
note("cohort accuracy / macro-F1 by k", af |> mutate(across(c(acc, f1), ~ round(.x, 1))))
afl <- af |> pivot_longer(c(acc, f1), names_to = "m", values_to = "v") |>
  mutate(m = factor(m, c("acc", "f1"), c("Accuracy", "Macro F1")))
p <- ggplot(afl, aes(k, v, colour = m)) + geom_line(linewidth = 0.9) + geom_point(size = 1.8) +
  facet_wrap(~ model, nrow = 2) +
  scale_colour_manual(values = c("#2B4C7E", "#D95F02"), name = NULL) +
  scale_x_continuous(breaks = 0:4) + scale_y_continuous(limits = c(30, 80), breaks = c(40, 60, 80)) +
  labs(x = "Context turns (k)", y = "%") + th
save(p, "turnos", 2.4)

q <- co |> filter(model == "Qwen3.5-4B") |>
  mutate(gold = factor(gold, rev(lv), c("positive", "neutral", "against")), pred = factor(pred, lv, c("against", "neutral", "positive"))) |>
  count(k, gold, pred, .drop = FALSE) |> group_by(k, gold) |> mutate(pct = 100 * n / sum(n)) |> ungroup() |>
  mutate(kl = factor(sprintf("k = %d", k)))
note("Qwen confusion (% per gold row)", q |> mutate(pct = round(pct)) |> select(k, gold, pred, pct) |> pivot_wider(names_from = pred, values_from = pct))
p <- ggplot(q, aes(pred, gold, fill = pct)) + geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%.0f", pct), colour = pct > 55), size = 3.3) +
  facet_wrap(~ kl, nrow = 1) +
  scale_fill_gradient(low = "#F4F6F9", high = "#2B4C7E", limits = c(0, 100), guide = "none") +
  scale_colour_manual(values = c(`FALSE` = "grey15", `TRUE` = "white"), guide = "none") +
  labs(x = "Qwen3.5-4B answer", y = "Reference") + th +
  theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 40, hjust = 1))
save(p, "qwen_confusao", 2.0)

# 4. Following the conversation tone (tom_discordante.R), orders pooled ---------------------------------
en <- c(negativo = "negative", neutro = "neutral", positivo = "positive")
tt <- tom |> filter(dataset == "mtcsd", maioria %in% names(en)) |> transmute(id, tom_en = en[maioria])
dt <- co |> inner_join(tt, by = "id") |>
  mutate(grupo = ifelse(gold == tom_en, "agrees", "disagrees"), segue = pred == tom_en)
nn <- dt |> distinct(id, grupo) |> count(grupo)
rt <- dt |> group_by(model, k, grupo) |> summarise(`Answer = conversation tone (%)` = 100 * mean(segue),
                                                   `Accuracy (%)` = 100 * mean(correct), .groups = "drop") |>
  pivot_longer(c(`Answer = conversation tone (%)`, `Accuracy (%)`), names_to = "medida", values_to = "v") |>
  left_join(nn, by = "grupo") |>
  mutate(grupo_l = sprintf("Comment %s with the tone (n = %d)", grupo, n),
         medida = factor(medida, c("Answer = conversation tone (%)", "Accuracy (%)"),
                         c("Follows\ntone (%)", "Accuracy\n(%)")))
note("tone: group sizes", nn)
note("tone: values", rt |> mutate(v = round(v, 1)) |> select(model, k, grupo, medida, v) |> pivot_wider(names_from = k, values_from = v))
p <- ggplot(rt, aes(k, v, colour = grupo_l)) + geom_line(linewidth = 0.8) + geom_point(size = 1.4) +
  facet_grid(medida ~ model, switch = "y") +
  scale_colour_manual(values = c("#2B4C7E", "#C0392B"), name = NULL) +
  scale_y_continuous(limits = c(0, 100), breaks = c(0, 50, 100)) + scale_x_continuous(breaks = 0:4) +
  labs(x = "Context turns (k)", y = NULL) + th +
  theme(strip.placement = "outside") + guides(colour = guide_legend(nrow = 1))
save(p, "tom", 2.5)

# 5. DeBERTa-NLI on neutral MT-CSD comments, k = 0 vs 4 (deberta_neutro.R, third view) -------------------
dn <- abl |> filter(dataset == "mtcsd", model == "deberta-nli") |> group_by(id) |> filter(max(k) >= 4) |> ungroup() |>
  filter(k %in% c(0, 4), gold == "neutral") |>
  inner_join(tom |> filter(dataset == "mtcsd", maioria %in% names(en)) |> transmute(id, tom = en[maioria]), by = "id")
nd <- dn |> distinct(id, tom) |> count(tom)
bd <- dn |> mutate(pred = factor(pred, lv)) |> count(tom, k, pred, .drop = FALSE) |> group_by(tom, k) |>
  mutate(pct = 100 * n / sum(n)) |> ungroup() |> left_join(nd |> rename(N = n), by = "tom") |>
  mutate(kk = factor(ifelse(k == 0, "no context", "4 turns"), c("no context", "4 turns")),
         painel = factor(sprintf("%s conversation\n(n = %d)", tools::toTitleCase(tom), N)),
         painel = factor(painel, levels(painel)[order(match(sub(" .*", "", levels(painel)), c("Neutral", "Negative", "Positive")))]))
note("DeBERTa neutral comments (% answers)", bd |> mutate(pct = round(pct, 1)) |> select(tom, N, k, pred, pct) |> pivot_wider(names_from = pred, values_from = pct))
# Clean style for this figure: no strip boxes or panel borders, light horizontal grid only,
# white gaps between stacked segments, dark labels on the grey segment and white on the coloured ones.
thc <- theme_minimal(base_size = BS) +
  theme(legend.position = "top", legend.margin = margin(0, 0, 0, 0), legend.box.spacing = unit(2, "pt"),
        panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(colour = "grey90", linewidth = 0.3),
        strip.text = element_text(face = "bold", hjust = 0, size = BS - 1, lineheight = 0.95), panel.spacing.x = unit(14, "pt"),
        axis.text = element_text(colour = "grey25"))
p <- ggplot(bd, aes(kk, pct, fill = pred)) + geom_col(width = 0.62, colour = "white", linewidth = 0.6) +
  geom_text(aes(label = ifelse(pct >= 9, sprintf("%.0f", pct), ""), colour = pred == "neutral", group = pred),
            position = position_stack(vjust = 0.5), size = 3.3, fontface = "bold", show.legend = FALSE) +
  facet_wrap(~ painel) +
  scale_fill_manual(values = ccol, name = NULL, labels = c(negative = "against", neutral = "neutral", positive = "in favor")) +
  scale_colour_manual(values = c(`TRUE` = "grey15", `FALSE` = "white"), guide = "none") +
  scale_y_continuous(breaks = c(0, 50, 100), expand = expansion(mult = c(0, 0.02))) +
  labs(x = NULL, y = "% of answers") + thc
save(p, "deberta", 1.95, w = 6)

# 6-7. Class imbalance and answer distribution, without MT-CSD (desbalanceamento.R, distribuicao.R) ------
tk <- pr |> filter(dataset %in% c("mustardpp", "sarcasm_nlp", "filatova"), condition == "no_context") |>
  mutate(tarefa = ifelse(sarcastic == 1, "T3", "T1"), dataset = factor(ds[dataset], ds[-2]))
g <- tk |> distinct(tarefa, dataset, id, gold) |> count(tarefa, dataset, gold) |>
  group_by(tarefa, dataset) |> mutate(pct = 100 * n / sum(n), N = sum(n)) |> ungroup()
ml <- g |> group_by(tarefa, dataset) |> summarise(maj = max(pct), N = first(N), .groups = "drop")
note("ground truth (%) and majority", g |> mutate(pct = round(pct, 1)) |> select(-n) |> pivot_wider(names_from = gold, values_from = pct))
gg <- g |> mutate(tarefa = factor(tarefa, c("T1", "T3"), c("T1 non-ironic", "T3 ironic")),
                  y = factor(sprintf("%s (n = %d)", dataset, N)), gold = factor(gold, rev(lv)))
gg <- gg |> mutate(y = factor(as.character(y), rev(unique(as.character(y)[order(tarefa, dataset)]))))
p <- ggplot(gg, aes(pct, y, fill = gold)) + geom_col(width = 0.75) +
  geom_text(data = gg |> filter(pct >= 8), aes(label = sprintf("%.0f%%", pct)),
            position = position_stack(vjust = 0.5), colour = "white", size = 3.2, fontface = "bold") +
  facet_grid(tarefa ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_fill_manual(values = ccol, name = NULL, breaks = lv) +
  scale_x_continuous(breaks = c(0, 50, 100), labels = function(x) paste0(x, "%"), expand = expansion(mult = c(0, 0.03))) +
  labs(x = "Share of items in the reference labels", y = NULL) + th +
  theme(panel.grid = element_blank(), panel.border = element_blank(), axis.ticks.y = element_blank(),
        strip.placement = "outside", strip.text.y.left = element_text(angle = 90))
save(p, "gold", 1.85, w = 6)

gd <- g |> transmute(tarefa, dataset, quem = "reference", classe = gold, pct)
pd <- tk |> count(tarefa, dataset, model, pred) |> group_by(tarefa, dataset, model) |> mutate(pct = 100 * n / sum(n)) |> ungroup() |>
  transmute(tarefa, dataset, quem = lab[match(model, ord)], classe = pred, pct)
dd <- bind_rows(gd, pd) |>
  mutate(quem = factor(quem, rev(c("reference", lab))), classe = factor(classe, lv),
         painel = factor(paste(tarefa, "·", dataset), c(paste("T1 ·", ds[-2]), paste("T3 ·", ds[-2]))))
note("answer shares (%)", dd |> mutate(pct = round(pct, 1)) |> select(painel, quem, classe, pct) |> pivot_wider(names_from = classe, values_from = pct, values_fill = 0))
p <- ggplot(dd, aes(pct, quem, fill = classe)) + geom_col(width = 0.75) +
  facet_wrap(~ painel, nrow = 2) +
  scale_fill_manual(values = ccol, name = NULL) + scale_x_continuous(breaks = c(0, 50, 100)) +
  labs(x = "% of answers", y = NULL) + th +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(face = ifelse(rev(c("reference", lab)) == "reference", "bold", "plain")))
save(p, "distribuicao", 2.8)

# 8. Irony separation (ironia_discrim.R), orders pooled ---------------------------------------------------
ir <- pr |> filter(condition == "no_context", dataset %in% c("mustardpp", "sarcasm_nlp", "filatova")) |>
  mutate(grp = case_when(sarcastic == 1 & gold == "negative" ~ "hit", sarcastic == 0 & gold != "negative" ~ "alarm"),
         neg = pred == "negative") |> filter(!is.na(grp)) |>
  group_by(model, dataset, grp, id) |> summarise(neg = mean(neg), .groups = "drop") |>
  group_by(model, dataset, grp) |> summarise(v = 100 * mean(neg), .groups = "drop") |>
  pivot_wider(names_from = grp, values_from = v) |>
  mutate(model = factor(model, ord, lab), dataset = factor(ds[dataset], ds[-2]), sep = hit - alarm)
note("irony: % negative on ironic (hit) and non-ironic non-negative (alarm)", ir |> mutate(across(c(hit, alarm, sep), ~ round(.x, 1))))
p <- ggplot(ir, aes(alarm, hit)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
  geom_segment(aes(xend = alarm, yend = alarm), colour = "grey70", linewidth = 0.5) +
  geom_point(aes(colour = model), size = 2.6) +
  facet_wrap(~ dataset) +
  scale_colour_manual(values = mcol, name = NULL) +
  coord_cartesian(xlim = c(0, 100), ylim = c(0, 100)) + scale_x_continuous(breaks = c(0, 50, 100)) + scale_y_continuous(breaks = c(0, 50, 100)) +
  labs(x = "\"Negative\" on non-ironic, non-negative items (%)", y = "\"Negative\" on ironic items (%)") + th +
  guides(colour = guide_legend(nrow = 1))
save(p, "ironia", 2.5)

# 9. Calibration (values from calibracao_erros.R) ------------------------------------------------------------
cal <- read_csv("calibracao-20261126.csv", show_col_types = FALSE) |>
  mutate(metric = factor(metric, c("ECE T1", "ECE T2", "ECE T3 ironic", "Confident errors\n(% of errors)", "Same answer in\nthe 3 orders (%)")), m = factor(m, lab),
         txt = ifelse(grepl("ECE", metric), sprintf("%.2f", v), sprintf("%.0f", v)))
top <- tibble(metric = factor(grep("ECE", levels(cal$metric), value = TRUE), levels(cal$metric)), y = 0.45)
p <- ggplot(cal, aes(m, v, fill = m)) + geom_col(width = 0.8) +
  geom_text(aes(label = txt), vjust = -0.3, size = 2.3) +
  geom_blank(data = top, aes(x = 1, y = y), inherit.aes = FALSE) +
  geom_hline(data = top |> mutate(y = 0.05), aes(yintercept = y), linetype = "dashed", colour = "grey25") +
  facet_wrap(~ metric, nrow = 1, scales = "free_y") +
  scale_fill_manual(values = mcol, name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(x = NULL, y = NULL) + th +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), panel.grid.major.x = element_blank()) +
  guides(fill = guide_legend(nrow = 1))
save(p, "calibracao", 2.1)

# 10. Qwen3.5-4B vs Kev-4B, order v1 ----------------------------------------------------------------------
qk <- pr |> filter(variant == "v1", model %in% c("qwen3.5-4b", "kev-4b")) |>
  mutate(tarefa = case_when(sarcastic != 1 & condition == "no_context" ~ "T1",
                            sarcastic != 1 & condition == "with_context" ~ "T2",
                            sarcastic == 1 & condition == "no_context" ~ "T3")) |>
  filter(!is.na(tarefa), !(tarefa == "T2" & !dataset %in% c("mustardpp", "mtcsd")))
qs <- qk |> group_by(tarefa, dataset, model) |>
  summarise(`Accuracy (%)` = 100 * mean(correct), `"Neutral" answers (%)` = 100 * mean(pred == "neutral"),
            ref = 100 * mean(gold == "neutral"), .groups = "drop") |>
  mutate(x = factor(sprintf("%s\n%s", tarefa, ds[dataset]),
                    unique(sprintf("%s\n%s", tarefa, ds[dataset]))[order(unique(paste(tarefa, match(dataset, names(ds)))))]),
         model = factor(model, c("qwen3.5-4b", "kev-4b"), c("Qwen3.5-4B", "Kev-4B")))
lvx <- qs |> distinct(tarefa, dataset, x) |> arrange(tarefa, match(dataset, names(ds))) |> pull(x) |> as.character()
qs <- qs |> mutate(x = factor(as.character(x), lvx))
note("Qwen vs Kev, order v1", qs |> mutate(across(where(is.numeric), ~ round(.x, 1))) |> select(-x))
ql <- qs |> pivot_longer(c(`Accuracy (%)`, `"Neutral" answers (%)`), names_to = "m", values_to = "v") |>
  mutate(m = factor(m, c("Accuracy (%)", "\"Neutral\" answers (%)")))
rf <- qs |> distinct(x, ref) |> mutate(m = factor("\"Neutral\" answers (%)", levels(ql$m)))
p <- ggplot(ql, aes(x, v, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  geom_errorbar(data = rf, aes(x = x, ymin = ref, ymax = ref, linetype = "reference"), inherit.aes = FALSE,
                width = 0.9, linewidth = 0.8) +
  facet_wrap(~ m, ncol = 1) +
  scale_fill_manual(values = mcol[c("Qwen3.5-4B", "Kev-4B")], name = NULL) +
  scale_linetype_manual(values = c(reference = "solid"), name = NULL, labels = "\"neutral\" in the reference") +
  scale_y_continuous(limits = c(0, 100)) +
  labs(x = NULL, y = "%") + th + theme(panel.grid.major.x = element_blank(), axis.text.x = element_text(size = BS - 3))
save(p, "qwenkev", 2.5)

close(chk)
