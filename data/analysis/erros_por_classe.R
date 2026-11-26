# Exploratory: accuracy per gold class (recall) with and without context, by number of
# context turns of the item. Non-sarcastic items, main prompt order (v1 / none).
# Usage (from experimento/analysis): Rscript erros_por_classe.R <RUN_ID>
# Output: figures/erros_por_classe-<RUN_ID>.pdf and counts printed to stdout.
suppressPackageStartupMessages({ library(dplyr); library(tidyr); library(readr); library(ggplot2) })
run_id <- commandArgs(TRUE)[1]
run <- file.path("../results", run_id)
lab <- c("twitter-roberta" = "twitter-roberta", "deberta-nli" = "DeBERTa-NLI", "qwen3.5-4b" = "Qwen3.5-4B",
         "mistral-7b" = "Mistral-7B", "kev-4b" = "Kev-4B", "laya" = "Laya")
fam <- c("twitter-roberta" = "BERT", "deberta-nli" = "BERT", "qwen3.5-4b" = "LLM", "mistral-7b" = "LLM",
         "kev-4b" = "System One", "laya" = "System One")
cols <- c("twitter-roberta" = "#6baed6", "DeBERTa-NLI" = "#08519c", "Qwen3.5-4B" = "#9e9ac8",
          "Mistral-7B" = "#54278f", "Kev-4B" = "#fd8d3c", "Laya" = "#a63603")
gold_lab <- c(negative = "Negativo / contra", neutral = "Neutro / nenhuma", positive = "Positivo / a favor")
ds_lab <- c(mustardpp = "MUStARD++", mtcsd = "MT-CSD")

lines <- readLines(file.path(run, "data", "items.jsonl"), encoding = "UTF-8")
turns <- tibble(id = sub('^\\{"id": "([^"]+)".*', "\\1", lines),
                k = lengths(regmatches(lines, gregexpr('\\{"speaker"', lines))))
d <- read_csv(file.path(run, "eval", "predictions.csv"), show_col_types = FALSE,
              col_select = c(model, dataset, id, sarcastic, gold, condition, variant, correct)) |>
  filter(variant %in% c("v1", "none"), sarcastic == 0, model %in% names(fam), dataset %in% names(ds_lab)) |>
  pivot_wider(names_from = condition, values_from = correct) |>
  inner_join(turns, by = "id") |>
  mutate(k = factor(ifelse(k >= 6, "6+", as.character(k)), c(1:5, "6+")),
         m = factor(lab[model], lab), f = factor(fam[model], c("BERT", "LLM", "System One")),
         ds = factor(ds_lab[dataset], ds_lab), g = factor(gold_lab[gold], gold_lab))

cat("Items per dataset x gold class x turns bucket:\n")
print(d |> distinct(ds, g, k, id) |> count(ds, g, k) |> pivot_wider(names_from = k, values_from = n), n = Inf)
cat("\nAccuracy per gold class, all buckets (no / with context):\n")
print(d |> group_by(ds, g, m) |>
        summarise(n = n(), no = round(100 * mean(no_context), 1), with = round(100 * mean(with_context), 1),
                  delta = with - no, .groups = "drop"), n = Inf)

s <- d |> group_by(ds, g, f, m, k) |>
  summarise(`Com contexto` = 100 * mean(with_context), `Sem contexto` = 100 * mean(no_context),
            n = n(), .groups = "drop") |>
  pivot_longer(c(`Com contexto`, `Sem contexto`), names_to = "cond", values_to = "acc")
# per model: light wide bar = without context (shadow), solid narrow bar = with context
pd <- position_dodge(width = 0.85)
p <- ggplot(s, aes(x = k, y = acc, fill = m, alpha = cond, width = ifelse(cond == "Sem contexto", 0.8, 0.45))) +
  geom_col(data = filter(s, cond == "Sem contexto"), position = pd) +
  geom_col(data = filter(s, cond == "Com contexto"), position = pd) +
  facet_grid(ds + g ~ f) + scale_fill_manual(values = cols, name = NULL) +
  scale_alpha_manual(values = c(`Com contexto` = 1, `Sem contexto` = 0.35), name = NULL) +
  scale_y_continuous(limits = c(0, 100), expand = expansion(mult = c(0, 0.02))) +
  labs(x = "Turnos de contexto do item", y = "Acurácia na classe (%)") +
  guides(fill = guide_legend(nrow = 1, order = 1), alpha = guide_legend(nrow = 1, order = 2)) +
  theme_bw(base_size = 22) +
  theme(legend.position = "top", legend.box = "vertical", panel.grid.minor = element_blank())
dir.create("figures", showWarnings = FALSE)
out <- file.path("figures", paste0("erros_por_classe-", run_id, ".pdf"))
ggsave(out, p, width = 18, height = 24)
cat("\nwrote", out, "\n")
