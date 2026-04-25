#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import nbformat


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = REPO_ROOT / "src" / "revisiting-uid.ipynb"
OUTPUT_NOTEBOOK = REPO_ROOT / "src" / "revisiting-uid-public.ipynb"


def replace_with_python_comment(message: str) -> str:
    return f'print("{message}")'


def empty_dataset_block(var_name: str, label: str) -> str:
    return "\n".join(
        [
            f'print("Skipping {label}: local data not available.")',
            f"{var_name} = pd.DataFrame()",
            f"{var_name}_agg_per_subject_sentence = pd.DataFrame()",
            f"{var_name}_mean_per_sen = pd.DataFrame()",
        ]
    )


def main() -> int:
    nb = nbformat.read(SOURCE_NOTEBOOK, as_version=4)

    replacements: dict[int, str] = {
        19: "\n".join(
            [
                "MODELS = ['gpt', 'ngram', 'bert']",
                'print("Public notebook skips transxl because the deprecated TransfoXL path is not compatible with the current transformers stack.")',
            ]
        ),
        17: "\n".join(
            [
                'CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints")',
                "os.makedirs(CHECKPOINT_DIR, exist_ok=True)",
                "",
                "def make_pickleable(obj):",
                "    if isinstance(obj, defaultdict):",
                "        return {k: make_pickleable(v) for k, v in obj.items()}",
                "    if isinstance(obj, dict):",
                "        return {k: make_pickleable(v) for k, v in obj.items()}",
                "    if isinstance(obj, list):",
                "        return [make_pickleable(v) for v in obj]",
                "    if isinstance(obj, tuple):",
                "        return tuple(make_pickleable(v) for v in obj)",
                "    return obj",
                "",
                "def normalize_alignment_token(token):",
                "    token = str(token).lower().strip()",
                "    token = token.replace('’', \"'\").replace('‘', \"'\").replace('`', \"'\")",
                "    token = re.sub(r'(?<=\\w)\\?(?=\\w)', \"'\", token)",
                "    return token.strip(punctuation)",
                "",
                "def checkpoint_path(name):",
                "    return os.path.join(CHECKPOINT_DIR, name)",
                "",
                "def has_checkpoint(name):",
                "    return os.path.exists(checkpoint_path(name))",
                "",
                "def save_pickle(obj, name):",
                "    path = checkpoint_path(name)",
                "    with open(path, 'wb') as f:",
                "        pickle.dump(make_pickleable(obj), f)",
                "    print(f'Saved checkpoint: {path}')",
                "",
                "def load_pickle(name):",
                "    path = checkpoint_path(name)",
                "    with open(path, 'rb') as f:",
                "        obj = pickle.load(f)",
                "    print(f'Loaded checkpoint: {path}')",
                "    return obj",
                "",
                "def has_stats_checkpoint(name):",
                "    return all(has_checkpoint(filename) for filename in [f'{name}_df.pkl', f'{name}_subject_sen.pkl', f'{name}_sen.pkl'])",
                "",
                "def pickle_stats(main, subject_sen, sen, name):",
                "    save_pickle(main, f'{name}_df.pkl')",
                "    save_pickle(subject_sen, f'{name}_subject_sen.pkl')",
                "    save_pickle(sen, f'{name}_sen.pkl')",
                "",
                "def load_stats(name):",
                "    return (",
                "        load_pickle(f'{name}_df.pkl'),",
                "        load_pickle(f'{name}_subject_sen.pkl'),",
                "        load_pickle(f'{name}_sen.pkl'),",
                "    )",
            ]
        ),
        13: "\n".join(
            [
                "def add_standard_columns(df, split_strings, lang='en'):",
                "    # ref token is sanity check. should be same as word",
                "    df['ref_token'] = df.apply(lambda x: split_strings[x['text_id']][x['new_ind']], axis=1)",
                "    df['centered_time'] = df['time'] - df.groupby(by=['WorkerId'])['time'].transform('mean')",
                "    df['prev_word'] = df.apply(lambda x: split_strings[x['text_id']][x['new_ind']-1] if x['new_ind']-1 >= 0 else '', axis=1)",
                "    df['word_len'] = df.apply(lambda x: len(x['word']), axis=1)",
                "    df['prev_word_len'] = df.apply(lambda x: len(x['prev_word']), axis=1)",
                "    df['freq'] = df.apply(lambda x: frequency(x['word'].strip().strip(punctuation).lower(), lang), axis=1)",
                "    df['prev_freq'] = df.apply(lambda x: frequency(x['prev_word'].strip(punctuation).lower(), lang), axis=1)",
                "",
                "def nancount(x):",
                "    return x.isnull().sum()",
                "",
                "def scalar_mode(x):",
                "    vals = pd.Series(x).dropna()",
                "    if vals.empty:",
                "        return np.nan",
                "    modes = vals.mode(dropna=True)",
                "    return modes.iloc[0] if not modes.empty else np.nan",
                "",
                "def produce_aggregate_per_subject_sentence(main_df, line_col=False):",
                "    aggregate_per_subject_sentence = main_df.groupby(by=['WorkerId', 'text_id', 'sentence_num'], sort=False).agg(",
                "        time_sum=('time', np.sum),",
                "        time_mean=('time', np.mean),",
                "        time_count_nonzero=('time', np.count_nonzero),",
                "        word_len_sum=('word_len', np.sum),",
                "        word_len_mean=('word_len', np.mean),",
                "        freq_nansum=('freq', np.nansum),",
                "        freq_nanmean=('freq', np.nanmean),",
                "        freq_nancount=('freq', nancount),",
                "        outlier_sum=('outlier', np.sum),",
                "    ).reset_index()",
                "    aggregate_per_subject_sentence = aggregate_per_subject_sentence.rename(columns={'WorkerId': 'WorkerId_', 'text_id': 'text_id_', 'sentence_num': 'sentence_num_'})",
                "    if line_col:",
                "        line_breaks = main_df.groupby(by=['WorkerId', 'text_id', 'sentence_num'], sort=False).agg(",
                "            line_breaks=(line_col, lambda x: len(np.unique(x)))",
                "        ).reset_index().rename(columns={'WorkerId': 'WorkerId_', 'text_id': 'text_id_', 'sentence_num': 'sentence_num_'})",
                "        aggregate_per_subject_sentence = aggregate_per_subject_sentence.merge(",
                "            line_breaks,",
                "            on=['WorkerId_', 'text_id_', 'sentence_num_'],",
                "            how='left',",
                "        )",
                "    aggregate_per_subject_sentence['id'] = aggregate_per_subject_sentence.apply(lambda x: str(int(x['text_id_'])) + '_' + str(int(x['sentence_num_'])), axis=1)",
                "    return aggregate_per_subject_sentence",
                "",
                "def produce_aggregate_per_sentence(aggregate_per_subject_sentence, remove_outliers=True):",
                "    aggregate_per_subject_sentence = aggregate_per_subject_sentence.copy()",
                "    for col in ['time_sum', 'time_count_nonzero', 'time_mean', 'outlier_sum']:",
                "        aggregate_per_subject_sentence[col] = pd.to_numeric(aggregate_per_subject_sentence[col], errors='coerce')",
                "    aggregate_per_sentence = aggregate_per_subject_sentence.groupby(by=['text_id_', 'sentence_num_'], sort=False).agg(",
                "        time_sum=('time_sum', 'mean'),",
                "        time_count_nonzero=('time_count_nonzero', scalar_mode),",
                "        time_mean=('time_mean', 'mean'),",
                "    ).reset_index()",
                "    if remove_outliers:",
                "        filtered = aggregate_per_subject_sentence.loc[aggregate_per_subject_sentence.outlier_sum == 0].copy()",
                "        tmp = filtered.groupby(by=['text_id_', 'sentence_num_'], sort=False).agg(",
                "            time_sum_NO=('time_sum', 'mean'),",
                "            time_count_nonzero_NO=('time_count_nonzero', scalar_mode),",
                "            time_mean_NO=('time_mean', 'mean'),",
                "        ).reset_index()",
                "        aggregate_per_sentence = aggregate_per_sentence.merge(tmp, on=['text_id_', 'sentence_num_'], how='left')",
                "    aggregate_per_sentence['id'] = aggregate_per_sentence.apply(lambda x: str(int(x['text_id_'])) + '_' + str(int(x['sentence_num_'])), axis=1)",
                "    return aggregate_per_sentence",
            ]
        ),
        2: "\n".join(
            [
                "%%R",
                'message("Using R libraries from ", Sys.getenv("R_LIBS_USER"))',
                '.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))',
            ]
        ),
        21: "\n".join(
            [
                'gpt3_probs = pd.read_csv("corpora/naturalstories/all_stories_gpt3.csv")',
                "# To get same indexing as stories db",
                'gpt3_probs["story"] = gpt3_probs["story"] + 1',
                "gpt3_probs['len'] = gpt3_probs.groupby(\"story\", sort=False)['offset'].shift(periods=-1, fill_value=0) - gpt3_probs['offset']",
                "gpt3_probs['new_token'] = gpt3_probs.apply(lambda x: x['token'] if x['len'] == len(x['token']) else x['token'] + ' ', axis=1)",
            ]
        ),
        22: "\n".join(
            [
                'stories_df = gpt3_probs.groupby(by=["story"], sort=False).agg({"new_token":[string_join]}).reset_index()',
                "stories = list(zip(stories_df['story'], stories_df['new_token', 'string_join']))",
                'if has_checkpoint("ns_stats.pkl"):',
                '    ns_stats = load_pickle("ns_stats.pkl")',
                "else:",
                "    ns_stats = corpus_stats(stories, models=MODELS)",
                '    save_pickle(ns_stats, "ns_stats.pkl")',
            ]
        ),
        23: "\n".join(
            [
                'natural_stories = pd.read_csv("corpora/naturalstories/processed_RTs.tsv", sep="\\t").drop_duplicates()',
                "natural_stories.rename(columns = {'RT':'time',",
                "                                   'item': 'text_id'}, inplace = True)",
                "natural_stories['new_ind'] = natural_stories['zone'] - 1",
                "natural_stories['sentence_num'] = natural_stories.apply(lambda x: bisect.bisect(ns_stats['sent_markers'][x['text_id']], x['new_ind']), axis=1)",
                "natural_stories = find_outliers(natural_stories, transform=np.log)",
            ]
        ),
        24: "\n".join(
            [
                'if has_stats_checkpoint("ns"):',
                '    natural_stories, ns_agg_per_subject_sentence, ns_mean_per_sen = load_stats("ns")',
                "else:",
                "    natural_stories, ns_agg_per_subject_sentence, ns_mean_per_sen = create_analysis_dfs(natural_stories, ns_stats, MODELS)",
                '    pickle_stats(natural_stories, ns_agg_per_subject_sentence, ns_mean_per_sen, "ns")',
            ]
        ),
        28: "\n".join(
            [
                "provo = pd.read_csv('corpora/provo.csv')",
                "provo.rename(columns = {'IA_DWELL_TIME':'time', 'Participant_ID': 'WorkerId', 'Word':'word', ",
                "                        \"Text_ID\":\"text_id\", \"Sentence_Number\":\"sentence_num\",",
                "                       \"IA_FIRST_RUN_DWELL_TIME\": 'time2', 'IA_FIRST_FIXATION_DURATION':'time3'}, inplace = True)",
                "provo = provo.dropna(subset=[\"Word_Number\"])",
                "provo = provo.astype({\"Word_Number\": 'Int64', \"sentence_num\": 'Int64'})",
                "provo['word'] = provo.apply(lambda x: MOSESNORMALIZER(x['word']).strip(), axis=1)",
                "provo['word'] = provo['word'].str.replace(r'(?<=\\w)\\?(?=\\w)', \"'\", regex=True)",
                "#fixing small discrepancy",
                "provo.loc[provo['word'] == '0.9', 'word'] = '90%'",
            ]
        ),
        29: "\n".join(
            [
                "provo_text = pd.read_csv('corpora/provo_norms.csv', encoding='mac_roman')[['Text_ID','Text']].drop_duplicates().sort_values(by=['Text_ID'])",
                "provo_text.drop(provo_text[(provo_text.Text_ID == 27) & (~provo_text.Text.str.contains(\"doesn't\", regex=False))].index, inplace=True)",
                "inds = provo_text.apply(lambda x: list(range(1,len(x['Text'].split())+1)), axis=1)",
                "inds = {i:j for i,j in zip(provo_text['Text_ID'], inds)}",
                "paragraphs = {i:j.replace(u\"\\uFFFD\", \"?\") for i,j in provo_text[['Text_ID','Text']].itertuples(index=False, name=None)}",
                "paragraphs_split = {i:[normalize_alignment_token(k) for k in j.split()] for i,j in paragraphs.items()}",
            ]
        ),
        26: replace_with_python_comment("Using cached or freshly computed Natural Stories statistics."),
        30: "\n".join(
            [
                'if has_checkpoint("provo_stats.pkl"):',
                '    provo_stats = load_pickle("provo_stats.pkl")',
                "else:",
                "    provo_stats = corpus_stats(paragraphs.items(), models=MODELS)",
                '    save_pickle(provo_stats, "provo_stats.pkl")',
            ]
        ),
        32: "\n".join(
            [
                'if has_stats_checkpoint("provo2"):',
                '    provo, provo_agg_per_subject_sentence, provo_mean_per_sen = load_stats("provo2")',
                "else:",
                "    provo, provo_agg_per_subject_sentence, provo_mean_per_sen = create_analysis_dfs(provo, provo_stats, MODELS)",
                '    pickle_stats(provo, provo_agg_per_subject_sentence, provo_mean_per_sen, "provo2")',
                'provo_agg_per_subject_sentence["time2_sum"] = provo.groupby(by=["WorkerId","text_id", "sentence_num", "model"]).agg({"time2":np.sum}).reset_index()["time2"]',
                'provo_agg_per_subject_sentence["time3_sum"] = provo.groupby(by=["WorkerId","text_id", "sentence_num", "model"]).agg({"time3":np.sum}).reset_index()["time3"]',
            ]
        ),
        31: "\n".join(
            [
                'provo["new_ind"] = provo["Word_Number"] - 2',
                'provo["new_ind"] = provo.apply(lambda x: x["new_ind"] + paragraphs_split[x["text_id"]][x["new_ind"]:].index(normalize_alignment_token(x["word"])), axis=1)',
                'provo["sentence_num"] = provo.apply(lambda x: bisect.bisect(provo_stats["sent_markers"][x["text_id"]], x["new_ind"]), axis=1)',
                "provo = find_outliers(provo.loc[provo['time'] != 0], transform=np.log)",
            ]
        ),
        33: replace_with_python_comment("Using cached or freshly computed Provo statistics."),
        35: "\n".join(
            [
                "ucl = pd.read_csv('corpora/ucl/selfpacedreading.RT.txt', sep='\\t')",
                "ucl.rename(columns = {'RT':'time', 'subj_nr': 'WorkerId', ",
                "                        \"sent_nr\":\"text_id\"}, inplace = True)",
                "ucl = ucl.dropna(subset=['word']).copy()",
                "ucl['word'] = ucl.apply(lambda x: MOSESNORMALIZER(x['word']).strip(), axis=1)",
            ]
        ),
        36: "\n".join(
            [
                "inds, paragraphs = zip(*ucl[['text_id','word_pos','word']].drop_duplicates().dropna().groupby(by = ['text_id']).apply(lambda x: ordered_string_join(zip(x['word_pos'], x['word']), ' ')))",
                'if has_checkpoint("ucl_stats.pkl"):',
                '    ucl_stats = load_pickle("ucl_stats.pkl")',
                "else:",
                "    ucl_stats = corpus_stats(list(enumerate(paragraphs,1)), models=MODELS)",
                '    save_pickle(ucl_stats, "ucl_stats.pkl")',
            ]
        ),
        38: "\n".join(
            [
                'if has_stats_checkpoint("ucl"):',
                '    ucl, ucl_agg_per_subject_sentence, ucl_mean_per_sen = load_stats("ucl")',
                "else:",
                "    ucl, ucl_agg_per_subject_sentence, ucl_mean_per_sen = create_analysis_dfs(ucl, ucl_stats, MODELS)",
                '    pickle_stats(ucl, ucl_agg_per_subject_sentence, ucl_mean_per_sen, "ucl")',
            ]
        ),
        39: replace_with_python_comment("Using cached or freshly computed UCL self-paced statistics."),
        41: "\n".join(
            [
                "ucl_eye = pd.read_csv('corpora/ucl/eyetracking.RT.txt', sep='\\t')",
                "ucl_eye.rename(columns = {'RTfirstpass':'time', 'subj_nr': 'WorkerId', ",
                "                        \"sent_nr\":\"text_id\"}, inplace = True)",
                "ucl_eye = ucl_eye.dropna(subset=['word']).copy()",
                "ucl_eye['word'] = ucl_eye.apply(lambda x: MOSESNORMALIZER(x['word']).strip(), axis=1)",
            ]
        ),
        42: "\n".join(
            [
                "joined = ucl_eye[['text_id','word_pos','word']].drop_duplicates().dropna().groupby(by = ['text_id']).apply(lambda x: ordered_string_join(zip(x['word_pos'], x['word']), ' '))",
                "inds, paragraphs = zip(*joined)",
                'if has_checkpoint("ucl_eye_stats.pkl"):',
                '    ucl_eye_stats = load_pickle("ucl_eye_stats.pkl")',
                "else:",
                "    ucl_eye_stats = corpus_stats(list(zip(joined.index, paragraphs)), models=MODELS)",
                '    save_pickle(ucl_eye_stats, "ucl_eye_stats.pkl")',
            ]
        ),
        44: "\n".join(
            [
                'if has_stats_checkpoint("ucl_eye"):',
                '    ucl_eye, ucl_eye_agg_per_subject_sentence, ucl_eye_mean_per_sen = load_stats("ucl_eye")',
                "else:",
                "    ucl_eye, ucl_eye_agg_per_subject_sentence, ucl_eye_mean_per_sen = create_analysis_dfs(ucl_eye, ucl_eye_stats, MODELS)",
                '    pickle_stats(ucl_eye, ucl_eye_agg_per_subject_sentence, ucl_eye_mean_per_sen, "ucl_eye")',
            ]
        ),
        45: replace_with_python_comment("Using cached or freshly computed UCL eye-tracking statistics."),
        69: "\n".join(
            [
                "cola = pd.read_csv('corpora/cola_public/raw/in_domain_train.tsv', sep='\\t', header=None, names=['ID','accept','NA','sentence'])",
                "cola = cola.drop(columns='NA')",
                "cola['text_id_'] = cola.index",
                "cola['sentence_num_'] = 0",
                "cola['sentence'] = cola.apply(lambda x: MOSESNORMALIZER(x['sentence']).strip(), axis=1)",
                'if has_checkpoint("cola_stats.pkl"):',
                '    cola_stats = load_pickle("cola_stats.pkl")',
                "else:",
                "    cola_stats = corpus_stats(list(enumerate(cola['sentence'])), models=MODELS, split_sens=False)",
                '    save_pickle(cola_stats, "cola_stats.pkl")',
            ]
        ),
        70: "\n".join(
            [
                'if has_checkpoint("cola.pkl"):',
                '    cola = load_pickle("cola.pkl")',
                "else:",
                "    cola = pd.concat([add_log_prob_aggregate_cols(cola, cola_stats, model=mod) for mod in MODELS])",
                "    add_lau_accept_measures(cola)",
                '    save_pickle(cola, "cola.pkl")',
            ]
        ),
        71: replace_with_python_comment("Using cached or freshly computed CoLA acceptability data."),
        73: "\n".join(
            [
                "bnc = pd.read_csv('corpora/bnc.csv', sep='\\t')",
                "bnc.rename(columns = {'mean_rating':'accept', 'text':'sentence', 'length':'len'}, inplace = True)",
                "bnc['text_id_'] = bnc.index",
                "bnc['sentence_num_'] = 0",
                "bnc['sentence'] = bnc.apply(lambda x: re.sub(r\"\\s+\", ' ', re.sub(r'[\\u4e00-\\u9fff|\\u00b0]+', '', MOSESNORMALIZER(x['sentence'].strip().replace('\"\"','\"')))), axis=1)",
                'if has_checkpoint("bnc_stats.pkl"):',
                '    bnc_stats = load_pickle("bnc_stats.pkl")',
                "else:",
                "    bnc_stats = corpus_stats(list(enumerate(bnc['sentence'])), models=MODELS, split_sens=False)",
                '    save_pickle(bnc_stats, "bnc_stats.pkl")',
            ]
        ),
        74: "\n".join(
            [
                'if has_checkpoint("bnc.pkl"):',
                '    bnc = load_pickle("bnc.pkl")',
                "else:",
                "    bnc = pd.concat([add_log_prob_aggregate_cols(bnc, bnc_stats, model=mod) for mod in MODELS])",
                "    add_lau_accept_measures(bnc)",
                '    save_pickle(bnc, "bnc.pkl")',
            ]
        ),
        75: replace_with_python_comment("Using cached or freshly computed BNC acceptability data."),
        78: "\n".join(
            [
                'if has_checkpoint("agg_per_subject_sentence_full.pkl"):',
                '    agg_per_subject_sentence_full = load_pickle("agg_per_subject_sentence_full.pkl")',
                "else:",
                '    agg_per_subject_sentence_full = pd.concat([ns_agg_per_subject_sentence.assign(dataset="Natural Stories"),',
                '                                               provo_agg_per_subject_sentence.assign(dataset="Provo"),',
                '                                               dundee_agg_per_subject_sentence.assign(dataset="Dundee"),',
                '                                               brown_agg_per_subject_sentence.assign(dataset="Brown"),',
                '                                               ucl_agg_per_subject_sentence.assign(dataset="UCL (R)")])',
                "    #agg_per_subject_sentence_full = geco_agg_per_subject_sentence.assign(dataset=\"GECO\")",
                "    agg_per_subject_sentence_full['WorkerId_'] = agg_per_subject_sentence_full['WorkerId_'].astype(str)",
                "    try:",
                "        # In case columns from different dfs are of different types",
                "        types = agg_per_subject_sentence_full.applymap(type).apply(set)",
                "        cols = types[types.apply(len) > 1].index",
                "        agg_per_subject_sentence_full[cols] = agg_per_subject_sentence_full[cols].apply(lambda x: x.astype(np.float64), 1)",
                "    except TypeError:",
                "        pass",
                '    save_pickle(agg_per_subject_sentence_full, "agg_per_subject_sentence_full.pkl")',
            ]
        ),
        79: "\n".join(
            [
                'if has_checkpoint("agg_per_sentence_full.pkl"):',
                '    agg_per_sentence_full = load_pickle("agg_per_sentence_full.pkl")',
                "else:",
                '    agg_per_sentence_full = pd.concat([ns_mean_per_sen.assign(dataset="Natural Stories"),',
                '                                       provo_mean_per_sen.assign(dataset="Provo"),',
                '                                       dundee_mean_per_sen.assign(dataset="Dundee"),',
                '                                       brown_mean_per_sen.assign(dataset="Brown"),',
                '                                       ucl_mean_per_sen.assign(dataset="UCL (R)")])',
                "    try:",
                "        types = agg_per_sentence_full.applymap(type).apply(set)",
                "        cols = types[types.apply(len) > 1].index",
                "        agg_per_sentence_full[cols] = agg_per_sentence_full[cols].apply(lambda x: x.astype(np.float64), 1)",
                "    except TypeError:",
                "        pass",
                '    save_pickle(agg_per_sentence_full, "agg_per_sentence_full.pkl")',
            ]
        ),
        81: "\n".join(
            [
                'if has_checkpoint("acceptability.pkl"):',
                '    acceptability = load_pickle("acceptability.pkl")',
                "else:",
                "    acceptability = pd.concat([cola.drop(['ID', 'sentence'], axis=1).assign(dataset='CoLA'),",
                "                               bnc.drop(['MOP', 'language', 'sentence', 'rating_list'], axis=1).assign(dataset='BNC')])",
                '    save_pickle(acceptability, "acceptability.pkl")',
            ]
        ),
        83: "\n".join(
            [
                "%%R",
                "library(lme4)",
                "library(ggplot2)",
                "library(dplyr)",
                'dir.create("checkpoints", showWarnings = FALSE)',
                'dir.create("figures", showWarnings = FALSE)',
                "",
                "save_plot <- function(plot, stem, width, height, dpi=300){",
                '    png_path <- file.path("figures", paste0(stem, ".png"))',
                '    pdf_path <- file.path("figures", paste0(stem, ".pdf"))',
                "    ggsave(filename=png_path, plot=plot, width=width, height=height, dpi=dpi, bg='white')",
                "    ggsave(filename=pdf_path, plot=plot, width=width, height=height, device=cairo_pdf, bg='white')",
                "    print(plot)",
                '    message("Saved figure: ", png_path)',
                '    message("Saved figure: ", pdf_path)',
                "    invisible(plot)",
                "}",
            ]
        ),
        90: "\n".join(
            [
                "%%R",
                'checkpoint_file <- file.path("checkpoints", "reading_time_cv.rds")',
                "if (file.exists(checkpoint_file)) {",
                "    out <- readRDS(checkpoint_file)",
                '    message("Loaded checkpoint: ", checkpoint_file)',
                "} else {",
                "    df_full <- filter(agg_per_subject_sentence_full, agg_per_subject_sentence_full$outlier_sum==0)",
                "    outcome <- 'time_sum'",
                "    predictors <- list(b=c('len', 'I(len*uni_log_prob_power_1.0)*ch_len'))",
                "    models <- c('bert','gpt','ngram')",
                "    datasets <- c('Dundee','Brown','Provo','Natural Stories')",
                "",
                "    dataset_func <- function(ds){",
                "        print(ds)",
                "        data_per_ds <- filter(df_full, df_full$dataset == ds)",
                "        if(nrow(data_per_ds) == 0){",
                "                    return(NULL)",
                "        }",
                "        predictors_func <- function(preds){",
                "            other_preds <- paste0(preds, collapse='+')",
                "            model_func <- function(model_name){",
                "                data <- filter(data_per_ds, data_per_ds$model == model_name)",
                "                if(nrow(data) == 0){",
                "                    return(NULL)",
                "                }",
                "                set.seed(42)",
                "                shuffled_order <- sample(nrow(data))",
                "                agg_baseline <- lme_cross_val(paste0(outcome,'~(len + 0 | WorkerId_)+', other_preds), data[shuffled_order,],outcome)",
                "                power_func <- function(x){",
                "                    pred <- paste0('log_prob_power_', x,':len')",
                "                    formula <- paste0(outcome, '~ ',pred,'+(',pred,'+ len + 0 | WorkerId_) +', other_preds)",
                "                    cv <- lme_cross_val(formula, data[shuffled_order,], outcome, num_folds=10)",
                "                    c(mean(cv-agg_baseline, na.rm=TRUE), var(cv-agg_baseline, na.rm=TRUE)/length(cv), mean(cv, na.rm=TRUE), sum(is.na(cv)))",
                "                }",
                "                cbind(labels, as.data.frame(do.call(rbind, lapply(powers_np_format, power_func))), model_name)",
                "            }",
                "            cbind(as.data.frame(do.call(rbind, lapply(models, model_func))), other_preds)",
                "        }",
                "        print(nrow(data_per_ds))",
                "        cbind(as.data.frame(do.call(rbind, lapply(predictors, predictors_func))), ds)",
                "    }",
                "    out <- as.data.frame(do.call(rbind, lapply(datasets, dataset_func)))",
                "    saveRDS(out, checkpoint_file)",
                '    write.csv(out, file.path("checkpoints", "reading_time_cv.csv"), row.names=FALSE)',
                '    message("Saved checkpoint: ", checkpoint_file)',
                "}",
            ]
        ),
        91: "\n".join(
            [
                "%%R",
                'checkpoint_file <- file.path("checkpoints", "acceptability_cv.rds")',
                "if (file.exists(checkpoint_file)) {",
                "    out_accept <- readRDS(checkpoint_file)",
                '    message("Loaded checkpoint: ", checkpoint_file)',
                "} else {",
                "    df_full <- acceptability",
                "    predictors <- list(b=c('1'))",
                "    models <- c('bert', 'ngram', 'gpt')",
                "    datasets <- c('BNC', 'CoLA')",
                "",
                "    dataset_func <- function(ds){",
                "        print(ds)",
                "        data_per_ds <- filter(df_full, df_full$dataset == ds)",
                "        if(nrow(data_per_ds) == 0){",
                "                    return(NULL)",
                "        }",
                "        family <- binomial",
                "        predictors_func <- function(preds){",
                "            other_preds <- paste0(preds, collapse='+')",
                "            model_func <- function(model_name){",
                "                d <- filter(data_per_ds, data_per_ds$model == model_name)",
                "                if(nrow(d) == 0){",
                "                    return(NULL)",
                "                }",
                "                set.seed(42)",
                "                shuffled_order <- sample(nrow(d))",
                "                baseline <- lm_cross_val(paste('accept ~ ', other_preds),",
                "                                  d[shuffled_order,],",
                "                                  'accept',",
                "                                  family)",
                "",
                "                power_func <- function(x){",
                "                        name <- paste0('log_prob_power_',x, ':len')",
                "                        formula <- paste0('accept ~ ', name,' +', other_preds)",
                "                        cv <- lm_cross_val(formula, d[shuffled_order,], 'accept', family)",
                "                        c(mean(cv-baseline, na.rm=TRUE), var(cv-baseline, na.rm=TRUE)/length(cv),mean(cv, na.rm=TRUE))",
                "                    }",
                "                cbind(labels, as.data.frame(do.call(rbind,lapply(powers_np_format, power_func))), model_name)",
                "                }",
                "            cbind(as.data.frame(do.call(rbind, lapply(models, model_func))),other_preds)",
                "        }",
                "        cbind(as.data.frame(do.call(rbind, lapply(predictors, predictors_func))), ds)",
                "    }",
                "",
                "    out_accept <- as.data.frame(do.call(rbind, lapply(datasets, dataset_func)))",
                "    saveRDS(out_accept, checkpoint_file)",
                '    write.csv(out_accept, file.path("checkpoints", "acceptability_cv.csv"), row.names=FALSE)',
                '    message("Saved checkpoint: ", checkpoint_file)',
                "}",
            ]
        ),
        89: "\n".join(
            [
                "%%R",
                "powers_np_format <- c('0.0','0.25', '0.5' , '0.75','1.0', '1.25', '1.5' , '1.75', '2.0', '2.25', '2.5', '2.75')",
                "labels <- as.numeric(powers_np_format)",
            ]
        ),
        92: "\n".join(
            [
                "%%R -w 1800 -h 520 -u px",
                "plot_df <- rbind(select(out, -c('V4')), out_accept)",
                "p_reading_acceptability <- ggplot(aes(x = labels, y = V1, color=model_name, fill=model_name), data=plot_df) + ",
                "    geom_hline(yintercept=0, size=0.25, color='grey82') +",
                "    geom_vline(aes(xintercept=1),linetype=2, size=0.4, color='grey45') +",
                "    geom_ribbon(aes(ymin=V1-sqrt(V2), ymax=V1+sqrt(V2)), alpha = 0.16, color=NA) +",
                "    geom_line(size=0.8) +",
                "    geom_point(size=2.6) +",
                '    labs(title="",y="Per Sentence ∆LogLik",x=expression(italic("k"))) +',
                '    scale_color_discrete(name = "", labels=c(bert=\'Bert\',gpt=\'GPT-2\', ngram=expression(paste(italic("n"),"-gram")),transxl=\'TransXL\')) +',
                '    scale_fill_discrete(guide="none") +',
                '    guides(color=guide_legend(nrow=1, byrow=TRUE, override.aes=list(size=3))) +',
                '    facet_wrap(~ds, scales="free_y", nrow=1) +',
                '    theme_minimal(base_family="serif") +',
                "    theme(text=element_text(size=15), ",
                "         axis.text=element_text(size=10),",
                "         axis.title=element_text(size=16),",
                "         strip.text=element_text(size=15),",
                "         legend.title=element_blank(),",
                "         legend.text=element_text(size=12),",
                "         legend.position='bottom',",
                "         legend.direction='horizontal',",
                '         legend.key.width=unit(1.1, "lines"),',
                '         panel.spacing.x=unit(1.2, "lines"),',
                '         panel.spacing.y=unit(0.8, "lines"),',
                "         aspect.ratio=0.82,",
                "         plot.margin=margin(6, 8, 6, 6))",
                'save_plot(p_reading_acceptability, "figure_reading_acceptability_delta_loglik", width=15, height=4.4)',
            ]
        ),
        93: "\n".join(
            [
                "%%R",
                'checkpoint_file <- file.path("checkpoints", "lau_acceptability_cv.rds")',
                "if (file.exists(checkpoint_file)) {",
                "    lau_out <- readRDS(checkpoint_file)",
                '    message("Loaded checkpoint: ", checkpoint_file)',
                "} else {",
                "    df_full <- acceptability",
                "    models <- c('bert', 'ngram', 'gpt')",
                "    predictors <- list(b=c('1'))",
                "    datasets <- c('BNC', 'CoLA')",
                "    lau_preds <- c('slor', 'normlp')",
                "    dataset_func <- function(ds){",
                "        data_per_ds <- filter(df_full, df_full$dataset == ds)",
                "        if(nrow(data_per_ds) == 0){",
                "                    return(NULL)",
                "        }",
                "        family <- binomial",
                "        predictors_func <- function(preds){",
                "            other_preds <- paste0(preds, collapse='+')",
                "            model_func <- function(model_name){",
                "                d <- filter(data_per_ds, data_per_ds$model == model_name)",
                "                if(nrow(d) == 0){",
                "                    return(NULL)",
                "                }",
                "                set.seed(42)",
                "                shuffled_order <- sample(nrow(d))",
                '                baseline <- lm_cross_val(paste("accept ~ ", other_preds),',
                "                                  d[shuffled_order,], ",
                "                                  'accept', ",
                "                                  family)",
                "",
                "                inner <- function(x){",
                '                        formula <- paste0("accept ~ ", x," +", other_preds)',
                "                        cv <- lm_cross_val(formula, d[shuffled_order,], 'accept', family)",
                "                        c(mean(cv-baseline, na.rm=TRUE), var(cv-baseline, na.rm=TRUE)/length(cv),mean(cv, na.rm=TRUE))",
                "                    }",
                "                cbind(lau_preds, as.data.frame(do.call(rbind,lapply(lau_preds, inner))), model_name)",
                "                }",
                "            cbind(as.data.frame(do.call(rbind, lapply(models, model_func))),other_preds)",
                "        }",
                "        cbind(as.data.frame(do.call(rbind, lapply(predictors, predictors_func))), ds)",
                "    }",
                "",
                "    lau_out <- as.data.frame(do.call(rbind, lapply(datasets, dataset_func)))",
                "    saveRDS(lau_out, checkpoint_file)",
                '    write.csv(lau_out, file.path("checkpoints", "lau_acceptability_cv.csv"), row.names=FALSE)',
                '    message("Saved checkpoint: ", checkpoint_file)',
                "}",
            ]
        ),
        94: "\n".join(
            [
                "%%R -w 1250 -h 560 -u px",
                "p_acceptability_baselines <- ggplot(aes(x = labels, y = V1, color=model_name, fill=model_name), data=out_accept) + ",
                "    geom_vline(aes(xintercept = 1), linetype=2, size=0.4, color='grey45') +",
                "    geom_line(size=0.8) +",
                "    geom_point(size=3.2) +",
                "    geom_ribbon(aes(ymin=V1-sqrt(V2), ymax=V1+sqrt(V2)), alpha = 0.15, color=NA) +",
                '    geom_hline(aes(yintercept = V1, color=model_name, linetype=lau_preds), data=lau_out) +',
                '    theme_minimal(base_family="serif") +',
                '    scale_linetype_discrete(name = "", labels=c(normlp=\'NormLP\', slor=\'SLOR\')) +',
                '    scale_color_discrete(name = "Model", labels=c(bert=\'Bert\', gpt=\'GPT-2\', ngram=\'n-gram\')) +',
                '    scale_fill_discrete(name = "Model", labels=c(bert=\'Bert\', gpt=\'GPT-2\', ngram=\'n-gram\')) +',
                '    labs(x = expression(italic("k")), y="Per Sentence ∆LogLik", title="Acceptability Baselines") +',
                '    guides(color=guide_legend(nrow=1, byrow=TRUE), fill="none", linetype=guide_legend(nrow=1, byrow=TRUE)) +',
                "    theme(text=element_text(size=15), ",
                "         title=element_text(size=18),",
                "         axis.text=element_text(size=10),",
                "         axis.title=element_text(size=16),",
                "         strip.text=element_text(size=15),",
                "         legend.text=element_text(size=11),",
                "         legend.title=element_text(size=12),",
                "         legend.position='bottom',",
                '         panel.spacing.x=unit(1.0, "lines"),',
                "         aspect.ratio=0.86,",
                "         plot.margin=margin(6, 8, 6, 6)) +",
                '    facet_wrap(~ds, scales="free_y", nrow=1) + ',
                "    coord_cartesian(xlim=c(0,2.75))",
                'save_plot(p_acceptability_baselines, "figure_acceptability_baselines", width=10.5, height=4.8)',
            ]
        ),
        98: "\n".join(
            [
                "%%R",
                'checkpoint_file <- file.path("checkpoints", "case_study2_variance.rds")',
                "if (file.exists(checkpoint_file)) {",
                "    df <- readRDS(checkpoint_file)",
                '    message("Loaded checkpoint: ", checkpoint_file)',
                "} else {",
                "    aggregate_per_sentence <- filter(agg_per_subject_sentence_full, ",
                "                                     agg_per_subject_sentence_full$model == 'gpt', ",
                "                                     agg_per_subject_sentence_full$log_prob_mean < 15,",
                "                                    agg_per_subject_sentence_full$outlier_sum==0)",
                "    datasets <- intersect(c('Dundee','Brown','Provo','Natural Stories'), unique(as.character(aggregate_per_sentence$dataset)))",
                "    if (length(datasets) == 0) {",
                '        stop("No reading-time datasets available for case study 2.")',
                "    }",
                "    #datasets <- c('CoLA','BNC')",
                "    df <- NULL",
                "    names <- c()",
                "    for(d in datasets){",
                "        print(d)",
                "        data <- filter(aggregate_per_sentence, dataset == d)",
                "        if(nrow(data) == 0){",
                "            next",
                "        }",
                '        names <- c(names, paste0(d, "_mean"), paste0(d,"_var"))',
                "        set.seed(42)",
                "        shuffled_order <- sample(nrow(data))",
                '        baseline <- lme_cross_val("time_sum ~   time_count_nonzero +len + I(len*uni_log_prob_power_1.0)*ch_len + (  len+0 | WorkerId_) ", ',
                "                                  data[shuffled_order,],",
                "                                 'time_sum')",
                "        #baseline <- lm_cross_val(\"accept~1\", data[shuffled_order,], 'accept', binomial)",
                "        out1 <- list()",
                "        out1['var'] <- c()",
                "        out1['mean'] <- c()",
                "        for(v in attributes){",
                '            pred <- paste0("log_prob_",v,":len ")',
                '            formula <- paste0("time_sum ~ ",pred," + time_count_nonzero+len +I(len*uni_log_prob_power_1.0)*ch_len+ (",pred," +len+0 | WorkerId_) ")',
                "            diff <- lme_cross_val(formula, data[shuffled_order,], 'time_sum') - baseline",
                "            #diff <- lm_cross_val(formula, data[shuffled_order,], 'accept', binomial) - baseline",
                "            out1[['var']] <- c(out1[['var']],  var(diff, na.rm=TRUE)/length(diff))",
                "            out1[['mean']] <- c(out1[['mean']],  mean(diff, na.rm=TRUE))",
                "        }",
                "",
                "        df <- cbind(df, out1[['mean']], out1[['var']])",
                "    }",
                "    if (is.null(df)) {",
                '        stop("Case study 2 had no non-empty datasets after filtering.")',
                "    }",
                "    colnames(df) <- names",
                "    df <- cbind(attributes, df)",
                "    saveRDS(df, checkpoint_file)",
                '    write.csv(as.data.frame(df), file.path("checkpoints", "case_study2_variance.csv"), row.names=FALSE)',
                '    message("Saved checkpoint: ", checkpoint_file)',
                "}",
            ]
        ),
        100: "\n".join(
            [
                "# Original notebook switched one dataset at a time here.",
                "# Public notebook batches over every currently available dataset instead.",
                'model = "gpt"',
                "case_study3_sources = [",
                '    ("Brown", brown),',
                '    ("Dundee", dundee),',
                '    ("Natural Stories", natural_stories),',
                '    ("Provo", provo),',
                '    ("UCL (R)", ucl),',
                '    ("UCL (Eye)", ucl_eye),',
                "]",
                "case_study3_frames = []",
                "case_study3_labels = []",
                "for ds_label, candidate in case_study3_sources:",
                "    if not isinstance(candidate, pd.DataFrame) or candidate.empty:",
                "        continue",
                '    data_full = candidate.drop(["word"], axis=1, errors="ignore").copy()',
                "    filtered = data_full.loc[(data_full['model'] == model) & (data_full['outlier'] == False)].copy()",
                "    if filtered.empty:",
                "        continue",
                "    filtered['case_study3_dataset'] = ds_label",
                "    case_study3_frames.append(filtered)",
                "    case_study3_labels.append(ds_label)",
                "if not case_study3_frames:",
                '    raise ValueError("No non-empty word-level dataset available for case study 3.")',
                "case_study3_data = pd.concat(case_study3_frames, ignore_index=True)",
                'print("Case study 3 datasets:", ", ".join(case_study3_labels))',
                "print(case_study3_data['case_study3_dataset'].value_counts().to_string())",
                "",
                "%R -i case_study3_data",
                "%R -i case_study3_labels",
                "%R -i model",
            ]
        ),
        101: "\n".join(
            [
                "%%R",
                "atts <- c('rolling_lvar1', 'rolling_lvar2', 'rolling_lvar3', 'rolling_lvar4', 'cum_lvar', 'diff2_sen','diff2_par','diff2_lang')",
                "att_names <- c('-1','-2','-3','-4','-n','sent','doc','lang')",
                "levels <- seq(length(atts))",
                "powers <- seq(0.25, 2.75, by=0.25)",
                "powers_np_format2 <- c('1.0', '1.25', '1.5' , '1.75', '2.0', '2.25', '2.5' )",
                "other_preds <- paste(c('log_prob', 'prev_log_prob', 'prev_freq*prev_word_len','freq*word_len'), collapse=' + ')",
                "",
                "case_study3_dataset_func <- function(ds_label){",
                "    data <- filter(case_study3_data, case_study3_dataset == ds_label)",
                "    if(nrow(data) == 0){",
                "        return(NULL)",
                "    }",
                "    set.seed(42)",
                "    shuffled_order <- sample(nrow(data))",
                "    baseline <- lme_cross_val(paste0('time ~', other_preds, '+ (1 | WorkerId)'), data[shuffled_order,], 'time')",
                "",
                "    predictor_func <- function(name){",
                "        formula <- paste0('time ~ ',name,'+(1 +', name,'| WorkerId)+', other_preds)",
                "        cv <- lme_cross_val(formula, data[shuffled_order,], 'time')",
                "        diff <- cv-baseline",
                "        c(mean(diff[!is.infinite(diff)], na.rm=TRUE), var(diff[!is.infinite(diff)], na.rm=TRUE)/length(cv), mean(cv[!is.infinite(cv)], na.rm=TRUE))",
                "    }",
                "",
                "    out <- list()",
                "    out[['cum_lvar']] <- predictor_func('cum_lvar')",
                "    out[['rolling_lvar1']] <- predictor_func('rolling_lvar1')",
                "    out[['rolling_lvar2']] <- predictor_func('rolling_lvar2')",
                "    out[['rolling_lvar3']] <- predictor_func('rolling_lvar3')",
                "    out[['rolling_lvar4']] <- predictor_func('rolling_lvar4')",
                "    out[['diff_par']] <- predictor_func('diff_par')",
                "    out[['diff2_par']] <- predictor_func('diff2_par')",
                "    out[['diff_sen']] <- predictor_func('diff_sen')",
                "    out[['diff2_sen']] <- predictor_func('diff2_sen')",
                "    out[['diff2_lang']] <- predictor_func('diff2_lang')",
                "",
                "    out_df <- as_tibble(cbind(do.call(rbind, out[atts]), model='GPT-2', df=ds_label, levels))",
                "    out_df[c('V1','V2','V3')] <- lapply(out_df[c('V1','V2','V3')], as.numeric)",
                "    out_df",
                "}",
            ]
        ),
        102: "\n".join(
            [
                "%%R",
                'message("Case study 3 helper functions are defined above; this cell is kept as a no-op for notebook compatibility.")',
            ]
        ),
        103: "\n".join(
            [
                "%%R",
                'checkpoint_file <- file.path("checkpoints", "case_study3_all_vars.rds")',
                "if (file.exists(checkpoint_file)) {",
                "    all_vars <- readRDS(checkpoint_file)",
                '    message("Loaded checkpoint: ", checkpoint_file)',
                "} else {",
                "    out_list <- lapply(case_study3_labels, case_study3_dataset_func)",
                "    out_list <- Filter(Negate(is.null), out_list)",
                "    if(length(out_list) == 0){",
                '        stop("Case study 3 produced no dataset-level outputs.")',
                "    }",
                "    all_vars <- bind_rows(out_list)",
                "    saveRDS(all_vars, checkpoint_file)",
                '    write.csv(as.data.frame(all_vars), file.path("checkpoints", "case_study3_all_vars.csv"), row.names=FALSE)',
                '    message("Saved checkpoint: ", checkpoint_file)',
                "}",
            ]
        ),
        104: "\n".join(
            [
                "%%R",
                "all_vars$df <- factor(all_vars$df, levels = case_study3_labels)",
                "all_vars <- arrange(all_vars, df, levels)",
            ]
        ),
        105: "\n".join(
            [
                "%%R",
                'message("Case study 3 all_vars already combines every available dataset; nothing else to assemble here.")',
            ]
        ),
        106: "\n".join(
            [
                "%%R -w 1700 -h 560 -u px",
                'p_case_study3 <- ggplot(aes(x = levels, y = V1, color=df), data=all_vars[all_vars$model=="GPT-2",]) + ',
                "    geom_hline(yintercept=0, size=0.25, color='grey85') +",
                "    geom_point(size=4.2) +",
                "    geom_errorbar(aes(ymin=V1-sqrt(V2), ymax=V1+sqrt(V2)), width=0.16, size=0.55) +",
                '    theme_minimal(base_family="serif") +',
                '    labs(x = "Window", y="Per Token ∆LogLik", title="")+',
                "    theme(text=element_text(size=15), ",
                "         title=element_text(size=16),",
                "         axis.text.x = element_text(angle=30, hjust=1, size=11),",
                "         axis.text.y = element_text(size=11),",
                "         axis.title = element_text(size=16),",
                "         strip.text.x = element_text(size=15),",
                "         aspect.ratio = 0.82,",
                '         legend.position = "none",',
                '         panel.spacing.x = unit(1.1, "lines"),',
                "         plot.margin = margin(6, 8, 6, 6)) +",
                "    scale_x_discrete(labels=att_names) +",
                '    facet_wrap(~df, scales="free_y", nrow=1) ',
                'save_plot(p_case_study3, "figure_case_study3_windows", width=14, height=4.8)',
            ]
        ),
        110: "\n".join(
            [
                "bnc_corr = load_pickle('bnc.pkl') if has_checkpoint('bnc.pkl') else bnc.copy()",
                "bnc_r = bnc_corr.loc[:,names+names2+['len', 'accept','model']].copy()",
                "bnc_r.loc[:,names+names2] = -bnc_r[names+names2].multiply(bnc_r['len'], axis='index')",
                "total = bnc_r.groupby('model').agg(['count']).iloc[0]",
                "bnc_r = bnc_r.groupby('model').corr().accept.reset_index()",
                "bnc_r['se'] = np.sqrt((1-bnc_r.accept**2)/(total[0]/2 - 2))",
                "%R -i bnc_r",
            ]
        ),
        111: "\n".join(
            [
                "%%R -w 1250 -h 560 -u px",
                'corrs <- rbind(cbind(filter(cola_r, cola_r$level_1 %in% unlist(names)),df="CoLA",p=rep(p,length(unique(cola_r$model)))),',
                '              cbind(filter(bnc_r, bnc_r$level_1 %in% unlist(names)),df="BNC",p=rep(p,length(unique(bnc_r$model)))))',
                'corrs_base <- rbind(cbind(filter(cola_r, cola_r$level_1 %in% unlist(names2)),df="CoLA"),',
                '              cbind(filter(bnc_r, bnc_r$level_1 %in% unlist(names2)),df="BNC"))',
                "",
                "p_correlation <- ggplot(aes(x = p, y = accept, color=model, fill=model), data=corrs) + ",
                "    geom_vline(aes(xintercept = 1),linetype=2, size=0.4, color='grey45') +",
                "    geom_line(size=0.8) +",
                "    geom_point(size=3.2) +",
                "    geom_ribbon(aes(ymin=accept-se, ymax=accept+se), alpha = 0.16, color=NA) +",
                "    geom_hline(aes(yintercept = as.numeric(accept), color=model, linetype=level_1), data=corrs_base) +",
                '    theme_minimal(base_family="serif") +',
                '    scale_linetype_discrete(name = "", labels=c(\'NormLP\', \'SLOR\'))+',
                '    scale_color_discrete(name = "Model", labels=c(\'Bert\',\'GPT-2\', \'n-gram\',\'TransXL\')) +',
                '    scale_fill_discrete(name = "Model", labels=c(\'Bert\',\'GPT-2\', \'n-gram\',\'TransXL\')) +',
                '    labs(x = "Exponent", y="Pearson\'s Correlation", title="Surprisal-Acceptability Correlation")+',
                '    guides(color=guide_legend(nrow=1, byrow=TRUE), fill="none", linetype=guide_legend(nrow=1, byrow=TRUE))+',
                '    theme(text=element_text(size=15), ',
                '         title=element_text(size=18),',
                "         axis.text=element_text(size=10),",
                "         axis.title=element_text(size=16),",
                "         strip.text=element_text(size=15),",
                "         legend.text=element_text(size=11),",
                "         legend.title=element_text(size=12),",
                "         legend.position='bottom',",
                '         panel.spacing.x=unit(1.0, "lines"),',
                "         aspect.ratio=0.86,",
                "         plot.margin=margin(6, 8, 6, 6)) +",
                '    facet_wrap(~df, scales="free_y", nrow=1) + ',
                "    coord_cartesian(xlim=c(0.25,2.75))",
                'save_plot(p_correlation, "figure_surprisal_acceptability_correlation", width=10.5, height=4.8)',
            ]
        ),
    }

    dundee_dir = REPO_ROOT / "src" / "corpora" / "dundee"
    brown_path = REPO_ROOT / "src" / "corpora" / "brown_spr.csv"
    geco_materials = REPO_ROOT / "src" / "corpora" / "DutchMaterials.csv"
    geco_reading = REPO_ROOT / "src" / "corpora" / "L1ReadingData.csv"

    if not dundee_dir.exists():
        for cell_index in range(48, 54):
            replacements[cell_index] = empty_dataset_block("dundee", "Dundee corpus")
    else:
        replacements[53] = replace_with_python_comment("Using freshly computed Dundee statistics.")

    if not brown_path.exists():
        for cell_index in range(55, 60):
            replacements[cell_index] = empty_dataset_block("brown", "Brown corpus")
    else:
        replacements[59] = replace_with_python_comment("Using freshly computed Brown statistics.")

    if not (geco_materials.exists() and geco_reading.exists()):
        for cell_index in range(61, 67):
            replacements[cell_index] = empty_dataset_block("geco", "GECO corpus")
    else:
        replacements[66] = replace_with_python_comment("Using freshly computed GECO statistics.")

    for index, source in replacements.items():
        nb.cells[index].source = source

    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUTPUT_NOTEBOOK)
    print(f"Wrote {OUTPUT_NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
