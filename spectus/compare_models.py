import sys
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
from pathlib import Path
import typer
from typing import List, Union, Dict
import yaml
import copy
import numpy as np
import ast

app = typer.Typer(pretty_exceptions_enable=False)


def discard_other_than_latest_eval(log_dict):
    """Discard all but the latest evaluation from the log file."""
    eval_keys = [key for key in log_dict.keys() if key.startswith("evaluation_")]
    eval_keys.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
    log_dict["evaluation"] = log_dict[eval_keys[0]]

    for key in eval_keys:
        del log_dict[key]


def fix_model_duplicity(models, new_model_name, log_dicts):
    """Fix duplicity of model names. If the model name is already used, add
    more info to all the occurences that don't have it yet and
    If there is no duplicity match, return the model name unchanged.

    The priority of added info is: 1. additional_naming_info, 2. dataset, 3. num_candidates

    Parameters
    ----------
    models: list
        List of model names.
    new_model_name: str
        New model name.
    log_dicts: list
        List of log dictionaries corresponding to the models.

    Returns
    -------
    new_model_name: str
        Fixed model name without duplicity.
    """
    duplicate_names = [(i, name) for i, name in enumerate(models) if new_model_name in name]
    duplicate_logdicts = [log_dicts[i] for (i, _) in duplicate_names]

    if not duplicate_names:
        return new_model_name

    duplicate_names.append((-1, new_model_name))
    duplicate_logdicts.append(log_dicts[-1])


    additional_infos = [log_dict["general"].get("additional_naming_info") for log_dict in duplicate_logdicts]
    datasets = [log_dict["dataset"].get("dataset_name") for log_dict in duplicate_logdicts]
    num_candidates = [log_dict["general"].get("num_candidates") for log_dict in duplicate_logdicts]

    # check if additional_naming_info will distinguishes the models
    if None not in additional_infos and len(additional_infos) == len(set(additional_infos)): # all are unique and not None
        new_infos = additional_infos
    # check if dataset will distinguishes the models
    elif len(datasets) == len(set(datasets)):
        new_infos = datasets
    elif len(num_candidates) == len(set(num_candidates)):
        new_infos = [str(nc) + "cands" for nc in num_candidates]
    else:
        new_infos = [f"{ai}_{ds}_{nc}" for ai, ds, nc in zip(additional_infos, datasets, num_candidates)]
        print("Warning: The model names might not be unique. Additional info, dataset and num_candidates are not unique, adding all of them to the model names.")

    for (i, used_name), new_info in zip(duplicate_names[:-1], new_infos[:-1]):
        if str(new_info) in used_name:
            continue
        models[i] = f"{used_name}_{new_info}"

    return f"{duplicate_names[-1][1]}_{new_infos[-1]}"


def load_predictions(model_dir):
    """Loads log files and prediction data."""
    model_name = Path(model_dir).parent.parent.name

    # Load predictions from df_best_predictions.jsonl
    pred_file_path = Path(model_dir) / 'df_best_predictions.jsonl'
    log_file_path = Path(model_dir) / 'log_file.yaml'

    best_predictions = pd.read_json(pred_file_path, lines=True)
    with open(log_file_path, "r", encoding="utf-8") as f:
        log_dict = yaml.safe_load(f)
    discard_other_than_latest_eval(log_dict)

    return model_name, best_predictions, log_dict


def load_all_predictions(model_dirs):
    """Loads logfiles and prediction data of all given models."""
    models = []
    dfs = []
    log_dicts = []

    if not model_dirs:
        return models, dfs, log_dicts

    print(model_dirs)

    for model_dir in model_dirs:
        try:
            model_name, df, log_dict = load_predictions(model_dir)
            log_dicts.append(log_dict)
            dfs.append(df)
            model_name = fix_model_duplicity(models, model_name, log_dicts)
            models.append(model_name)
        except Exception as e:
            raise ValueError(f"Problem with loading {model_dir}: {repr(e)}")

    return models, dfs, log_dicts


def validate_ground_truth(dfs):
    """Ensure that all models have the same ground truth (gt_smiles) values."""
    gt_smiles = dfs[0].get('gt_smiles')
    for i, df in enumerate(dfs[1:], start=1):
        if gt_smiles is None or not gt_smiles.equals(df.get('gt_smiles')):
            raise ValueError(f"Ground truth mismatch between models at index 0 and {i}.")
    print("Ground truth columns match across all models.")


def create_all_similarities_dataframe(dfs, models, column):
    """Creates a DataFrame of all predictions' similarities for the given column."""
    similarity_df = pd.DataFrame()
    for model, df in zip(models, dfs):
        similarity_df[model] = df[column]
    return similarity_df


def perform_wilcoxon_test(similarity_df):
    """Performs the Wilcoxon signed-rank test on the similarity DataFrame."""
    cols = similarity_df.columns
    stat, p_value = wilcoxon(similarity_df[cols[0]], similarity_df[cols[1]])
    return stat, p_value


def perform_friedman_test(similarity_df):
    """Performs the Friedman test on the similarity DataFrame."""
    stat, p_value = friedmanchisquare(*[similarity_df[col] for col in similarity_df.columns])
    return stat, p_value


def perform_nemenyi_test(similarity_df):
    """Performs the Nemenyi post-hoc test."""
    posthoc = sp.posthoc_nemenyi_friedman(similarity_df)
    return posthoc


def significance_stars(p_value):
    """
    Given a p-value, return the significance level in stars:
    - *** for p < 0.001
    - ** for p < 0.01
    - * for p < 0.05
    - . for p < 0.1
    - no stars for p >= 0.1
    """
    if p_value < 0.001:
        return "*** highly significant"
    elif p_value < 0.01:
        return "** very significant"
    elif p_value < 0.05:
        return "* significant"
    elif p_value < 0.1:
        return ". close but not significant"
    else:
        return "not significant"

def check_significance_of_inequality(similarity_df):
    """Check the significance of the inequality between models.

    Parameters
    ----------
    similarity_df: pd.DataFrame
        DataFrame with gt similarities of all the predictions (rows) for all models (columns).

    Returns
    -------
    test_name: str
        Name of the performed test (Wilcoxon/Friedman).
    stat: float
        Test statistic.
    p_value: float
        P-value of the test.
    nemenyi_prob: pd.DataFrame or None
        Nemenyi post-hoc test results if significant difference, None otherwise."""

    num_models = len(similarity_df.columns)
    if num_models < 2:
        raise ValueError("At least two models are required for comparison.")

    elif num_models == 2:
        test_name = "Wilcoxon signed-rank test"
        print("Only two models found. Performing Wilcoxon signed-rank test...")
        stat, p_value = perform_wilcoxon_test(similarity_df)
        nemenyi_prob = None

    elif num_models > 2:
        test_name = "Friedman test"
        print("More than two models found. Performing Friedman test...")
        stat, p_value = perform_friedman_test(similarity_df)

        # If significant difference, perform Nemenyi post-hoc test
        if p_value < 0.05:
            print("Performing Nemenyi post-hoc test...")
            nemenyi_prob = perform_nemenyi_test(similarity_df)
        else:
            nemenyi_prob = None
            print("No significant difference in probsort similarities.")

    return test_name, stat, p_value, nemenyi_prob


def compute_rate_of_wins(similarities1: pd.Series, similarities2: pd.Series):
    """Compute the rate of wins of the first list over the second list."""
    win_rate = (similarities1 > similarities2).sum() / len(similarities1)
    at_least_as_good_rate = (similarities1 >= similarities2).sum() / len(similarities1)

    return win_rate, at_least_as_good_rate

def compute_rate_of_real_wins(df_best_predictions1: pd.DataFrame, df_best_predictions2: pd.DataFrame, type: str = "simil"):
    """Compute the rate of wins of the first list over the second list.
    A real win takes into consideration not only the similarity but also the identity of the candidate.
    The type parameter can be either 'simil' or 'prob'."""

    alag_similarity = df_best_predictions1[f"{type}_best_simil_morgan_tanimoto"] >= df_best_predictions2[f"{type}_best_simil_morgan_tanimoto"]
    higher_similarity = df_best_predictions1[f"{type}_best_simil_morgan_tanimoto"] > df_best_predictions2[f"{type}_best_simil_morgan_tanimoto"]
    exact_match1 = df_best_predictions1[f"{type}_best_smiless_morgan_tanimoto"] == df_best_predictions1["gt_smiles"]
    exact_match2 = df_best_predictions2[f"{type}_best_smiless_morgan_tanimoto"] == df_best_predictions2["gt_smiles"]
    real_wins = higher_similarity | (exact_match1 & ~exact_match2)
    real_alag = alag_similarity & ~(~exact_match1 & exact_match2)

    win_rate = real_wins.sum() / len(df_best_predictions1)
    at_least_as_good_rate = real_alag.sum() / len(df_best_predictions1)

    return win_rate, at_least_as_good_rate

def compute_mean_difference(similarities1: pd.Series, similarities2: pd.Series):
    """Compute the mean difference between two lists of similarities."""
    return (similarities1 - similarities2).mean()


def compute_rate_of_wins_for_two_groups(similarity_df, models1, models2):
    """Compute the rate of wins of the first group over the second group."""
    wins_df = pd.DataFrame(index=models1, columns=models2)
    at_least_as_good_df = pd.DataFrame(index=models1, columns=models2)

    for model1 in models1:
        for model2 in models2:
            win_rate, at_least_as_good_rate = compute_rate_of_wins(similarity_df[model1], similarity_df[model2])
            wins_df.loc[model1, model2] = win_rate
            at_least_as_good_df.loc[model1, model2] = at_least_as_good_rate

    return wins_df, at_least_as_good_df

def compute_rate_of_real_wins_for_two_groups(group1_dfs, group1_names, group2_dfs, group2_names):
    """Compute the rate of real wins of the first group over the second group.
    A real win takes into consideration not only the similarity but also the identity of the candidate."""
    wins_df = pd.DataFrame(index=group1_names, columns=group2_names)
    at_least_as_good_df = pd.DataFrame(index=group1_names, columns=group2_names)

    for model1, df_1 in zip(group1_names, group1_dfs):
        for model2, df_2 in zip(group2_names, group2_dfs):
            win_rate, at_least_as_good_rate = compute_rate_of_real_wins(df_1, df_2)
            wins_df.loc[model1, model2] = win_rate
            at_least_as_good_df.loc[model1, model2] = at_least_as_good_rate
    return wins_df, at_least_as_good_df

def compute_mean_differences_for_two_groups(similarity_df, models1, models2):
    """Compute the mean differences between the first group and the second group."""
    mean_diff_df = pd.DataFrame(index=models1, columns=models2)

    for model1 in models1:
        for model2 in models2:
            mean_diff = compute_mean_difference(similarity_df[model1], similarity_df[model2])
            mean_diff_df.loc[model1, model2] = mean_diff

    return mean_diff_df


def compute_db_search_performance(similarity_df, db_search_names, fp_type="morgan", fp_simil="tanimoto"):
    """Creates a table showing how close is the performance of spectral
    db searches (sss, hss) to the structural db searches (morgan_tanimoto).
    The table contains the rates of structural searches finding the
    closest structural candidate."""

    structural_searches = []
    spectral_searches = []
    for name in db_search_names:
        if f"{fp_type}_{fp_simil}" in name:
            structural_searches.append(name)
        else:
            spectral_searches.append(name)
    if not structural_searches or not spectral_searches:
        print("Found only structural or only spectral searches -> db search performance comparison not possible.")
        return None

    _, rate_of_closest_candidates = compute_rate_of_wins_for_two_groups(similarity_df, spectral_searches, structural_searches)
    return rate_of_closest_candidates


def get_info_from_logfile(log_dict: dict, value_path: str):
    """Extract a value from the log file based on the given path."""
    keys = value_path.split("/")
    value = copy.deepcopy(log_dict)
    for key in keys:
        value = value.get(key, False)
        if not value:
            raise ValueError(f"Value {key} at path {value_path} not found in the log file.")

    try:
        value = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        pass

    assert isinstance(value, (str, int, float, list)), f"Value at path {value_path} is not a string, int, float or list. We got value: {value}."
    return value


def get_info_from_logfiles_for_all(log_dicts: Dict[str, Dict], value_paths: List[str]) -> pd.DataFrame:
    """Create a comparison table for all models with values from the log files.

    Parameters
    ----------
    log_dicts: dict
        Dictionary of log files. Keys are model names, values are log dictionaries
    value_paths: List[str]
        List of paths to the values in the log files to be extracted.
        The format is 'key1/key2/key3/...' for nested dictionaries.

    Return
    ------
    info_df: pd.DataFrame
        DataFrame with the extracted values (columns) for all models (rows).
    """
    value_names = [path.split("/")[-1] for path in value_paths]
    df = pd.DataFrame(index = list(log_dicts.keys()), columns = value_names)
    for model, log_dict in log_dicts.items():
        for path, name in zip(value_paths, value_names):
            try:
                df.loc[model, name] = get_info_from_logfile(log_dict, path)
            except ValueError as e:
                print(f"Error in model {model}: {e}")
                df.loc[model, name] = None
    return df


def sort_all_dfs(dfs_columns: List[tuple]):
    """Sort all DataFrames in the list by the given column."""
    for df, column in dfs_columns:
        if df is not None:
            df.sort_values(by=column, ascending=False, inplace=True)


def compare_log_dicts(models_prediction_paths, value_paths: List[str]):
    model_names, _, model_log_dicts = load_all_predictions(models_prediction_paths)
    named_log_dicts = {k: v for k, v in zip(model_names, model_log_dicts)}
    return get_info_from_logfiles_for_all(named_log_dicts, value_paths)


def compare_models(models_prediction_paths: Union[List[str], None] = None,
                   db_search_prediction_paths: Union[List[str], None] = None,
                   fp_type: str = "morgan",
                   fp_simil: str = "tanimoto"):
    """
    Compare models based on their predictions and optionally database search predictions.

    Parameters:
    ----------
    models_prediction_paths: List[str]
        Paths to the directories containing model predictions.
    db_search_prediction_paths: Union[List[str], None]
        Paths to the directories containing database search predictions.
    fp_type: str
        Preferred fingerprint type used for similarity calculations.
    fp_simil: str
        Preferred similarity metric used for similarity calculations.

    Returns:
    -------
    output: dict
    Output dictionary with results and tables for the model comparison:

        output = {
            models: list_of_model_names,
            db_searches: list_of_db_search_names,
            mean_simils: df (sorted by similsort, both db_searches and models, bot probsort and similsort),
            significance_test: {
                test_name: str (Friedman/Wilcoxon),
                stat_probsort: float,
                p_value_probsort: float,
                stat_similsort: float,
                p_value_similsort: float,
                nemenyi_probsort: df or none,
                nemenyi_similsort: df or none
            },
            wins_over_db_search_probsort: df or none,
            wins_over_db_search_similsort: df or none,
            at_least_as_good_as_db_search_probsort: df or none,
            at_least_as_good_as_db_search_similsort: df or none,
            fpsd_score_probsort: df or none,
            fpsd_score_similsort: df or none,
            db_search_performance: df or none (what was the portion of predictions where db_search hit the MT closest molecule)
    """

    model_names, model_dfs, model_log_dicts = load_all_predictions(models_prediction_paths)
    db_search_names, db_search_dfs, db_search_log_dicts = load_all_predictions(db_search_prediction_paths)

    all_predictors_names, all_dfs = model_names + db_search_names, model_dfs + db_search_dfs

    if len(all_predictors_names) == 1:
        raise ValueError("Only one model found - not much to compare. At least two models are required for comparison.")
    if len(all_predictors_names) == 0:
        raise ValueError("No models or database searches found. Specify at least 2 models and or database searches using models_prediction_paths and db_search_prediction_paths.")

    all_log_dicts = {k: v for k, v in zip(all_predictors_names, model_log_dicts + db_search_log_dicts)}

    # Validate that ground truth columns are the same across all models
    validate_ground_truth(all_dfs)

    # Create two DataFrames: one for 'prob_best_simil_{fp_type}_{fp_simil}', one for 'simil_best_simil_{fp_type}_{fp_simil}'
    probsort_df = create_all_similarities_dataframe(all_dfs, all_predictors_names, f'prob_best_simil_{fp_type}_{fp_simil}')
    similsort_df = create_all_similarities_dataframe(all_dfs, all_predictors_names, f'simil_best_simil_{fp_type}_{fp_simil}')

    # do significance test for probsort and similsort
    test_name, stat_probsort, p_value_probsort, nemenyi_prob_probsort = check_significance_of_inequality(probsort_df)
    _, stat_similsort, p_value_similsort, nemenyi_prob_similsort = check_significance_of_inequality(similsort_df)

    mean_simils_df = pd.DataFrame({"similsort_mean_simil": similsort_df.mean().values,
                                   "probsort_mean_simil": probsort_df.mean().values},
                                   index=similsort_df.columns)
    exact_matches_df = get_info_from_logfiles_for_all(all_log_dicts,
                                                      ["evaluation/precise_preds_stats/rate_of_precise_preds_similsort",
                                                       "evaluation/precise_preds_stats/rate_of_precise_preds_probsort"] )

    if len(model_names) > 0 and len(db_search_names) > 0:
        wins_over_db_search_probsort_df, at_least_as_good_probsort_df = compute_rate_of_wins_for_two_groups(probsort_df, model_names, db_search_names)
        wins_over_db_search_similsort_df, at_least_as_good_similsort_df = compute_rate_of_wins_for_two_groups(similsort_df, model_names, db_search_names)
        real_wins_over_db_search_similsort_df, real_at_least_as_good_similsort_df = compute_rate_of_real_wins_for_two_groups(model_dfs, model_names, db_search_dfs, db_search_names)
        fpsd_score_probsort_df = compute_mean_differences_for_two_groups(probsort_df, model_names, db_search_names)
        fpsd_score_similsort_df = compute_mean_differences_for_two_groups(similsort_df, model_names, db_search_names)
    else:
        wins_over_db_search_probsort_df, wins_over_db_search_similsort_df = None, None
        real_wins_over_db_search_similsort_df, real_at_least_as_good_similsort_df = None, None
        at_least_as_good_probsort_df, at_least_as_good_similsort_df = None, None
        fpsd_score_probsort_df, fpsd_score_similsort_df = None, None


    if len(db_search_names) > 0:
        db_search_performance_df = compute_db_search_performance(similsort_df, db_search_names, fp_type, fp_simil)
    else:
        db_search_performance_df = None


    sort_all_dfs([(mean_simils_df, "similsort_mean_simil"),
                  (exact_matches_df, "rate_of_precise_preds_similsort"),
                  (wins_over_db_search_probsort_df, wins_over_db_search_probsort_df.columns[0] if wins_over_db_search_probsort_df is not None else ""),
                  (wins_over_db_search_similsort_df, wins_over_db_search_similsort_df.columns[0] if wins_over_db_search_similsort_df is not None else ""),
                  (at_least_as_good_probsort_df, at_least_as_good_probsort_df.columns[0] if at_least_as_good_probsort_df is not None else ""),
                  (at_least_as_good_similsort_df, at_least_as_good_similsort_df.columns[0] if at_least_as_good_similsort_df is not None else ""),
                  (real_wins_over_db_search_similsort_df, real_wins_over_db_search_similsort_df.columns[0] if real_wins_over_db_search_similsort_df is not None else ""),
                  (real_at_least_as_good_similsort_df, real_at_least_as_good_similsort_df.columns[0] if real_at_least_as_good_similsort_df is not None else ""),
                  (fpsd_score_probsort_df, fpsd_score_probsort_df.columns[0] if fpsd_score_probsort_df is not None else ""),
                  (fpsd_score_similsort_df, fpsd_score_similsort_df.columns[0] if fpsd_score_similsort_df is not None else ""),
                  (db_search_performance_df, db_search_performance_df.columns[0] if db_search_performance_df is not None else "")])


    output = {
        "models": model_names,
        "db_searches": db_search_names,
        "mean_simils": mean_simils_df,
        "exact_matches": exact_matches_df,
        "significance_test": {
            "test_name": test_name,
            "stat_probsort": stat_probsort,
            "p_value_probsort": p_value_probsort,
            "stat_similsort": stat_similsort,
            "p_value_similsort": p_value_similsort,
            "nemenyi_probsort": nemenyi_prob_probsort,
            "nemenyi_similsort": nemenyi_prob_similsort
        },
        "wins_over_db_search_probsort": wins_over_db_search_probsort_df,
        "wins_over_db_search_similsort": wins_over_db_search_similsort_df,
        "at_least_as_good_as_db_search_probsort": at_least_as_good_probsort_df,
        "at_least_as_good_as_db_search_similsort": at_least_as_good_similsort_df,
        "real_wins_over_db_search_similsort": real_wins_over_db_search_similsort_df,
        "real_at_least_as_good_similsort": real_at_least_as_good_similsort_df,
        "fpsd_score_probsort": fpsd_score_probsort_df,
        "fpsd_score_similsort": fpsd_score_similsort_df,
        "db_search_performance": db_search_performance_df,
        "log_dicts": all_log_dicts
    }

    return output


@app.command()
def main(additional_info: str = typer.Option(..., help="Additional information to be added to the output file name."),
         models_prediction_paths: str = typer.Option("", help="Paths to the directories containing model predictions separated by any whitespace."),
         db_search_prediction_paths: str = typer.Option("", help="Paths to the directories containing database search predictions separated by any whitespace."),
         save_path: str = typer.Option("", help="Path to the directory where the output file will be saved."),
         fp_type: str = typer.Option("morgan", help="Fingerprint type used for similarity calculations."),
         fp_simil: str = typer.Option("tanimoto", help="Similarity metric used for similarity calculations."),
         print_latex: bool = typer.Option(False, help="Print the output also in LaTeX format.")):

    models_paths_list = models_prediction_paths.split()
    db_search_paths_list = db_search_prediction_paths.split()

    comparison  = compare_models(models_paths_list, db_search_paths_list, fp_type, fp_simil)
    significance_test = comparison["significance_test"]
    # Save results to a file
    one_of_the_paths = Path((models_paths_list + db_search_paths_list)[0])
    dataset_name = one_of_the_paths.parent.name
    outfile_name = f"model_comparison_on_{dataset_name}{'_' + additional_info if additional_info else ''}.txt"

    # MARKDOWN OUTPUT
    outfile_path = one_of_the_paths.parent.parent.parent.parent / "results" / outfile_name if save_path == "" else Path(save_path) / outfile_name
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile_path, 'w') as f:
        f.write("### Statistical significance of model comparison ###\n\n\n".upper())
        f.write(f"Mean {fp_type} {fp_simil} similarities with ground truth\n")
        f.write(comparison["mean_simils"].to_markdown() + "\n\n")
        f.write("Exact matches\n")
        f.write(comparison["exact_matches"].to_markdown() + "\n\n")
        f.write("Results of the statistical test:\n")
        f.write("--------------------------------\n")
        f.write(f"{significance_test['test_name']} (probsort): statistic={significance_test['stat_probsort']}, p-value={significance_test['p_value_probsort']}   {significance_stars(significance_test['p_value_probsort'])}\n")
        f.write(f"{significance_test['test_name']} (similsort): statistic={significance_test['stat_similsort']}, p-value={significance_test['p_value_probsort']}   {significance_stars(significance_test['p_value_similsort'])}")
        if len(comparison['models']) + len(comparison["db_searches"]) > 2:
            if significance_test['p_value_probsort'] < 0.05:
                f.write(f"\n\n\nThe {significance_test['test_name']} was SIGNIFICANT -> Nemenyi post-hoc test (probsort):\n")
                f.write(significance_test['nemenyi_probsort'].to_markdown())
            if significance_test['p_value_similsort'] < 0.05:
                f.write(f"\n\n\nThe {significance_test['test_name']} was SIGNIFICANT -> Nemenyi post-hoc test (similsort):\n")
                f.write(significance_test['nemenyi_similsort'].to_markdown())

        if comparison['db_searches'] and comparison['models']:
            f.write("\n\n\nRate of wins over database search predictions (probsort):\n")
            f.write(comparison['wins_over_db_search_probsort'].to_markdown() + "\n")
            f.write("\n\nRate of wins over database search predictions (similsort):\n")
            f.write(comparison['wins_over_db_search_similsort'].to_markdown() + "\n")
            f.write("\n\nRate of at least as good as database search predictions (probsort):\n")
            f.write(comparison['at_least_as_good_as_db_search_probsort'].to_markdown() + "\n")
            f.write("\n\nRate of at least as good as database search predictions (similsort):\n")
            f.write(comparison['at_least_as_good_as_db_search_similsort'].to_markdown() + "\n")
            f.write("\n\nRate of REAL wins over database search predictions - including exact matches (similsort):\n")
            f.write(comparison['real_wins_over_db_search_similsort'].to_markdown() + "\n")
            f.write("\n\nRate of REAL at least as good as database search predictions - including exact matches (similsort):\n")
            f.write(comparison['real_at_least_as_good_similsort'].to_markdown() + "\n")
            f.write("\n\nMean difference in similarity to database search predictions (FPSD probsort):\n")
            f.write(comparison['fpsd_score_probsort'].to_markdown() + "\n")
            f.write("\n\nMean difference in similarity to database search predictions (FPSD similsort):\n")
            f.write(comparison['fpsd_score_similsort'].to_markdown())
        if comparison['db_search_performance'] is not None:
            f.write(f"\n\n\nPerformance of database searches (spectral searches' rate of finding best candidates according to {fp_type} {fp_simil} structural search):\n")
            f.write(comparison['db_search_performance'].to_markdown())

    # LATEX OUTPUT
    if print_latex:
        latex_outfile_path = outfile_path.with_suffix('.tex')
        with open(latex_outfile_path, 'w') as f:
            f.write("### Statistical significance of model comparison ###\n\n\n".upper())
            f.write(f"Mean {fp_type} {fp_simil} similarities with ground truth\n")
            f.write(comparison["mean_simils"].to_latex() + "\n\n")
            f.write("Exact matches\n")
            f.write(comparison["exact_matches"].to_latex() + "\n\n")
            f.write("Results of the statistical test:\n")
            f.write("--------------------------------\n")
            f.write(f"{significance_test['test_name']} (probsort): statistic={significance_test['stat_probsort']}, p-value={significance_test['p_value_probsort']}   {significance_stars(significance_test['p_value_probsort'])}\n")
            f.write(f"{significance_test['test_name']} (similsort): statistic={significance_test['stat_similsort']}, p-value={significance_test['p_value_probsort']}   {significance_stars(significance_test['p_value_similsort'])}")
            if len(comparison['models']) + len(comparison["db_searches"]) > 2:
                if significance_test['p_value_probsort'] < 0.05:
                    f.write(f"\n\n\nThe {significance_test['test_name']} was SIGNIFICANT -> Nemenyi post-hoc test (probsort):\n")
                    f.write(significance_test['nemenyi_probsort'].to_latex())
                if significance_test['p_value_similsort'] < 0.05:
                    f.write(f"\n\n\nThe {significance_test['test_name']} was SIGNIFICANT -> Nemenyi post-hoc test (similsort):\n")
                    f.write(significance_test['nemenyi_similsort'].to_latex())

            if comparison['db_searches'] and comparison['models']:
                f.write("\n\n\nRate of wins over database search predictions (probsort):\n")
                f.write(comparison['wins_over_db_search_probsort'].to_latex() + "\n")
                f.write("\n\nRate of wins over database search predictions (similsort):\n")
                f.write(comparison['wins_over_db_search_similsort'].to_latex() + "\n")
                f.write("\n\nRate of at least as good as database search predictions (probsort):\n")
                f.write(comparison['at_least_as_good_as_db_search_probsort'].to_latex() + "\n")
                f.write("\n\nRate of at least as good as database search predictions (similsort):\n")
                f.write(comparison['at_least_as_good_as_db_search_similsort'].to_latex() + "\n")
                f.write("\n\nRate of REAL wins over database search predictions - including exact matches (similsort):\n")
                f.write(comparison['real_wins_over_db_search_similsort'].to_latex() + "\n")
                f.write("\n\nRate of REAL at least as good as database search predictions - including exact matches (similsort):\n")
                f.write(comparison['real_at_least_as_good_similsort'].to_latex() + "\n")
                f.write("\n\nMean difference in similarity to database search predictions (FPSD probsort):\n")
                f.write(comparison['fpsd_score_probsort'].to_latex() + "\n")
                f.write("\n\nMean difference in similarity to database search predictions (FPSD similsort):\n")
                f.write(comparison['fpsd_score_similsort'].to_latex())
            if comparison['db_search_performance'] is not None:
                f.write(f"\n\n\nPerformance of database searches (spectral searches' rate of finding best candidates according to {fp_type} {fp_simil} structural search):\n")
                f.write(comparison['db_search_performance'].to_latex())


if __name__ == "__main__":
    app()
