#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合成データ品質評価スクリプト

目的:
    ・元データ（実データ）と合成データの統計的・構造的な再現性を評価する
    ・合成データの妥当性（不自然な値・制約違反）の有無を確認する
    ・評価結果を「数値」と「可視化」の両方で確認できるようにする

ポイント:
    ・数値が出たときに「それが良いのか悪いのか」がわかるよう、
      閾値ベースの簡易評価（◎/○/△/×）を付与する
    ・絶対的な指標がないものは、可視化を画像ファイルとして出力し、
      見比べて判断できるようにする
    ・GitHubでそのまま共有できるよう、1ファイルのPythonスクリプト形式で記述
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional


# =====================================================
# 設定（必要に応じて書き換えてください）
# =====================================================

# 入力ファイルパス
MASTER_REAL_PATH = "master_real.csv"
MASTER_SYN_PATH  = "master_syn.csv"
YEARLY_REAL_PATH = "yearly_real.csv"
YEARLY_SYN_PATH  = "yearly_syn.csv"

# 出力先ディレクトリ（図の保存先）
FIG_DIR = "figs"

# 評価に使うカラム名（スキーマに合わせて調整）
YEAR_COL         = "YEAR"
CODE_COL         = "コード"
INDUSTRY_COL     = "業種分類"
MARKET_COL       = "市場・商品区分"
VALUE_COL_MAIN   = "売上高"   # 時系列評価・業種別評価のメイン指標

TOTAL_ASSET_COL  = "総資産"
DEBT_COL         = "負債"
EQUITY_COL       = "純資産"

# レンジチェックのルール（下限, 上限）
RANGE_RULES_YEARLY: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "売上高": (0, None),
    "営業利益": (None, None),  # 赤字許容なら下限なし
    "総資産": (0, None),
    "負債": (0, None),
    # "純資産": (None, None),  # マイナス純資産を許容するなら下限なし
}

# 外れ値率を確認するカラム
OUTLIER_COLS = ["売上高", "営業利益", "総資産"]


# =====================================================
# ユーティリティ（評価ラベルなど）
# =====================================================

def score_ks(stat: float) -> str:
    """
    KS統計量に対する簡易評価ラベルを返す。
    一般的な「経験的な目安」に基づく主観的な評価。
    """
    if stat < 0.05:
        return "◎ 非常によく一致（ほぼ同じ分布とみなせる）"
    elif stat < 0.10:
        return "○ 概ね一致（用途によっては十分）"
    elif stat < 0.20:
        return "△ やや差がある（重要指標なら要検討）"
    else:
        return "× 大きく異なる（再学習や修正候補）"


def score_js(dist: float) -> str:
    """
    Jensen-Shannon 距離に対する簡易評価ラベル。
    0に近いほどカテゴリ分布が似ている。
    """
    if dist < 0.05:
        return "◎ 非常に近いカテゴリ分布"
    elif dist < 0.15:
        return "○ おおむね近いカテゴリ分布"
    elif dist < 0.30:
        return "△ かなり差があるカテゴリも存在"
    else:
        return "× カテゴリ構成が大きく異なる"


def score_corr_mean_abs_diff(diff: float) -> str:
    """
    相関行列の平均絶対差分に対する簡易評価ラベル。
    0に近いほどカラム同士の関係性が似ている。
    """
    if diff < 0.05:
        return "◎ 相関構造は非常によく再現されている"
    elif diff < 0.10:
        return "○ おおむね再現できている"
    elif diff < 0.20:
        return "△ 相関構造にややズレがある"
    else:
        return "× 相関構造がかなり異なる"


def score_ratio_small_is_good(ratio: float) -> str:
    """
    違反率など「小さいほど良い」指標に対する評価ラベル。
    """
    if ratio < 0.01:
        return "◎ ほぼ問題なし"
    elif ratio < 0.05:
        return "○ 一部に問題はあるが、全体としては許容範囲"
    elif ratio < 0.15:
        return "△ それなりの頻度で問題がある（要注意）"
    else:
        return "× 多数の問題が存在（要修正）"


def ensure_fig_dir():
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR, exist_ok=True)


# =====================================================
# データ読み込み
# =====================================================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CSVからデータを読み込む。
    実プロジェクトでは、ファイルパスやエンコーディング等を環境に合わせて調整してください。
    """
    print("=== データ読み込み ===")
    master_real = pd.read_csv(MASTER_REAL_PATH)
    master_syn  = pd.read_csv(MASTER_SYN_PATH)
    yearly_real = pd.read_csv(YEARLY_REAL_PATH)
    yearly_syn  = pd.read_csv(YEARLY_SYN_PATH)

    print(f"master_real: {master_real.shape}, master_syn: {master_syn.shape}")
    print(f"yearly_real: {yearly_real.shape}, yearly_syn: {yearly_syn.shape}")
    return master_real, master_syn, yearly_real, yearly_syn


# =====================================================
# 1. 数値カラムの分布比較（平均・分散・分位点 + KS検定 + 可視化）
# =====================================================

def plot_hist_overlay(df_real: pd.DataFrame,
                      df_syn: pd.DataFrame,
                      col: str,
                      bins: int = 40) -> None:
    """
    実データと合成データのヒストグラムを重ねて出力する。
    絶対的な指標がない場合でも、視覚的に分布の違いを確認できる。
    """
    ensure_fig_dir()
    plt.figure(figsize=(6, 4))
    real_values = df_real[col].dropna().values
    syn_values  = df_syn[col].dropna().values

    if len(real_values) == 0 or len(syn_values) == 0:
        plt.close()
        return

    plt.hist(real_values, bins=bins, alpha=0.5, density=True, label="real")
    plt.hist(syn_values,  bins=bins, alpha=0.5, density=True, label="synthetic")
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"hist_{col}.png")
    plt.savefig(path)
    plt.close()
    print(f"    - ヒストグラム図を保存: {path}")


def compare_numeric_distributions(df_real: pd.DataFrame,
                                  df_syn: pd.DataFrame,
                                  name: str,
                                  top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    数値カラムごとに実データと合成データの分布差を評価する。

    評価観点:
        ・各数値カラムの分布（平均・分散・形状）がどの程度再現されているか
        ・KS統計量が小さいカラムほど「分布が近い」
        ・評価結果として「◎/○/△/×」の簡易ラベルを付与し、上位の悪いカラムは可視化
    """
    print(f"\n===== [{name}] 数値分布の再現性チェック =====")
    print("評価観点: 各数値カラムの分布（平均・分散・形状）が実データとどれくらい近いかを確認する。")
    print("         KS統計量を用いて定量評価し、さらに分布を重ね描きすることで視覚的にも確認する。\n")

    numeric_cols = df_real.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("※ 数値カラムが存在しません。スキップします。")
        return pd.DataFrame(), pd.DataFrame()

    # 要約統計量（参考用）
    percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    desc_real = df_real[numeric_cols].describe(percentiles=percentiles).T
    desc_syn  = df_syn[numeric_cols].describe(percentiles=percentiles).T
    summary = desc_real.join(desc_syn, lsuffix="_real", rsuffix="_syn")

    # KS検定
    rows: List[Tuple[str, float, float, str]] = []
    for col in numeric_cols:
        r = df_real[col].dropna()
        s = df_syn[col].dropna()
        if len(r) == 0 or len(s) == 0:
            continue
        ks_stat, pval = ks_2samp(r, s)
        label = score_ks(ks_stat)
        rows.append((col, ks_stat, pval, label))
    if not rows:
        print("※ KS検定を実行できる十分なデータがありません。")
        return summary, pd.DataFrame()

    ks_df = pd.DataFrame(rows, columns=["col", "ks_stat", "pval", "eval"])
    ks_df = ks_df.sort_values("ks_stat", ascending=False)

    # 全体としての品質感
    cond_good = ks_df["ks_stat"] < 0.10
    ratio_good = cond_good.mean()
    print("▼ KS統計量による全体評価（数値カラム単位）")
    print(f"    KS < 0.10 のカラム割合: {ratio_good:.1%} → {score_ratio_small_is_good(1 - ratio_good)}")

    # 上位 n カラムを表示
    print("\n▼ KS統計量が大きい（≒分布差が大きい）上位カラム")
    print(ks_df.head(top_n).to_string(index=False))

    # 上位の悪いカラムについてヒストグラムを出力
    print("\n▼ 分布差が大きいカラムのヒストグラムを出力（目視確認用）")
    for col in ks_df.head(top_n)["col"]:
        print(f"  カラム: {col}")
        plot_hist_overlay(df_real, df_syn, col)

    return summary, ks_df


# =====================================================
# 2. カテゴリ分布の比較（業種・市場区分など + 可視化）
# =====================================================

def plot_category_bar(freq_real: pd.Series,
                      freq_syn: pd.Series,
                      col: str) -> None:
    """
    カテゴリごとの構成比を棒グラフで可視化する。
    """
    ensure_fig_dir()
    categories = sorted(set(freq_real.index) | set(freq_syn.index))
    x = np.arange(len(categories))
    width = 0.4

    real_vals = freq_real.reindex(categories, fill_value=0.0).values
    syn_vals  = freq_syn.reindex(categories, fill_value=0.0).values

    plt.figure(figsize=(max(6, len(categories) * 0.4), 4))
    plt.bar(x - width/2, real_vals, width, label="real")
    plt.bar(x + width/2, syn_vals,  width, label="synthetic")
    plt.xticks(x, categories, rotation=90)
    plt.ylabel("ratio")
    plt.title(f"Category distribution: {col}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"catdist_{col}.png")
    plt.savefig(path)
    plt.close()
    print(f"    - カテゴリ分布図を保存: {path}")


def compare_categorical_distribution(df_real: pd.DataFrame,
                                     df_syn: pd.DataFrame,
                                     col: str,
                                     name: str) -> Tuple[pd.DataFrame, float]:
    """
    カテゴリカラムの分布を比較する。

    評価観点:
        ・業種や市場区分などのカテゴリ構成比が実データと似ているか
        ・JS距離が小さいほどカテゴリ構成が似ている（定量評価）
        ・棒グラフによる分布比較（視覚評価）
    """
    print(f"\n===== [{name}] カテゴリ分布の再現性チェック: {col} =====")
    print("評価観点: カテゴリ（業種・区分など）の構成比が、実データと合成データで近いかを確認する。")
    print("         Jensen-Shannon距離を用いて定量評価し、棒グラフで視覚的にも確認する。\n")

    freq_real = df_real[col].value_counts(normalize=True)
    freq_syn  = df_syn[col].value_counts(normalize=True)

    all_idx = sorted(set(freq_real.index) | set(freq_syn.index))
    p = freq_real.reindex(all_idx, fill_value=0.0).values
    q = freq_syn.reindex(all_idx, fill_value=0.0).values

    js_dist = jensenshannon(p, q)
    eval_label = score_js(js_dist)

    dist_df = pd.DataFrame({
        "category": all_idx,
        "real_ratio": p,
        "syn_ratio": q,
        "abs_diff": np.abs(p - q)
    }).sort_values("abs_diff", ascending=False)

    print(f"Jensen-Shannon距離: {js_dist:.4f} → {eval_label}")
    print("▼ 構成比の差が大きいカテゴリ上位")
    print(dist_df.head(10).to_string(index=False))

    print("\n▼ カテゴリ分布の棒グラフを出力（目視確認用）")
    plot_category_bar(freq_real, freq_syn, col)

    return dist_df, js_dist


# =====================================================
# 3. 相関構造の比較（多変量の関係性 + ヒートマップ）
# =====================================================

def plot_corr_heatmap(corr: pd.DataFrame, title: str, fname: str) -> None:
    """
    相関行列をヒートマップとして保存する。
    """
    ensure_fig_dir()
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path)
    plt.close()
    print(f"    - 相関ヒートマップを保存: {path}")


def correlation_diff(df_real: pd.DataFrame,
                     df_syn: pd.DataFrame,
                     name: str) -> Optional[pd.DataFrame]:
    """
    数値カラム間の相関行列を比較する。

    評価観点:
        ・「カラム同士の関係性」（相関構造）がどの程度再現されているか
        ・平均絶対差分が小さいほど構造が似ている
        ・実・合成・差分のヒートマップを見比べることで、どの関係性が崩れているかを把握
    """
    print(f"\n===== [{name}] 相関構造の再現性チェック =====")
    print("評価観点: 財務指標など、カラム同士の関係性（相関）がどれくらい再現されているかを確認する。")
    print("         相関行列の差分の平均が小さいほど、関係性が保たれていると解釈できる。\n")

    numeric_cols = df_real.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        print("※ 数値カラムが存在しません。スキップします。")
        return None

    corr_real = df_real[numeric_cols].corr()
    corr_syn  = df_syn[numeric_cols].corr()

    diff = (corr_real - corr_syn).abs()
    # 上三角部分の平均（対角除く）
    mask = np.triu(np.ones_like(diff, dtype=bool), k=1)
    mean_abs_diff = diff.where(mask).stack().mean()
    label = score_corr_mean_abs_diff(mean_abs_diff)

    print(f"平均絶対相関差分: {mean_abs_diff:.4f} → {label}")

    print("\n▼ 相関ヒートマップを出力（real / synthetic / abs diff）")
    plot_corr_heatmap(corr_real, f"{name} corr (real)", f"corr_{name}_real.png")
    plot_corr_heatmap(corr_syn,  f"{name} corr (synthetic)", f"corr_{name}_syn.png")
    plot_corr_heatmap(diff,      f"{name} corr abs diff", f"corr_{name}_diff.png")

    return diff


# =====================================================
# 4. 時系列の成長率・パターン評価（3年分 + 可視化）
# =====================================================

def add_growth(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    指定カラムについて、コード単位で年次成長率を計算して新カラムを追加する。
    """
    df = df.sort_values([CODE_COL, YEAR_COL])
    df[value_col + "_growth"] = df.groupby(CODE_COL)[value_col].pct_change()
    return df


def plot_time_series_examples(yearly_real: pd.DataFrame,
                              yearly_syn: pd.DataFrame,
                              value_col: str,
                              n_examples: int = 5) -> None:
    """
    代表的なコードをいくつかピックアップし、YEAR×値の推移を折れ線グラフで比較する。
    「増加/減少/凸凹」などのパターンを目視で確認する。
    """
    ensure_fig_dir()
    codes = yearly_real[CODE_COL].dropna().unique()
    if len(codes) == 0:
        return

    examples = np.random.choice(codes, size=min(n_examples, len(codes)), replace=False)
    print(f"    - 時系列サンプルとして可視化するコード: {examples}")

    for code in examples:
        r = yearly_real[yearly_real[CODE_COL] == code].sort_values(YEAR_COL)
        s = yearly_syn[yearly_syn[CODE_COL] == code].sort_values(YEAR_COL)
        if len(r) == 0 or len(s) == 0:
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(r[YEAR_COL], r[value_col], marker="o", label="real")
        plt.plot(s[YEAR_COL], s[value_col], marker="o", label="synthetic")
        plt.title(f"{value_col} time series (コード={code})")
        plt.xlabel(YEAR_COL)
        plt.ylabel(value_col)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(FIG_DIR, f"timeseries_{value_col}_code_{code}.png")
        plt.savefig(path)
        plt.close()
        print(f"      - 時系列図を保存: {path}")


def evaluate_time_series(yearly_real: pd.DataFrame,
                         yearly_syn: pd.DataFrame,
                         value_col: str = VALUE_COL_MAIN) -> None:
    """
    時系列（YEAR）の推移パターンを評価する。

    評価観点:
        ・年次成長率の分布が似ているか（KS検定）
        ・3年分の形状パターン（増加・減少・その他）の構成比が似ているか
        ・代表的なコードの折れ線グラフを見比べて、レベルや変動の自然さを視覚的に確認
    """
    print(f"\n===== [yearly] 時系列傾向の再現性チェック ({value_col}) =====")
    print("評価観点: 企業ごとの年次推移が、成長率分布やパターン（増加・減少など）の観点から再現されているかを確認する。")
    print("         ・成長率分布が似ているか（KS統計量）")
    print("         ・パターン分類（増加/減少/その他）の構成比が似ているか")
    print("         ・サンプル企業の折れ線グラフでレベルと形状を目視確認する\n")

    if value_col not in yearly_real.columns or value_col not in yearly_syn.columns:
        print(f"※ {value_col} が存在しないためスキップします。")
        return
    if YEAR_COL not in yearly_real.columns or YEAR_COL not in yearly_syn.columns:
        print(f"※ {YEAR_COL} が存在しないためスキップします。")
        return

    yearly_real_g = add_growth(yearly_real.copy(), value_col)
    yearly_syn_g  = add_growth(yearly_syn.copy(),  value_col)

    # 成長率分布比較
    r = yearly_real_g[value_col + "_growth"].replace([np.inf, -np.inf], np.nan).dropna()
    s = yearly_syn_g[value_col + "_growth"].replace([np.inf, -np.inf], np.nan).dropna()

    if len(r) > 0 and len(s) > 0:
        ks_stat, pval = ks_2samp(r, s)
        label = score_ks(ks_stat)
        print(f"成長率分布のKS統計量: {ks_stat:.4f}, p値: {pval:.4g} → {label}")
    else:
        print("※ 成長率の有効データが不足しています。")

    # 成長率ヒストグラム（目視用）
    print("\n▼ 成長率分布のヒストグラムを出力（目視確認用）")
    plot_hist_overlay(yearly_real_g, yearly_syn_g, value_col + "_growth", bins=40)

    # パターン分類
    def classify_pattern(series: pd.Series) -> str:
        s_val = series.values
        if len(s_val) < 3:
            return "short"
        inc = (s_val[2] > s_val[1] > s_val[0])
        dec = (s_val[2] < s_val[1] < s_val[0])
        if inc:
            return "increasing"
        elif dec:
            return "decreasing"
        else:
            return "other"

    pat_real = yearly_real.sort_values([CODE_COL, YEAR_COL]) \
                          .groupby(CODE_COL)[value_col].apply(classify_pattern)
    pat_syn  = yearly_syn.sort_values([CODE_COL, YEAR_COL]) \
                         .groupby(CODE_COL)[value_col].apply(classify_pattern)

    dist_real = pat_real.value_counts(normalize=True)
    dist_syn  = pat_syn.value_counts(normalize=True)

    print("\n▼ パターン構成比（実データ）")
    print(dist_real.to_string())
    print("\n▼ パターン構成比（合成データ）")
    print(dist_syn.to_string())

    # 時系列のサンプルプロット
    print("\n▼ サンプル企業の時系列推移を出力（目視確認用）")
    plot_time_series_examples(yearly_real, yearly_syn, value_col, n_examples=5)


# =====================================================
# 5. 業種別の統計・傾向評価（親子結合 + 可視化）
# =====================================================

def evaluate_industry_profile(master_real: pd.DataFrame,
                              master_syn: pd.DataFrame,
                              yearly_real: pd.DataFrame,
                              yearly_syn: pd.DataFrame,
                              industry_col: str = INDUSTRY_COL,
                              value_col: str = VALUE_COL_MAIN) -> Optional[pd.DataFrame]:
    """
    業種別の財務プロファイルを比較する。

    評価観点:
        ・業種ごとの平均・分散・中央値などの統計量が似ているか
        ・業種ごとの売上レベルやばらつきが大きく異ならないか
        ・箱ひげ図を用いて業種 × 値の分布を視覚的に比較
    """
    print(f"\n===== [industry] 業種別プロファイルの再現性チェック ({value_col}) =====")
    print("評価観点: 業種ごとに売上高などの水準・ばらつきが実データと合成データで似ているかを確認する。")
    print("         業種別の分布が大きくずれている場合、業種構造の保持に課題があると判断できる。\n")

    if industry_col not in master_real.columns or industry_col not in master_syn.columns:
        print(f"※ 親テーブルに {industry_col} がないためスキップします。")
        return None

    # 親子結合
    yearly_real_join = yearly_real.merge(
        master_real[[CODE_COL, industry_col]], on=CODE_COL, how="left"
    )
    yearly_syn_join  = yearly_syn.merge(
        master_syn[[CODE_COL, industry_col]], on=CODE_COL, how="left"
    )

    if value_col not in yearly_real_join.columns or value_col not in yearly_syn_join.columns:
        print(f"※ {value_col} が存在しないためスキップします。")
        return None

    def industry_stats(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby(industry_col)[value_col].agg(["mean", "std", "median", "count"])

    real_stats = industry_stats(yearly_real_join)
    syn_stats  = industry_stats(yearly_syn_join)

    diff_ind = real_stats.join(syn_stats, lsuffix="_real", rsuffix="_syn")

    print("▼ 業種別の統計量比較（例として先頭5業種）")
    print(diff_ind.head().to_string())

    # 箱ひげ図（業種別分布を比較）
    ensure_fig_dir()
    yearly_real_join["type"] = "real"
    yearly_syn_join["type"]  = "synthetic"
    plot_df = pd.concat([yearly_real_join[[industry_col, value_col, "type"]],
                         yearly_syn_join[[industry_col, value_col, "type"]]],
                        ignore_index=True)

    plt.figure(figsize=(max(8, len(plot_df[industry_col].unique()) * 0.6), 4))
    # boxplotだけを使う（ライブラリに依存しないやり方：グループごとに描画しても良いが簡易にpandasのboxplotでもOK）
    # ここではMatplotlibでシンプルに描くために少しトリックを使う
    # （GitHub共有前提なので必要以上に凝らない）
    cats = sorted(plot_df[industry_col].unique())
    x_positions = np.arange(len(cats))
    width = 0.35

    for i, t in enumerate(["real", "synthetic"]):
        data = []
        positions = []
        for j, cat in enumerate(cats):
            vals = plot_df[(plot_df[industry_col] == cat) & (plot_df["type"] == t)][value_col].dropna()
            if len(vals) == 0:
                continue
            data.append(vals)
            if t == "real":
                positions.append(j - width/2)
            else:
                positions.append(j + width/2)
        if data:
            bp = plt.boxplot(
                data,
                positions=positions,
                widths=width*0.9,
                patch_artist=True,
                manage_ticks=False,
                labels=[""] * len(data),
            )
            for patch in bp["boxes"]:
                patch.set_alpha(0.5)
            # 色指定なし（デフォルト色を利用）

    plt.xticks(x_positions, cats, rotation=90)
    plt.ylabel(value_col)
    plt.title(f"{value_col} by {industry_col} (real vs synthetic)")
    # 簡易凡例
    plt.plot([], [], label="real")       # ダミー
    plt.plot([], [], label="synthetic")  # ダミー
    plt.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"box_{value_col}_by_{industry_col}.png")
    plt.savefig(path)
    plt.close()
    print(f"▼ 業種別箱ひげ図を保存: {path}")

    return diff_ind


# =====================================================
# 6. 妥当性チェック（レンジ・会計恒等式・外れ値）
# =====================================================

def check_value_ranges(df: pd.DataFrame,
                       rules: Dict[str, Tuple[Optional[float], Optional[float]]],
                       name: str) -> Dict[str, int]:
    """
    カラムごとにレンジルールを適用し、違反件数を集計する。

    評価観点:
        ・明らかにおかしな値（負の売上、高すぎる比率など）がどの程度存在するか
        ・カラムごとの違反率に対して、閾値を用いた評価ラベルを付与
    """
    print(f"\n===== [{name}] 値の範囲チェック =====")
    print("評価観点: 各カラムにありえない値（例: 負の売上、異常な比率）が含まれていないかを確認する。")
    print("         違反件数・違反率に基づき、どのカラムに問題が集中しているかを把握する。\n")

    violations: Dict[str, int] = {}
    n = len(df)
    for col, (min_v, max_v) in rules.items():
        if col not in df.columns:
            continue
        series = df[col]
        cond = pd.Series(False, index=series.index)
        if min_v is not None:
            cond |= series < min_v
        if max_v is not None:
            cond |= series > max_v
        cnt = cond.sum()
        violations[col] = cnt
        ratio = cnt / n if n > 0 else 0.0
        label = score_ratio_small_is_good(ratio)
        print(f"  カラム: {col} / 違反件数: {cnt} / 違反率: {ratio:.4%} → {label}")

    return violations


def check_balance_constraint(df: pd.DataFrame,
                             name: str,
                             total_col: str = TOTAL_ASSET_COL,
                             debt_col: str = DEBT_COL,
                             equity_col: str = EQUITY_COL,
                             tol_ratio: float = 0.01,
                             tol_abs: float = 1e6) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    会計恒等式: 総資産 ≒ 負債 + 純資産 をチェックする。

    評価観点:
        ・財務諸表の基本的な整合性が保たれているか
        ・恒等式からの乖離が大きいレコードが多いかどうか
        ・違反率に対して評価ラベルを付与
    """
    print(f"\n===== [{name}] 会計恒等式チェック =====")
    print("評価観点: 総資産 ≒ 負債 + 純資産 の整合性が保たれているかを確認する。")
    print("         一定の誤差（総資産の1% + 定数）を超える違反が多いと、会計的に不自然と判断できる。\n")

    for c in [total_col, debt_col, equity_col]:
        if c not in df.columns:
            print(f"※ {c} が存在しないため会計恒等式チェックをスキップします。")
            return None, None

    diff = df[total_col] - (df[debt_col] + df[equity_col])
    tol = df[total_col].abs() * tol_ratio + tol_abs
    violation = (diff.abs() > tol)

    total = len(df)
    cnt = violation.sum()
    ratio = cnt / total if total > 0 else 0.0
    label = score_ratio_small_is_good(ratio)

    print(f"  会計恒等式違反件数: {cnt} / {total} （{ratio:.4%}） → {label}")

    return diff, violation


def outlier_rate(df: pd.DataFrame, col: str, method: str = "iqr") -> float:
    """
    外れ値率を計算する。

    評価観点:
        ・外れ値の頻度が実データと極端に異ならないか
        ・「外れ値が多すぎる」または「極端に少なすぎる」場合、
          分布の裾や極端値の扱いに課題がある可能性
    """
    x = df[col].dropna()
    if len(x) == 0:
        return np.nan
    if method == "iqr":
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        return ((x < lower) | (x > upper)).mean()
    else:
        z = (x - x.mean()) / x.std(ddof=0)
        return (z.abs() > 3).mean()


def compare_outlier_rates(yearly_real: pd.DataFrame,
                          yearly_syn: pd.DataFrame,
                          cols: List[str]) -> None:
    """
    外れ値率を実データと合成データで比較する。

    評価観点:
        ・外れ値率が実データと大きく乖離していないか
        ・乖離が大きい場合、極端値の扱いや裾の形状に問題がある可能性
    """
    print("\n===== [yearly] 外れ値率の比較 =====")
    print("評価観点: 外れ値の出現頻度が実データと合成データで大きく乖離していないかを確認する。")
    print("         外れ値が多すぎる/少なすぎる場合、分布の裾や極端値の扱いに課題があると判断できる。\n")

    for col in cols:
        if col not in yearly_real.columns or col not in yearly_syn.columns:
            print(f"※ {col} が存在しないためスキップします。")
            continue
        r_rate = outlier_rate(yearly_real, col)
        s_rate = outlier_rate(yearly_syn,  col)
        if np.isnan(r_rate) or np.isnan(s_rate):
            print(f"  カラム: {col} / 有効データ不足のためスキップ")
            continue
        diff = abs(r_rate - s_rate)
        # ここでは「外れ値率の差」が大きいかどうかを簡易評価
        if diff < 0.01:
            label = "◎ 外れ値率はほぼ同じ"
        elif diff < 0.03:
            label = "○ 多少の差はあるが許容範囲"
        elif diff < 0.07:
            label = "△ 差がやや大きい（要確認）"
        else:
            label = "× 差が大きい（極端値の扱いに課題）"

        print(f"  カラム: {col}")
        print(f"    実データ 外れ値率: {r_rate:.4%}")
        print(f"    合成データ外れ値率: {s_rate:.4%}")
        print(f"    差: {diff:.4%} → {label}")


# =====================================================
# 7. 全評価をまとめて実行するメイン関数
# =====================================================

def run_all_evaluations(master_real: pd.DataFrame,
                        master_syn: pd.DataFrame,
                        yearly_real: pd.DataFrame,
                        yearly_syn: pd.DataFrame) -> None:
    """
    一連の評価をまとめて実行するメイン関数。
    """
    print("######################################")
    print("# 合成データ品質評価（実行開始）")
    print("######################################\n")

    # --- 1. 統計的再現性（数値分布） ---
    compare_numeric_distributions(master_real, master_syn, "master")
    compare_numeric_distributions(yearly_real, yearly_syn, "yearly")

    # --- 2. カテゴリ分布（例: 業種分類 / 市場・商品区分） ---
    for col in [INDUSTRY_COL, MARKET_COL]:
        if col in master_real.columns and col in master_syn.columns:
            compare_categorical_distribution(master_real, master_syn, col, "master")
        else:
            print(f"\n※ カテゴリカラム {col} が親テーブルに見つからないためスキップします。")

    # --- 3. 相関構造 ---
    correlation_diff(yearly_real, yearly_syn, "yearly")

    # --- 4. 時系列傾向 ---
    if YEAR_COL in yearly_real.columns and YEAR_COL in yearly_syn.columns:
        evaluate_time_series(yearly_real, yearly_syn, value_col=VALUE_COL_MAIN)
    else:
        print(f"\n※ {YEAR_COL} カラムが存在しないため時系列評価をスキップします。")

    # --- 5. 業種別プロファイル ---
    evaluate_industry_profile(master_real, master_syn, yearly_real, yearly_syn,
                              industry_col=INDUSTRY_COL, value_col=VALUE_COL_MAIN)

    # --- 6. 妥当性チェック ---
    check_value_ranges(yearly_syn, RANGE_RULES_YEARLY, "yearly_synthetic")

    check_balance_constraint(yearly_syn, "yearly_synthetic",
                             total_col=TOTAL_ASSET_COL,
                             debt_col=DEBT_COL,
                             equity_col=EQUITY_COL)

    compare_outlier_rates(yearly_real, yearly_syn, cols=OUTLIER_COLS)

    print("\n######################################")
    print("# 合成データ品質評価（実行完了）")
    print("# ・ログの数値と◎/○/△/×のコメントで定量的な良し悪しを把握")
    print("# ・figs/ 以下の画像で分布・相関・時系列・業種別の視覚的な違いを確認")
    print("######################################")


# =====================================================
# 8. スクリプトとして実行されたときのエントリポイント
# =====================================================

if __name__ == "__main__":
    # 1. データ読み込み
    master_real_df, master_syn_df, yearly_real_df, yearly_syn_df = load_data()

    # 2. 評価実行
    run_all_evaluations(master_real_df, master_syn_df, yearly_real_df, yearly_syn_df)
