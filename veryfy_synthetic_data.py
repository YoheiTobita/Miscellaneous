import pandas as pd


# 3. 前処理・整合性チェック
# 3.1 スキーマ & 型の一致
# 実データ・合成データで
# カラム名集合
# 各カラムの dtype（int/float/object…）
# が一致しているかを確認。

# 仮：ファイル名は適宜変更してください
master_real = pd.read_csv("master_real.csv")
master_syn  = pd.read_csv("master_syn.csv")
yearly_real = pd.read_csv("yearly_real.csv")
yearly_syn  = pd.read_csv("yearly_syn.csv")

def compare_schema(df_real, df_syn, name):
    print(f"=== {name} schema ===")
    print("real dtypes:")
    print(df_real.dtypes)
    print("\nsynthetic dtypes:")
    print(df_syn.dtypes)
    print("\nMissing in synthetic:", set(df_real.columns) - set(df_syn.columns))
    print("Extra in synthetic:", set(df_syn.columns) - set(df_real.columns))

compare_schema(master_real, master_syn, "master")
compare_schema(yearly_real, yearly_syn, "yearly")

# 3.2 キー制約・親子関係のチェック
# 親：コード が ユニーク か
# 子：(コード, YEAR) が 一意 か
# 子の コード が必ず親に存在するか（referential integrity）
def check_keys(master, yearly, name):
    print(f"=== key check ({name}) ===")
    # 親キー
    dup_master = master['コード'].duplicated().sum()
    print("master duplicate コード:", dup_master)

    # 子キー
    dup_yearly = yearly[['コード', 'YEAR']].duplicated().sum()
    print("yearly duplicate (コード, YEAR):", dup_yearly)

    # 参照整合性
    missing_parents = (~yearly['コード'].isin(master['コード'])).sum()
    print("yearly rows with コード not in master:", missing_parents)

check_keys(master_real, yearly_real, "real")
check_keys(master_syn, yearly_syn, "synthetic")

4. 統計的再現性（全特徴量）
# ポイント：
# 「主要項目だけ」ではなく、全数値カラムをループで評価する
# 実データをゴールドスタンダードとして、差分を指標化する
# 4.1 数値カラムの単変量分布比較
# 平均・標準偏差・分位点（5%, 25%, 50%, 75%, 95%）
# 分布距離（例：Kolmogorov-Smirnov, Wasserstein distance）
import numpy as np
from scipy.stats import ks_2samp

def numeric_distribution_summary(df, numeric_cols):
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    desc = df[numeric_cols].describe(percentiles=q).T
    return desc

def compare_numeric_distributions(df_real, df_syn, name):
    numeric_cols = df_real.select_dtypes(include=[np.number]).columns
    print(f"=== numeric distribution ({name}) ===")
    summary_real = numeric_distribution_summary(df_real, numeric_cols)
    summary_syn  = numeric_distribution_summary(df_syn, numeric_cols)

    # サマリ
    # print(summary_real.join(summary_syn, lsuffix="_real", rsuffix="_syn"))

    # KS距離など
    rows = []
    for col in numeric_cols:
        r = df_real[col].dropna()
        s = df_syn[col].dropna()
        if len(r) > 0 and len(s) > 0:
            ks_stat, pval = ks_2samp(r, s)
            rows.append((col, ks_stat, pval))
    ks_df = pd.DataFrame(rows, columns=["col", "ks_stat", "pval"]).sort_values("ks_stat", ascending=False)
    return ks_df

ks_master = compare_numeric_distributions(master_real, master_syn, "master")
ks_yearly = compare_numeric_distributions(yearly_real, yearly_syn, "yearly")
print(ks_master.head())
print(ks_yearly.head())

# 4.2 カテゴリカラムの分布比較
# 業種分類, 市場・商品区分 など
# カテゴリ別のシェア（%）を比較
# チャイ二乗距離や Jensen-Shannon 距離等
from scipy.spatial.distance import jensenshannon

def compare_categorical_distribution(df_real, df_syn, col):
    freq_real = df_real[col].value_counts(normalize=True)
    freq_syn  = df_syn[col].value_counts(normalize=True)

    # index を揃える
    all_idx = sorted(set(freq_real.index) | set(freq_syn.index))
    p = freq_real.reindex(all_idx, fill_value=0.0).values
    q = freq_syn.reindex(all_idx, fill_value=0.0).values

    js_div = jensenshannon(p, q)  # JS距離
    return all_idx, p, q, js_div

for col in ['業種分類', '市場・商品区分']:
    if col in master_real.columns:
        idx, p, q, js = compare_categorical_distribution(master_real, master_syn, col)
        print(col, "JS distance:", js)

# 4.3 相関構造（多変量）の比較
# 会計データは相関構造が重要（負債が増えれば総資産も増える等）
# 実データと合成データの 相関行列の差分（ノルム） を見る
def correlation_diff(df_real, df_syn, name):
    numeric_cols = df_real.select_dtypes(include=[np.number]).columns
    corr_real = df_real[numeric_cols].corr()
    corr_syn  = df_syn[numeric_cols].corr()
    # 差の Frobenius ノルム
    diff = (corr_real - corr_syn).abs()
    mean_abs_diff = diff.values[np.triu_indices_from(diff, k=1)].mean()
    print(f"{name} mean abs corr diff:", mean_abs_diff)
    return diff

corr_diff_yearly = correlation_diff(yearly_real, yearly_syn, "yearly")

# 5. 視覚的な分布比較（ヒストグラム・箱ひげ・散布図）
# ここでは 「全カラムループ」＋「代表例を可視化」 の組み合わせがおすすめです。
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hist_overlay(df_real, df_syn, col, bins=30):
    plt.figure(figsize=(6,4))
    sns.histplot(df_real[col], stat="density", bins=bins, label="real", alpha=0.5)
    sns.histplot(df_syn[col],  stat="density", bins=bins, label="synthetic", alpha=0.5)
    plt.title(f"Histogram: {col}")
    plt.legend()
    plt.tight_layout()

def plot_box_side_by_side(df_real, df_syn, col):
    plt.figure(figsize=(4,4))
    tmp = pd.DataFrame({
        col: pd.concat([df_real[col], df_syn[col]], ignore_index=True),
        "type": ["real"] * len(df_real) + ["synthetic"] * len(df_syn)
    })
    sns.boxplot(x="type", y=col, data=tmp)
    plt.title(f"Boxplot: {col}")
    plt.tight_layout()

# 例：売上高、営業利益、総資産、従業員数などの代表カラムを可視化
# 散布図：売上高 vs 営業利益, 総資産 vs 負債, 従業員数 vs 売上高 などを real/syn で色分け
def plot_scatter(df_real, df_syn, x, y):
    plt.figure(figsize=(6,5))
    plt.scatter(df_real[x], df_real[y], alpha=0.3, label="real")
    plt.scatter(df_syn[x], df_syn[y], alpha=0.3, label="synthetic")
    plt.xlabel(x); plt.ylabel(y)
    plt.legend()
    plt.title(f"Scatter: {x} vs {y}")
    plt.tight_layout()

# 6. 時系列傾向の評価（3年分）
# 6.1 単社ごとのレベル&成長率分布
# 各 コード について
# 売上高の年次成長率（(t / t-1 - 1)）、営業利益率などを算出
# 実 vs 合成で
# 成長率の分布
# 利益率の分布
# 売上高の推移パターン（増加・減少・横ばいの比率）
def add_growth(df, col_value):
    df = df.sort_values(['コード', 'YEAR'])
    df[col_value + "_growth"] = df.groupby('コード')[col_value].pct_change()
    return df

yearly_real = add_growth(yearly_real, '売上高')
yearly_syn  = add_growth(yearly_syn, '売上高')

# 成長率分布比較（KSなど）
ks_growth = ks_2samp(
    yearly_real['売上高_growth'].dropna(),
    yearly_syn['売上高_growth'].dropna()
)
print("売上高_growth KS:", ks_growth)

# ヒストグラムで比較
plot_hist_overlay(yearly_real.dropna(), yearly_syn.dropna(), '売上高_growth', bins=30)

# 6.2 パターン分類（増加・減少・凸凹）
# 3年分（YEARが3つ）なら、例えば：
# 売上高 グラフ形状を
# monotonically increasing / decreasing / その他
# と分類し、パターンの構成比を比較
def classify_pattern(series):
    # series: YEAR昇順の売上高
    s = series.values
    if len(s) < 3:
        return "short"
    inc = (s[2] > s[1] > s[0])
    dec = (s[2] < s[1] < s[0])
    if inc:
        return "increasing"
    elif dec:
        return "decreasing"
    else:
        return "other"

def pattern_distribution(df):
    patterns = df.sort_values(['コード', 'YEAR']).groupby('コード')['売上高'].apply(classify_pattern)
    return patterns.value_counts(normalize=True)

print("real pattern dist:\n", pattern_distribution(yearly_real))
print("syn  pattern dist:\n", pattern_distribution(yearly_syn))

# 6.3 時系列の視覚的確認
# 代表的な数社をサンプリングし、売上高 / 営業利益 の YEAR 推移を折れ線で real vs syn を比較。
def plot_time_series_example(df_real, df_syn, code, col):
    fig, ax = plt.subplots(figsize=(6,4))
    r = df_real[df_real['コード'] == code].sort_values('YEAR')
    s = df_syn[df_syn['コード'] == code].sort_values('YEAR')
    ax.plot(r['YEAR'], r[col], marker='o', label='real')
    ax.plot(s['YEAR'], s[col], marker='o', label='synthetic')
    ax.set_title(f"{col} time series (コード={code})")
    ax.legend()
    plt.tight_layout()

7. 業種別の傾向保持
7.1 業種別の基本統計
親テーブル 業種分類 × 子テーブルの財務指標
例：業種分類 × 売上高 の平均, 分散, 利益率 etc.
実 vs 合成で「業種プロファイル」が似ているかを見る

# 親子結合
yearly_real_join = yearly_real.merge(master_real[['コード', '業種分類']], on='コード', how='left')
yearly_syn_join  = yearly_syn.merge(master_syn[['コード', '業種分類']],   on='コード', how='left')

def industry_stats(df, value_col):
    return df.groupby('業種分類')[value_col].agg(['mean', 'std', 'median', 'count'])

real_ind_stats = industry_stats(yearly_real_join, '売上高')
syn_ind_stats  = industry_stats(yearly_syn_join,  '売上高')

# 差分
diff_ind = real_ind_stats.join(syn_ind_stats, lsuffix="_real", rsuffix="_syn")
print(diff_ind.head())

7.2 業種別分布の比較（視覚）
Violin plot / boxplot で 業種分類 × 売上高 を real/syn 並べる
散布図で 業種分類 を色分け

def plot_industry_box(df_real, df_syn, col):
    df_real_ = df_real.copy()
    df_real_['type'] = 'real'
    df_syn_  = df_syn.copy()
    df_syn_['type'] = 'synthetic'
    tmp = pd.concat([df_real_, df_syn_], ignore_index=True)

    plt.figure(figsize=(10,4))
    sns.boxplot(x='業種分類', y=col, hue='type', data=tmp)
    plt.xticks(rotation=90)
    plt.title(f"{col} by industry (real vs synthetic)")
    plt.tight_layout()

8. 妥当性チェック（値の範囲・制約）
8.1 単項目のレンジチェック

例（財務データ系）

売上高, 利益, 総資産, 負債, 純資産 ≥ 0 （一部マイナス許容するかは業種次第）

比率（ROE, 利益率など）は常識的な範囲（例：[-3, 3]）に収まるか

# 例: 汎用的なレンジルールを dict で定義
range_rules = {
    '売上高': (0, None),
    '営業利益': (None, None),  # 赤字を許容するなら下限なし
    '総資産': (0, None),
    '負債': (0, None),
    '純資産': (None, None),   # マイナス純資産もありうる
}

def check_ranges(df, rules):
    violations = {}
    for col, (min_v, max_v) in rules.items():
        if col not in df.columns:
            continue
        series = df[col]
        cond = pd.Series(False, index=series.index)
        if min_v is not None:
            cond |= series < min_v
        if max_v is not None:
            cond |= series > max_v
        violations[col] = cond.sum()
    return violations

print("range violations (synthetic):")
print(check_ranges(yearly_syn, range_rules))

8.2 カラム間制約（会計恒等式）

例：

総資産 ≈ 負債 + 純資産（誤差は丸めや分類の関係で許容）

営業利益 ≤ 経常利益 ≤ 税引前当期純利益 のような階層構造（科目構造に応じて）

合成データでこの制約が大きく崩れていないか

def check_balance(df):
    # カラム名は実データに合わせて変更してください
    if not set(['総資産', '負債', '純資産']).issubset(df.columns):
        return None
    diff = df['総資産'] - (df['負債'] + df['純資産'])
    # 許容誤差（例：総資産の1% + 定数）
    tol = df['総資産'].abs() * 0.01 + 1e6
    violation = (diff.abs() > tol)
    print("balance violations:", violation.sum(), "/", len(df))
    return diff, violation

diff_syn, viol_syn = check_balance(yearly_syn)

8.3 外れ値・極端なパターン

Zスコアや IQR で外れ値を検出し、
実 vs 合成 で「極端値の頻度」が近いか、異常に多く/少なくないかを比較

def outlier_rate(df, col, method="iqr"):
    x = df[col].dropna()
    if method == "iqr":
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        return ((x < lower) | (x > upper)).mean()
    else:
        z = (x - x.mean()) / x.std()
        return (z.abs() > 3).mean()

for col in ['売上高', '営業利益', '総資産']:
    if col in yearly_real.columns:
        print(col, "outlier rate real:", outlier_rate(yearly_real, col))
        print(col, "outlier rate syn :", outlier_rate(yearly_syn,  col))

9. 追加で入れておきたい観点（AI視点）
9.1 欠損パターンの再現性

「どのカラムにどの程度 NA があるか」
-> 実と合成で似ているか

def missing_profile(df):
    return df.isna().mean()

miss_real = missing_profile(yearly_real)
miss_syn  = missing_profile(yearly_syn)
missing_compare = pd.DataFrame({"real": miss_real, "syn": miss_syn})

9.2 クラスタ構造の再現性

数値カラムを標準化して PCA / t-SNE し、クラスター構造を real / syn で重ねて可視化

あるいは、KMeansクラスタリングして cluster ラベルの比率を比較

→ これにより「財務的特徴空間における配置」がどれくらい似ているかを評価できます。

9.3 プライバシー・過学習の簡易チェック

合成データのレコードが実データのレコードと「ほぼ一致」していないか（距離がゼロ or 非常に小さい）を確認

最近傍距離（real ↔ syn）分布を比較し、「ほぼコピー」がないかを見る

10. 修正方針（問題が見つかった場合）
10.1 単純な対処（ポストプロセス）

レンジのクリッピング

例：売上高 < 0 の場合は 0 に丸める、比率は [-3, 3] にクリップなど

会計恒等式の調整

例：総資産 を固定して 負債 と 純資産 を割合でスケーリングし合計を合わせる

業種ごとの再スケーリング

業種分類 ごとに、実データの平均・分散に近づくよう
合成値を線形変換（平均・標準偏差マッチング）

def match_mean_std(s_syn, target_mean, target_std, eps=1e-8):
    s = s_syn.copy()
    cur_mean, cur_std = s.mean(), s.std() + eps
    s = (s - cur_mean) / cur_std * target_std + target_mean
    return s

10.2 生成モデル側の改良

条件付き生成：業種・市場区分・YEAR を入力として生成することで、

業種別の分布差や時系列パターンを自然に制御

制約付き生成 / 損失関数へのペナルティ

総資産 - 負債 - 純資産 の誤差をペナルティに
負の売上・極端な比率などに対してもペナルティ
分布距離を損失に組み込む
代表的なカラムについて、実 vs 合成の分布距離（例えば MMD, Wasserstein）を最小化するよう学習

10.3 評価に基づくフィードバックループ
上記の評価指標（KS distance, 相関差, 制約違反数 etc.）を「スコア」としてまとめる
バージョンごとにスコアを記録 → 改善が見える形にする
最終的には「合格ライン」（例えば：
主な数値カラムの KS statistic < 0.1
会計恒等式の違反率 < 1%
負の売上ゼロ etc.）
を設定し、その閾値を満たすことを目標にする

11. まとめ
すべての数値・カテゴリ特徴量に対して：
単変量分布・カテゴリ分布・相関構造を比較
時系列：
レベルだけでなく、成長率・パターン（増加/減少など）の構成比を比較
業種：
業種分類 を軸に統計量・分布・相関を比較し、業種プロファイルの再現を確認
妥当性：
値のレンジ、会計恒等式、比率、外れ値などをチェック
追加観点：
欠損パターン、クラスタ構造、プライバシー（過学習）まで含めるとより安心
もしよければ次のステップとして：
実際に使う「評価レポートのフォーマット（表・図の構成）」
または「指標を1つのスコアにまとめる設計」
も一緒に設計できます。どちらを先に具体化したいか教えてもらえれば、それ前提でコードとレポート雛形も出します。
