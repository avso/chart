###################
# 必要なライブラリ
# pip install yfinance
# pip install mplfinance
# pip install japanize-matplotlib
###################
import os
from datetime import datetime, timedelta
import dateutil

import numpy as np
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import japanize_matplotlib
japanize_matplotlib.japanize()


####################
# グローバル設定
####################
# データ開始年
start_year = 1980

# ティッカーを指定
# 日本の個別株を取得する場合は証券コードの末尾に「.T」を付与する
tickers = {
    '^N225': '日経平均',
    '^DJI': 'ダウ',
    '^IXIC': 'NASDAQ',
    '^NDX': 'NASDAQ100',
    '^GSPC': 'S&P500',
    'BRK-B': 'バークシャーハサウェイ'
}

# チャートデータ保存フォルダー
CHART_DATA_PATH = 'chart_data/'
# チャート画像保存フォルダー
CHART_PATH = 'chart/'

# チャートスタイルの設定値を作る
yahoo = mpf.make_mpf_style(
    # 基本はyahooの設定値を使う。
    base_mpf_style='yahoo',
    # font.family を matplotlibに設定されている値にする。
    rc={'font.family': plt.rcParams['font.family'][0]},
    # チャートカラー設定
    marketcolors=mpf.make_marketcolors(
        up='#00b060',
        down='#fe3032',
        edge='inherit',
        wick='#606060',
        ohlc='inherit',
        volume={'up': '#4dc790', 'down': '#fd6b6c'},
        vcdopcod=False,  # Volume Color Depends On Price Change On Day
        alpha=0.9,
    ),
)


####################
# チャート生成クラス
####################
class chart:

    def __init__(self):
        self.main()

    def main(self):
        # 取得期間の開始日付を指定
        start = datetime(start_year, 1, 1).strftime('%Y-%m-%d')

        # 取得期間の終了日付の翌日を指定
        end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        for ticker, ticker_name in tickers.items():
            OLD_CHART_PATH = CHART_PATH + str(ticker_name) + '/'

            ####################
            # フォルダ用意
            ####################
            os.makedirs(CHART_PATH + ticker_name, exist_ok=True)

            print(ticker_name + 'のCSVを取得中...')
            ####################
            # 株価データ取得
            ####################
            # 既存のCSVファイルを読み取って最新の日付を取得
            existing_csv_path = CHART_DATA_PATH + ticker_name + '_日足.csv'
            existing_df = None
            if os.path.exists(existing_csv_path):
                existing_df = pd.read_csv(existing_csv_path, index_col='Date', parse_dates=True, header=0)
                start = existing_df.index[-1].strftime('%Y-%m-%d')

            # yfinanceライブラリを用いて指定した条件でデータを取得
            df = yf.Ticker(ticker).history(start=start, end=end, interval='1d')

            # データフレームの開始年を取得
            df_year = np.datetime64(df.index.values[0], 'Y').astype(int) + 1970
            print(str(df_year) + '年以降データを取得できました')

            # 既存CSVが存在する場合
            if existing_df is not None:
                # 既存CSVのデータフレームと取得したデータフレームを結合
                df = existing_df.combine_first(df)
                # 列順が変わってしまうため元の列順へ変更
                df = df[existing_df.columns]

            #####################
            # テクニカル指標
            #####################
            print(ticker_name + 'のテクニカル指標を算出中...')

            # 全期間のテクニカルを計算すると時間かかるので差分＋過去4ヶ月分のデータを用いて
            start_index = (datetime.strptime(start, '%Y-%m-%d') - dateutil.relativedelta.relativedelta(months=4)).strftime('%Y-%m-%d')
            df_subset = df.loc[start_index:end].copy()

            # 移動平均線
            df['SMA5'] = df_subset['Close'].rolling(window=5, min_periods=0).mean()
            df['SMA25'] = df_subset['Close'].rolling(window=25, min_periods=0).mean()
            df['SMA75'] = df_subset['Close'].rolling(window=75, min_periods=0).mean()

            # FTD(フォロースルーデイ)
            # df['RecentVolume'] = df['Volume'].rolling(window=10, min_periods=0).mean().shift(1).fillna(0)
            # df['RecentVolume'] = df['Volume'].rolling(window=5, min_periods=0).mean().shift(1).fillna(0)
            df_subset['RecentVolume'] = df_subset['Volume'].shift(1).fillna(0)
            df['RecentVolume'] = df_subset['RecentVolume']
            df_subset['VolumeMA'] = df_subset['Volume'].rolling(window=50, min_periods=0).mean().fillna(0)
            df['VolumeMA'] = df_subset['VolumeMA']
            # 出来高の条件
            # condition_recent = df['Volume'] > df['RecentVolume']
            # condition_recent = df['Volume'] >= df['RecentVolume'] * 1.05
            # condition_recent = df['Volume'] >= df['RecentVolume'] * 1.10
            condition_recent = df_subset['Volume'] >= df_subset['RecentVolume'] * 1.15
            # condition_recent = df['Volume'] >= df['RecentVolume'] * 1.20
            # condition_vma = df['Volume'] > df['VolumeMA']
            # condition_vma = df['Volume'] >= df['VolumeMA'] * 1.05
            condition_vma = df_subset['Volume'] >= df_subset['VolumeMA'] * 1.15
            # condition_vma = df['Volume'] >= df['VolumeMA'] * 1.20
            # 上昇幅の条件
            # condition_increase = df['Close'] >= df['Close'].shift(1) * 1.0100
            condition_increase = df_subset['Close'] >= df_subset['Close'].shift(1) * 1.0125
            # 底打ちの条件
            condition_rebound = np.array([chart.check_rebound(df_subset, i) for i in range(len(df_subset))])

            # FTD判定
            # df['FTD'] = np.where(condition_recent & condition_increase & condition_rebound & condition_vma, True, False)
            # df['FTD'] = np.where(condition_recent & condition_increase & condition_rebound, True, False)
            # df['FTD'] = np.where(condition_increase & condition_rebound & condition_vma, True, False)
            df_subset['FTD'] = np.where((condition_recent | condition_vma) & condition_increase & condition_rebound, True, False)
            df['FTD'] = df_subset['FTD']
            # FTDの目印の出力位置
            df_subset['FTD_POS'] = np.where(df_subset['FTD'], df_subset['Low'] * 0.995, np.nan)
            df['FTD_POS'] = df_subset['FTD_POS']

            # 逆FTD判定
            # 下落幅の条件
            condition_decrease = df_subset['Close'] <= df_subset['Close'].shift(1) * 0.9900
            # 反落の条件
            condition_pullback = np.array([chart.check_pullback(df_subset, i) for i in range(len(df_subset))])
            # 逆FTD判定
            # df['R_FTD'] = np.where(condition_recent & condition_decrease & condition_pullback & condition_vma, True, False)
            # df['R_FTD'] = np.where(condition_recent & condition_decrease & condition_pullback, True, False)
            # df['R_FTD'] = np.where(condition_decrease & condition_pullback & condition_vma, True, False)
            df_subset['R_FTD'] = np.where((condition_recent | condition_vma) & condition_decrease & condition_pullback, True, False)
            df['R_FTD'] = df_subset['R_FTD']
            df_subset['R_FTD_POS'] = np.where(df_subset['R_FTD'], df_subset['High'] * 1.005, np.nan)
            df['R_FTD_POS'] = df_subset['R_FTD_POS']

            ####################
            # CSV形式で保存
            ####################
            df.to_csv(CHART_DATA_PATH + str(ticker_name) + '_日足.csv', encoding='utf8')

            ####################
            # チャート生成
            ####################
            # 1年ごとに画像出力する
            current_year = datetime.now().year
            for year in range(current_year, df_year - 1, -1):  # 更新分だけチャート出力
                print(str(year) + 'のチャートを生成中...')
                file_name = str(ticker_name) + '_' + str(year) + '_日足.png'
                diff_year = current_year - year

                # 今年分はchart直下に出力する。過去分はフォルダにまとめる
                if diff_year == 0:
                    save_file_path = CHART_PATH + file_name
                    if (os.path.exists(OLD_CHART_PATH + file_name)):
                        os.remove(OLD_CHART_PATH + file_name)
                else:
                    save_file_path = OLD_CHART_PATH + file_name
                    if (os.path.exists(CHART_PATH + file_name)):
                        os.remove(OLD_CHART_PATH + file_name)

                # 表示期間のスタートを指定
                graphStart = (datetime(current_year + 1, 1, 1) - dateutil.relativedelta.relativedelta(years=diff_year + 1)).strftime('%Y-%m-%d')
                graphEnd = (datetime(current_year + 1, 1, 1) - dateutil.relativedelta.relativedelta(years=diff_year)).strftime('%Y-%m-%d')
                # graphStart = (datetime.now() - dateutil.relativedelta.relativedelta(months=6)).strftime('%Y-%m-%d')  # 半年前

                # テクニカル指標の描画
                apd = [
                    mpf.make_addplot(df[graphStart:graphEnd]['SMA5'], panel=0, color='blue', width=1, alpha=0.7),
                    mpf.make_addplot(df[graphStart:graphEnd]['SMA25'], panel=0, color='red', width=1, alpha=0.7),
                    mpf.make_addplot(df[graphStart:graphEnd]['SMA75'], panel=0, color='magenta', width=1, alpha=0.7),
                ]
                if not df[graphStart:graphEnd]['FTD_POS'].dropna().empty:
                    apd.append(mpf.make_addplot(df[graphStart:graphEnd]['FTD_POS'], type='scatter', markersize=120, marker='^', color='blue'))
                if not df[graphStart:graphEnd]['R_FTD_POS'].dropna().empty:
                    apd.append(mpf.make_addplot(df[graphStart:graphEnd]['R_FTD_POS'], type='scatter', markersize=120, marker='v', color='magenta'))

                # yahooファイナンススタイルのチャートを生成して保存
                # mpf.plot(df[graphStart:], addplot=apd_day_ave, type='candle', datetime_format='%m/%d', xrotation=360, tight_layout=False, volume=True, figratio=(19, 9), style='yahoo')
                mpf.plot(
                    df[graphStart:graphEnd],  # 使用するデータフレームを第一引数に指定
                    type='candle',  # グラフ表示の種類
                    style=yahoo,  # 表示スタイル

                    # チャートのサイズの設定
                    figratio=(2, 1),  # 図のサイズをタプルで指定
                    figscale=1.5,  # 図の大きさの倍率を指定、デフォルトは1
                    tight_layout=True,  # 図の端の余白を狭くして最適化するかどうかを指定

                    # 軸の設定
                    datetime_format='%Y/%m/%d',
                    # xlim=('2021-01-01', '2021-02-01'), # X軸の日付の範囲をタプルで指定 指定無しならデータフレームを元に自動で設定される
                    # ylim=(25000, 30000), # Y軸の範囲をタプルで指定 指定無しなら自動で設定される
                    xrotation=45,  # X軸の日付ラベルの回転角度を指定
                    axisoff=False,  # 軸の非表示

                    # データの表示設定
                    volume=True,  # ボリュームの表示
                    show_nontrading=False,  # データがない日付の表示

                    # ラベルの設定
                    title=ticker_name,  # チャートのタイトル
                    ylabel='価格',  # チャートのY軸ラベル
                    ylabel_lower='出来高',  # ボリュームを表示する場合は、ボリュームのグラフのY軸ラベル

                    # 追加プロット
                    addplot=apd,

                    # 保存先
                    savefig=save_file_path
                )

    ####################
    # FTDの底打ち判定関数
    ####################
    @staticmethod
    def check_rebound(df, index):
        # 指定されたインデックスでの日付を取得
        current_date = df.index[index]
        # 1ヶ月前の日付を計算
        start_date = current_date - dateutil.relativedelta.relativedelta(months=1)
        # 直近1ヶ月間のデータを取得
        days = df.loc[start_date:current_date]
        # 最安終値が最後に発生した日付を見つける
        lowest_date = days[days['Low'] == days['Low'].min()].index[-1]
        # 行数の差分を取得
        row_difference = df.index.get_loc(current_date) - df.index.get_loc(lowest_date)
        return row_difference > 3

    ####################
    # 逆FTDの反落判定関数
    ####################
    @staticmethod
    def check_pullback(df, index):
        # 指定されたインデックスでの日付を取得
        current_date = df.index[index]
        # 1ヶ月前の日付を計算
        start_date = current_date - dateutil.relativedelta.relativedelta(months=1)
        # 直近1ヶ月間のデータを取得
        days = df.loc[start_date:current_date]
        # 最高終値が最後に発生した日付を見つける
        highest_date = days[days['High'] == days['High'].max()].index[-1]
        # 行数の差分を取得
        row_difference = df.index.get_loc(current_date) - df.index.get_loc(highest_date)
        return row_difference > 3


# チャート生成実行
chart()
